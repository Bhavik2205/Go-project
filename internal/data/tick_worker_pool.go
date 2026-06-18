package data

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"

	"github.com/Bhavik2205/ML-Bot/internal/marketdata"
	"go.uber.org/zap"
)

var ErrTickWorkerPoolClosed = errors.New("tick worker pool closed")

const (
	DefaultWorkerCount = 16
	WorkerQueueSize    = 10000
)

// TickWorkerPool distributes ticks to workers based on instrument token.
// Order is preserved per instrument because the same token always routes
// to the same worker.
type TickWorkerPool struct {
	workers   []chan marketdata.NormalizedTick
	workerCnt int

	wg sync.WaitGroup

	ctx    context.Context
	cancel context.CancelFunc

	closed atomic.Bool
}

// NewTickWorkerPool creates a worker pool that preserves ordering
// for each instrument token.
func NewTickWorkerPool(
	ctx context.Context,
	workerCount int,
	processor func(context.Context, marketdata.NormalizedTick),
) *TickWorkerPool {

	if workerCount <= 0 {
		workerCount = DefaultWorkerCount
	}

	poolCtx, cancel := context.WithCancel(ctx)

	p := &TickWorkerPool{
		workers:   make([]chan marketdata.NormalizedTick, workerCount),
		workerCnt: workerCount,
		ctx:       poolCtx,
		cancel:    cancel,
	}

	for i := 0; i < workerCount; i++ {

		ch := make(chan marketdata.NormalizedTick, WorkerQueueSize)
		p.workers[i] = ch

		p.wg.Add(1)

		go func(workerID int, ch <-chan marketdata.NormalizedTick) {
			defer p.wg.Done()

			defer func() {
				if r := recover(); r != nil {
					zap.L().Error(
						"tick worker panic recovered",
						zap.Int("worker_id", workerID),
						zap.Any("panic", r),
					)
				}
			}()

			for {
				select {

				case tick, ok := <-ch:
					if !ok {
						return
					}

					processor(poolCtx, tick)

				case <-poolCtx.Done():
					return
				}
			}

		}(i, ch)
	}

	zap.L().Info(
		"tick worker pool started",
		zap.Int("workers", workerCount),
		zap.Int("queue_size", WorkerQueueSize),
	)

	return p
}

// Submit routes a tick to the correct worker.
//
// IMPORTANT:
//
// This is intentionally BLOCKING.
//
// If workers fall behind, upstream publishers will slow down
// instead of losing market data.
//
// Institutional systems prefer backpressure over data loss.
func (p *TickWorkerPool) Submit(
	tick marketdata.NormalizedTick,
) error {

	if p.closed.Load() {
		return ErrTickWorkerPoolClosed
	}

	if p.workerCnt == 0 {
		return nil
	}

	idx := int(tick.InstrumentToken % uint32(p.workerCnt))

	select {

	case p.workers[idx] <- tick:
		return nil

	case <-p.ctx.Done():
		return context.Canceled
	}
}

// QueueDepth returns current queue depth for a worker.
func (p *TickWorkerPool) QueueDepth(workerID int) int {

	if workerID < 0 || workerID >= len(p.workers) {
		return 0
	}

	return len(p.workers[workerID])
}

// TotalQueueDepth returns total queued ticks.
func (p *TickWorkerPool) TotalQueueDepth() int {

	total := 0

	for _, ch := range p.workers {
		total += len(ch)
	}

	return total
}

func (p *TickWorkerPool) TotalCapacity() int {

	total := 0

	for _, ch := range p.workers {
		total += cap(ch)
	}

	return total
}

// Close performs a graceful shutdown.
// It cancels the pool context and waits for all workers to finish.
// It does NOT close the worker channels to avoid send-on-closed-channel panics.
func (p *TickWorkerPool) Close() {

	if !p.closed.CompareAndSwap(false, true) {
		return
	}

	zap.L().Info("stopping tick worker pool")

	p.cancel()

	p.wg.Wait()

	zap.L().Info("tick worker pool stopped")
}
