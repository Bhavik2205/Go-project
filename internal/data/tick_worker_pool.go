package data

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/marketdata"
	"github.com/Bhavik2205/ML-Bot/internal/observability"
	"github.com/Bhavik2205/ML-Bot/internal/ringbuffer"
	"go.uber.org/zap"
)

var ErrTickWorkerPoolClosed = errors.New("tick worker pool closed")

const (
	DefaultWorkerCount   = 16
	DefaultRingBufferCap = 8192 // per-worker ring buffer capacity; must be power of 2
)

// TickWorkerPool distributes ticks to workers based on instrument token.
// Order is preserved per instrument because the same token always routes
// to the same worker (token % workerCount).
//
// Each worker owns a dedicated SPSCRingBuffer.  Submit is non-blocking on the
// ring buffer: if the worker's buffer is full the tick is dropped and the
// drop counter is incremented rather than blocking the caller.  This keeps
// the ingestion hot-path allocation-free and latency-predictable.
type TickWorkerPool struct {
	rings     []*ringbuffer.SPSCRingBuffer
	workerCnt int

	wg sync.WaitGroup

	ctx    context.Context
	cancel context.CancelFunc

	closed atomic.Bool

	// observability ticker — reports per-worker depth every second
	metricStop chan struct{}
}

// NewTickWorkerPool creates a worker pool backed by per-worker ring buffers.
// ringBufCap is the capacity of each ring buffer; it must be a power of 2.
// Pass 0 to use DefaultRingBufferCap.
func NewTickWorkerPool(
	ctx context.Context,
	workerCount int,
	processor func(context.Context, marketdata.NormalizedTick),
) *TickWorkerPool {
	return NewTickWorkerPoolWithCap(ctx, workerCount, DefaultRingBufferCap, processor)
}

// NewTickWorkerPoolWithCap is like NewTickWorkerPool but lets the caller
// specify the per-worker ring buffer capacity (must be a power of 2).
func NewTickWorkerPoolWithCap(
	ctx context.Context,
	workerCount int,
	ringBufCap int,
	processor func(context.Context, marketdata.NormalizedTick),
) *TickWorkerPool {

	if workerCount <= 0 {
		workerCount = DefaultWorkerCount
	}
	if ringBufCap <= 0 {
		ringBufCap = DefaultRingBufferCap
	}

	poolCtx, cancel := context.WithCancel(ctx)

	p := &TickWorkerPool{
		rings:      make([]*ringbuffer.SPSCRingBuffer, workerCount),
		workerCnt:  workerCount,
		ctx:        poolCtx,
		cancel:     cancel,
		metricStop: make(chan struct{}),
	}

	for i := 0; i < workerCount; i++ {
		rb, err := ringbuffer.NewSPSCRingBuffer(ringBufCap)
		if err != nil {
			// Only fails if cap is not a power of 2 — panic is appropriate here.
			panic(fmt.Sprintf("tick worker pool: invalid ring buffer capacity %d: %v", ringBufCap, err))
		}
		p.rings[i] = rb

		// Seed the capacity gauge once — it never changes.
		observability.TickWorkerRingBufferCapacity.
			WithLabelValues(fmt.Sprintf("%d", i)).
			Set(float64(rb.Cap()))

		p.wg.Add(1)
		go func(workerID int, rb *ringbuffer.SPSCRingBuffer) {
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
				// Try to consume a tick.
				if tick, ok := rb.TryConsume(); ok {
					processor(poolCtx, tick)
					continue
				}

				// If the context is cancelled, double-check that the buffer is empty
				// by attempting to consume one more time. This ensures we drain
				// any tick that arrived between the previous TryConsume and now.
				if poolCtx.Err() != nil {
					if tick, ok := rb.TryConsume(); ok {
						processor(poolCtx, tick)
						continue
					}
					return // buffer is empty, safe to exit
				}

				// Buffer empty and context still active — yield CPU.
				runtime.Gosched()
			}
		}(i, rb)
	}

	// Start per-worker depth reporter.
	go p.reportMetrics()

	zap.L().Info(
		"tick worker pool started",
		zap.Int("workers", workerCount),
		zap.Int("ring_buffer_cap_per_worker", ringBufCap),
	)

	return p
}

// Submit routes a tick to the correct worker's ring buffer.
//
// This is NON-BLOCKING.  If the target ring buffer is full the tick is
// dropped and the TicksDropped counter is incremented.
// Callers that need strict backpressure should monitor TicksDropped and
// the per-worker depth gauges to detect saturation.
func (p *TickWorkerPool) Submit(tick marketdata.NormalizedTick) error {
	if p.closed.Load() {
		return ErrTickWorkerPoolClosed
	}
	if p.workerCnt == 0 {
		return nil
	}

	idx := int(tick.InstrumentToken % uint32(p.workerCnt))
	if !p.rings[idx].Publish(tick) {
		observability.TicksDropped.Inc()
	}
	return nil
}

// QueueDepth returns the current ring buffer depth for the given worker.
func (p *TickWorkerPool) QueueDepth(workerID int) int {
	if workerID < 0 || workerID >= len(p.rings) {
		return 0
	}
	return p.rings[workerID].Len()
}

// TotalQueueDepth returns the total number of ticks queued across all workers.
func (p *TickWorkerPool) TotalQueueDepth() int {
	total := 0
	for _, rb := range p.rings {
		total += rb.Len()
	}
	return total
}

// TotalCapacity returns the total ring buffer capacity across all workers.
func (p *TickWorkerPool) TotalCapacity() int {
	total := 0
	for _, rb := range p.rings {
		total += rb.Cap()
	}
	return total
}

// reportMetrics publishes per-worker depth gauges every second.
func (p *TickWorkerPool) reportMetrics() {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			for i, rb := range p.rings {
				observability.TickWorkerRingBufferDepth.
					WithLabelValues(fmt.Sprintf("%d", i)).
					Set(float64(rb.Len()))
			}
		case <-p.metricStop:
			return
		}
	}
}

// Close performs a graceful shutdown.
// It cancels the pool context, waits for all workers to finish (after draining their buffers),
// and stops the metrics reporter.
func (p *TickWorkerPool) Close() {
	if !p.closed.CompareAndSwap(false, true) {
		return
	}

	zap.L().Info("stopping tick worker pool")

	p.cancel()
	p.wg.Wait()
	close(p.metricStop)

	zap.L().Info("tick worker pool stopped")
}
