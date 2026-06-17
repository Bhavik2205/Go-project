package tickbus

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/marketdata"
	"go.uber.org/zap"
)

type InProcessTickBus struct {
	subscribers  []chan marketdata.NormalizedTick
	mu           sync.RWMutex
	droppedTicks uint64
	lastLogTime  int64
}

func NewInProcess() *InProcessTickBus {
	return &InProcessTickBus{}
}

// Publish sends a tick to all subscribers. It applies backpressure instead of
// dropping data: if a subscriber is saturated we wait until it drains or the
// caller's context is cancelled. The subscriber slice is copied under read lock
// so Subscribe() is never blocked behind a slow publish loop.
func (b *InProcessTickBus) Publish(ctx context.Context, tick marketdata.NormalizedTick) error {
	b.mu.RLock()
	subs := make([]chan marketdata.NormalizedTick, len(b.subscribers))
	copy(subs, b.subscribers)
	b.mu.RUnlock()

	for _, ch := range subs {
		if cap(ch) > 0 && len(ch) == cap(ch) {
			b.maybeLogBackpressureWarning(len(ch), cap(ch))
		}
		select {
		case ch <- tick:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

// maybeLogBackpressureWarning logs a warning at most once per second while a
// subscriber is forcing the publisher to wait.
func (b *InProcessTickBus) maybeLogBackpressureWarning(depth, capacity int) {
	now := time.Now().UnixNano()
	last := atomic.LoadInt64(&b.lastLogTime)
	if now-last >= int64(time.Second) {
		if atomic.CompareAndSwapInt64(&b.lastLogTime, last, now) {
			zap.L().Warn("TickBus subscriber saturated; publisher waiting for backpressure to clear",
				zap.Int("queue_depth", depth),
				zap.Int("queue_capacity", capacity))
		}
	}
}

func (b *InProcessTickBus) Subscribe(ctx context.Context) (<-chan marketdata.NormalizedTick, error) {
	ch := make(chan marketdata.NormalizedTick, 1000)
	b.mu.Lock()
	b.subscribers = append(b.subscribers, ch)
	b.mu.Unlock()
	return ch, nil
}

func (b *InProcessTickBus) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	for _, ch := range b.subscribers {
		close(ch)
	}
	b.subscribers = nil
	return nil
}

// Stats returns the current number of dropped ticks.
func (b *InProcessTickBus) Stats() (droppedTicks uint64) {
	return atomic.LoadUint64(&b.droppedTicks)
}

// DroppedTicks is a convenience wrapper around Stats().
func (b *InProcessTickBus) DroppedTicks() uint64 {
	return b.Stats()
}
