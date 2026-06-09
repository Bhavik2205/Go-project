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

// Publish sends a tick to all subscribers. It is non‑blocking: if a subscriber's channel
// is full, the tick is dropped and the drop counter is incremented. The subscriber slice
// is copied under read lock to avoid deadlocking Subscribe() during a slow publish.
func (b *InProcessTickBus) Publish(ctx context.Context, tick marketdata.NormalizedTick) error {
	// Take a snapshot of subscribers under read lock
	b.mu.RLock()
	subs := make([]chan marketdata.NormalizedTick, len(b.subscribers))
	copy(subs, b.subscribers)
	b.mu.RUnlock()

	for _, ch := range subs {
		select {
		case ch <- tick:
		case <-ctx.Done():
			return ctx.Err()
		default:
			// Channel is full – drop the tick
			atomic.AddUint64(&b.droppedTicks, 1)
			b.maybeLogDropWarning()
		}
	}
	return nil
}

// maybeLogDropWarning logs a warning at most once per second.
func (b *InProcessTickBus) maybeLogDropWarning() {
	now := time.Now().UnixNano()
	last := atomic.LoadInt64(&b.lastLogTime)
	if now-last >= int64(time.Second) {
		if atomic.CompareAndSwapInt64(&b.lastLogTime, last, now) {
			zap.L().Warn("Tick dropped because subscriber channel full",
				zap.Uint64("total_dropped", atomic.LoadUint64(&b.droppedTicks)))
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
