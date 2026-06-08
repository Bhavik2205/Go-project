package tickbus

import (
	"context"
	"sync"

	"github.com/Bhavik2205/ML-Bot/internal/marketdata"
)

type InProcessTickBus struct {
	subscribers []chan marketdata.NormalizedTick
	mu          sync.RWMutex
}

func NewInProcess() *InProcessTickBus {
	return &InProcessTickBus{}
}

func (b *InProcessTickBus) Publish(ctx context.Context, tick marketdata.NormalizedTick) error {
	b.mu.RLock()
	defer b.mu.RUnlock()
	for _, ch := range b.subscribers {
		select {
		case ch <- tick:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
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
