package tickbus

import (
	"context"
	"sync"

	"github.com/Bhavik2205/ML-Bot/internal/marketdata"
)

type DualTickBus struct {
	local *InProcessTickBus
	redis *RedisTickBus
}

func NewDual(local *InProcessTickBus, redis *RedisTickBus) *DualTickBus {
	return &DualTickBus{local: local, redis: redis}
}

func (d *DualTickBus) Publish(ctx context.Context, tick marketdata.NormalizedTick) error {
	var wg sync.WaitGroup
	var err1, err2 error
	wg.Add(2)
	go func() { err1 = d.local.Publish(ctx, tick); wg.Done() }()
	go func() { err2 = d.redis.Publish(ctx, tick); wg.Done() }()
	wg.Wait()
	if err1 != nil {
		return err1
	}
	return err2
}

func (d *DualTickBus) Subscribe(ctx context.Context) (<-chan marketdata.NormalizedTick, error) {
	// For subscribing, you must choose one backend (or merge channels).
	// Usually you'd want to subscribe from the local bus for performance.
	return d.local.Subscribe(ctx)
}

func (d *DualTickBus) Close() error {
	_ = d.local.Close()
	return d.redis.Close()
}
