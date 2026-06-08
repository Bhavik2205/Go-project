package tickbus

import (
	"context"

	"github.com/Bhavik2205/ML-Bot/internal/marketdata"
)

type TickBus interface {
	// Publish sends a normalized tick to all subscribers.
	Publish(ctx context.Context, tick marketdata.NormalizedTick) error

	// Subscribe returns a channel that will receive all published ticks.
	Subscribe(ctx context.Context) (<-chan marketdata.NormalizedTick, error)

	// Close shuts down the bus and releases resources.
	Close() error
}
