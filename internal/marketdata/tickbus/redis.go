package tickbus

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/cache"
	"github.com/Bhavik2205/ML-Bot/internal/marketdata"
	"go.uber.org/zap"
)

// typed context key to avoid string collisions
type ctxKey int

const (
	// ProcessedAtNanosKey can be used by callers to inject the processing timestamp.
	ProcessedAtNanosKey ctxKey = iota
)

type RedisTickBus struct {
	redis   *cache.RedisClient
	channel string
}

func NewRedis(redisClient *cache.RedisClient, channel string) *RedisTickBus {
	return &RedisTickBus{
		redis:   redisClient,
		channel: channel,
	}
}

func (b *RedisTickBus) Publish(ctx context.Context, tick marketdata.NormalizedTick) error {
	// Safely extract processed_at_nanos, fallback to current time if missing
	var nanos int64
	if v, ok := ctx.Value(ProcessedAtNanosKey).(int64); ok {
		nanos = v
	} else {
		nanos = time.Now().UnixNano()
	}
	enriched := struct {
		Symbol           string                    `json:"symbol"`
		ProcessedAtNanos int64                     `json:"processed_at_nanos"`
		Tick             marketdata.NormalizedTick `json:"tick"`
	}{
		Symbol:           tick.Symbol,
		ProcessedAtNanos: nanos,
		Tick:             tick,
	}
	data, err := json.Marshal(enriched)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	return b.redis.Publish(b.channel, data)
}

func (b *RedisTickBus) Subscribe(ctx context.Context) (<-chan marketdata.NormalizedTick, error) {
	out := make(chan marketdata.NormalizedTick, 1000)
	pubsub := b.redis.Subscribe(ctx, b.channel)

	go func() {
		defer close(out)
		defer pubsub.Close()
		ch := pubsub.Channel()
		for {
			select {
			case msg, ok := <-ch:
				if !ok {
					return
				}
				var enriched struct {
					Tick marketdata.NormalizedTick `json:"tick"`
				}
				if err := json.Unmarshal([]byte(msg.Payload), &enriched); err != nil {
					zap.L().Error("Failed to unmarshal tick from Redis", zap.Error(err))
					continue
				}
				select {
				case out <- enriched.Tick:
				case <-ctx.Done():
					return
				}
			case <-ctx.Done():
				return
			}
		}
	}()
	return out, nil
}

func (b *RedisTickBus) Close() error {
	return nil
}
