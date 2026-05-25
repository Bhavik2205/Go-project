package api

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/cache" // Import your Redis client
	kitemodels "github.com/zerodha/gokiteconnect/v4/models"
	kiteticker "github.com/zerodha/gokiteconnect/v4/ticker"
	"go.uber.org/zap" // Use zap logger
)

// RedisMarketDataChannel defines the Redis Pub/Sub channel for market data.
const RedisMarketDataChannel = "market_data_ticks"

// SubscribeToTicks subscribes to ticks and calls the given handler on each tick
func (z *ZerodhaClient) SubscribeToTicks(infos []*InstrumentInfo, redisClient *cache.RedisClient) error {
	if redisClient == nil {
		return fmt.Errorf("RedisClient is nil, cannot publish ticks")
	}

	tokens := make([]uint32, 0, len(infos))
	tokenToLabel := make(map[uint32]string)

	for _, info := range infos {
		tokens = append(tokens, info.Token)
		tokenToLabel[info.Token] = fmt.Sprintf("%s (%s)", info.Symbol, info.Exchange)
	}

	z.Ticker = kiteticker.New(z.APIKey, z.AccessToken)

	z.Ticker.OnConnect(func() {
		zap.L().Info("✅ Connected to Zerodha WebSocket.")
		if err := z.Ticker.Subscribe(tokens); err != nil {
			zap.L().Error("❌ Zerodha Subscribe error", zap.Error(err))
		}
		if err := z.Ticker.SetMode(kiteticker.ModeFull, tokens); err != nil {
			zap.L().Error("❌ Zerodha SetMode error", zap.Error(err))
		}
	})

	lastPrices := make(map[uint32]float32)
	lastVolumes := make(map[uint32]int)

	z.Ticker.OnTick(func(tick kitemodels.Tick) {
		prevPrice := lastPrices[tick.InstrumentToken]
		currentPrice := float32(tick.LastPrice)

		prevVolume := lastVolumes[tick.InstrumentToken]
		currentVolume := tick.VolumeTraded // Already uint32

		if currentPrice != prevPrice || currentVolume != uint32(prevVolume) {
			lastPrices[tick.InstrumentToken] = currentPrice
			lastVolumes[tick.InstrumentToken] = int(currentVolume)

			label := tokenToLabel[tick.InstrumentToken]

			enrichedTick := struct {
				Symbol           string          `json:"symbol"`
				ProcessedAtNanos int64           `json:"processed_at_nanos"` // Timestamp when this was processed by the API gateway
				Tick             kitemodels.Tick `json:"tick"`
			}{
				Symbol:           label,
				ProcessedAtNanos: time.Now().UnixNano(),
				Tick:             tick,
			}

			if jsonData, err := json.Marshal(enrichedTick); err == nil {
				err := redisClient.Publish(RedisMarketDataChannel, jsonData)
				if err != nil {
					zap.L().Error("❌ Failed to publish tick to Redis",
						zap.Uint32("instrument_token", tick.InstrumentToken),
						zap.Error(err),
					)
				}
			} else {
				zap.L().Error("❌ Failed to marshal enriched tick data for Redis",
					zap.Uint32("instrument_token", tick.InstrumentToken),
					zap.Error(err),
				)
			}
		}
	})

	z.Ticker.OnError(func(err error) {
		zap.L().Error("❌ Zerodha WebSocket error", zap.Error(err))
	})

	z.Ticker.OnClose(func(code int, reason string) {
		zap.L().Warn("🔌 Zerodha WebSocket closed", zap.Int("code", code), zap.String("reason", reason))
	})

	go func() {
		defer func() {
			if r := recover(); r != nil {
				zap.L().Error("Panic in Zerodha Ticker Serve goroutine", zap.Any("recover", r))
			}
		}()
		z.Ticker.Serve()
	}()
	return nil
}
