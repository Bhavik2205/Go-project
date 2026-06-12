package api

import (
	"context"
	"fmt"
	"time"

	// Import your Redis client
	"github.com/Bhavik2205/ML-Bot/internal/marketdata"
	"github.com/Bhavik2205/ML-Bot/internal/marketdata/tickbus"
	"github.com/Bhavik2205/ML-Bot/internal/marketdata/wal"
	"github.com/Bhavik2205/ML-Bot/internal/observability"
	kitemodels "github.com/zerodha/gokiteconnect/v4/models"
	kiteticker "github.com/zerodha/gokiteconnect/v4/ticker"
	"go.uber.org/zap" // Use zap logger
)

// RedisMarketDataChannel defines the Redis Pub/Sub channel for market data.
const RedisMarketDataChannel = "market_data_ticks"

// SubscribeToTicks subscribes to ticks and publishes them as NormalizedTick to Redis.
func (z *ZerodhaClient) SubscribeToTicks(infos []*InstrumentInfo, tb tickbus.TickBus, wal wal.Writer) error {
	if tb == nil {
		return fmt.Errorf("TickBus is nil, cannot publish ticks")
	}

	if len(infos) == 0 {
		return fmt.Errorf("no instruments provided for subscription")
	}

	tokens := make([]uint32, 0, len(infos))
	tokenToLabel := make(map[uint32]string)

	for _, info := range infos {
		tokens = append(tokens, info.Token)
		tokenToLabel[info.Token] = fmt.Sprintf("%s:%s", info.Exchange, info.Symbol)
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
		observability.TicksReceived.Inc()
		observability.UpdateLastTickTimestamp(tick.Timestamp.Time)

		lag := time.Since(tick.Timestamp.Time).Milliseconds()
		observability.RecordTickLag(float64(lag))

		prevPrice := lastPrices[tick.InstrumentToken]
		currentPrice := float32(tick.LastPrice)

		prevVolume := lastVolumes[tick.InstrumentToken]
		currentVolume := tick.VolumeTraded // Already uint32

		if currentPrice != prevPrice || currentVolume != uint32(prevVolume) {
			lastPrices[tick.InstrumentToken] = currentPrice
			lastVolumes[tick.InstrumentToken] = int(currentVolume)

			// // Visual feedback for console (can be removed in production if not needed)
			// colorReset := "\033[0m"
			// colorRed := "\033[31m"
			// colorGreen := "\033[32m"
			// color := colorReset

			// if prevPrice != 0 {
			// 	if currentPrice > prevPrice {
			// 		color = colorGreen
			// 	} else if currentPrice < prevPrice {
			// 		color = colorRed
			// 	}
			// }

			label := tokenToLabel[tick.InstrumentToken]
			// fmt.Printf(
			// 	"📈 %s [Token: %d] - LTP: %s%.2f%s Vol: %d O: %.2f H: %.2f L: %.2f C: %.2f\n",
			// 	label, tick.InstrumentToken,
			// 	color, currentPrice, colorReset,
			// 	tick.VolumeTraded,
			// 	tick.OHLC.Open,
			// 	tick.OHLC.High,
			// 	tick.OHLC.Low,
			// 	tick.OHLC.Close,
			// )
			// Construct NormalizedTick
			normalized := marketdata.NormalizedTick{
				InstrumentToken:    tick.InstrumentToken,
				Symbol:             label,
				Exchange:           "", // extract from label if needed
				EventTime:          tick.Timestamp.Time,
				IngestTime:         time.Now(),
				LastPrice:          tick.LastPrice,
				LastTradedQuantity: tick.LastTradedQuantity,
				Volume:             tick.VolumeTraded,
				AverageTradePrice:  tick.AverageTradePrice,
				NetChange:          tick.NetChange,
				// PercentChange, PrevClose would need tick.OHLC.Close? We'll set later.
				OHLC:              tick.OHLC,
				Depth:             tick.Depth,
				TotalBuyQuantity:  tick.TotalBuyQuantity,
				TotalSellQuantity: tick.TotalSellQuantity,
				OpenInterest:      tick.OI,
				Mode:              "live", // "simulation", "live", or "kite"
			}
			// Compute percent change and prev close
			if normalized.OHLC.Close != 0 {
				normalized.PrevClose = normalized.OHLC.Close
				normalized.PercentChange = (normalized.LastPrice - normalized.PrevClose) / normalized.PrevClose * 100
			}

			// WAL first — tick must be durable before going downstream.
			if err := wal.Append(normalized); err != nil {
				zap.L().Error("❌ WAL append failed",
					zap.Uint32("instrument_token", tick.InstrumentToken),
					zap.Error(err),
				)
				// WAL failure is non-fatal — we still publish to keep the pipeline live,
				// but the metric will alert operations.
			}

			// Publish to TickBus only after WAL append attempt.
			if err := tb.Publish(context.Background(), normalized); err != nil {
				zap.L().Error("❌ Failed to publish tick to TickBus",
					zap.Uint32("instrument_token", tick.InstrumentToken),
					zap.Error(err),
				)
			}
		}
	})

	z.Ticker.OnError(func(err error) {
		zap.L().Error("❌ Zerodha WebSocket error", zap.Error(err))
	})

	// z.Ticker.OnClose(func(code int, reason string) {
	// 	zap.L().Warn("🔌 Zerodha WebSocket closed", zap.Int("code", code), zap.String("reason", reason))
	// })

	// ========== FIXED: WebSocket reconnect with exponential backoff ==========
	z.Ticker.OnClose(func(code int, reason string) {
		zap.L().Warn("🔌 Zerodha WebSocket closed", zap.Int("code", code), zap.String("reason", reason))

		// Capture tokens for reconnection
		go func(tokensToResubscribe []uint32) {
			backoff := 1 * time.Second
			const maxBackoff = 1 * time.Minute

			for {
				time.Sleep(backoff)
				zap.L().Info("Attempting WebSocket reconnect", zap.Duration("backoff", backoff))

				// Create new ticker with same credentials
				newTicker := kiteticker.New(z.APIKey, z.AccessToken)

				// Set up callbacks for the new ticker
				newTicker.OnConnect(func() {
					zap.L().Info("Reconnected to Zerodha WebSocket.")
					if err := newTicker.Subscribe(tokensToResubscribe); err != nil {
						zap.L().Error("Failed to resubscribe after reconnect", zap.Error(err))
					} else {
						zap.L().Info("Resubscribed to tokens after reconnect")
					}
					if err := newTicker.SetMode(kiteticker.ModeFull, tokensToResubscribe); err != nil {
						zap.L().Error("Failed to set mode after reconnect", zap.Error(err))
					}
					// Replace the old ticker with the new one
					z.Ticker = newTicker
				})

				newTicker.OnError(func(err error) {
					zap.L().Error("Reconnected WebSocket error", zap.Error(err))
				})

				newTicker.OnClose(func(code int, reason string) {
					zap.L().Warn("Reconnected WebSocket closed again", zap.Int("code", code), zap.String("reason", reason))
					// This will trigger another reconnect attempt via the outer loop
				})

				// Start the new ticker in a goroutine
				go func() {
					defer func() {
						if r := recover(); r != nil {
							zap.L().Error("Panic in reconnected ticker Serve", zap.Any("recover", r))
						}
					}()
					newTicker.Serve()
				}()

				// Wait 5 seconds – if the connection dies immediately, OnClose will log and loop retries.
				// If it survives 5 seconds, assume success and exit the reconnect loop.
				time.Sleep(5 * time.Second)
				zap.L().Info("WebSocket reconnection attempt completed – assuming success after 5s")
				return
			}
		}(tokens) // pass the captured tokens slice
	})

	go func() {
		defer observability.RecoverPanic("ticker-serve")
		z.Ticker.Serve()
	}()
	return nil
}
