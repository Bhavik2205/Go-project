package api

import (
	"context"
	"fmt"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/marketdata"
	"github.com/Bhavik2205/ML-Bot/internal/marketdata/tickbus"
	"github.com/Bhavik2205/ML-Bot/internal/marketdata/wal"
	"github.com/Bhavik2205/ML-Bot/internal/observability"
	kitemodels "github.com/zerodha/gokiteconnect/v4/models"
	kiteticker "github.com/zerodha/gokiteconnect/v4/ticker"
	"go.uber.org/zap"
)

// RedisMarketDataChannel defines the Redis Pub/Sub channel for market data.
const RedisMarketDataChannel = "market_data_ticks"

// SubscribeToTicks subscribes to ticks and publishes them as NormalizedTick to the TickBus.
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
		currentVolume := tick.VolumeTraded

		if currentPrice != prevPrice || currentVolume != uint32(prevVolume) {
			lastPrices[tick.InstrumentToken] = currentPrice
			lastVolumes[tick.InstrumentToken] = int(currentVolume)

			label := tokenToLabel[tick.InstrumentToken]
			normalized := marketdata.NormalizedTick{
				InstrumentToken:    tick.InstrumentToken,
				Symbol:             label,
				Exchange:           "",
				EventTime:          tick.Timestamp.Time,
				IngestTime:         time.Now(),
				LastPrice:          tick.LastPrice,
				LastTradedQuantity: tick.LastTradedQuantity,
				Volume:             tick.VolumeTraded,
				AverageTradePrice:  tick.AverageTradePrice,
				NetChange:          tick.NetChange,
				OHLC:               tick.OHLC,
				Depth:              tick.Depth,
				TotalBuyQuantity:   tick.TotalBuyQuantity,
				TotalSellQuantity:  tick.TotalSellQuantity,
				OpenInterest:       tick.OI,
				Mode:               "live",
			}
			if normalized.OHLC.Close != 0 {
				normalized.PrevClose = normalized.OHLC.Close
				normalized.PercentChange = (normalized.LastPrice - normalized.PrevClose) / normalized.PrevClose * 100
			}

			if err := wal.Append(normalized); err != nil {
				zap.L().Error("❌ WAL append failed",
					zap.Uint32("instrument_token", tick.InstrumentToken),
					zap.Error(err),
				)
			}
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

	z.Ticker.OnClose(func(code int, reason string) {
		zap.L().Warn(
			"🔌 Zerodha WebSocket closed",
			zap.Int("code", code),
			zap.String("reason", reason),
		)

		go func(tokensToResubscribe []uint32) {
			const (
				initialBackoff = 1 * time.Second
				maxBackoff     = 1 * time.Minute
				connectTimeout = 10 * time.Second
			)

			backoff := initialBackoff

			for {
				zap.L().Info(
					"Attempting WebSocket reconnect",
					zap.Duration("backoff", backoff),
				)

				newTicker := kiteticker.New(z.APIKey, z.AccessToken)

				connected := make(chan struct{}, 1)

				newTicker.OnConnect(func() {
					zap.L().Info("✅ Reconnected to Zerodha WebSocket")

					if err := newTicker.Subscribe(tokensToResubscribe); err != nil {
						zap.L().Error(
							"Failed to resubscribe after reconnect",
							zap.Error(err),
						)
						return
					}

					if err := newTicker.SetMode(
						kiteticker.ModeFull,
						tokensToResubscribe,
					); err != nil {
						zap.L().Error(
							"Failed to set mode after reconnect",
							zap.Error(err),
						)
						return
					}

					z.Ticker = newTicker

					select {
					case connected <- struct{}{}:
					default:
					}
				})

				newTicker.OnError(func(err error) {
					zap.L().Error(
						"Reconnect attempt failed",
						zap.Error(err),
					)
				})

				newTicker.OnClose(func(code int, reason string) {
					zap.L().Warn(
						"Reconnected WebSocket closed again",
						zap.Int("code", code),
						zap.String("reason", reason),
					)
				})

				go func() {
					defer func() {
						if r := recover(); r != nil {
							zap.L().Error(
								"Panic in reconnect Serve",
								zap.Any("panic", r),
							)
						}
					}()

					newTicker.Serve()
				}()

				select {
				case <-connected:
					zap.L().Info(
						"Successfully reconnected and resubscribed",
					)
					return

				case <-time.After(connectTimeout):
					zap.L().Warn(
						"Reconnect timed out",
						zap.Duration("timeout", connectTimeout),
					)
				}

				time.Sleep(backoff)

				if backoff < maxBackoff {
					backoff *= 2

					if backoff > maxBackoff {
						backoff = maxBackoff
					}
				}
			}
		}(tokens)
	})

	go func() {
		defer observability.RecoverPanic("ticker-serve")
		z.Ticker.Serve()
	}()
	return nil
}
