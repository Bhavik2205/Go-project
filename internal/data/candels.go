// internal/data/candles.go
package data

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/cache"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"github.com/gorilla/websocket"
	kitemodels "github.com/zerodha/gokiteconnect/v4/models"
	"go.uber.org/zap"
	"gorm.io/gorm/clause"

	"github.com/Bhavik2205/ML-Bot/internal/indicators" // Import indicators package to use its Candle struct
)

const (
	marketOpenHour    = 9
	marketOpenMinute  = 15
	marketCloseHour   = 23 //actual 15
	marketCloseMinute = 59 //actual 30
	// Market timezone for consistency with broker data. Assuming IST (Asia/Kolkata)
	marketTimezone = "Asia/Kolkata"
)

// CandleData holds the current state of a candle being built from real-time ticks.
// It includes the OHLCV data, trade count, and the last tick time.
// The `mu` mutex protects this individual candle's data when it's accessed
// by functions like `flushCandle`, ensuring a consistent state during DB operations.
type CandleData struct {
	InstrumentToken uint32
	Interval        string     // e.g., "1m", "5m", "1h"
	Timestamp       time.Time  // Start time of the candle (truncated to the interval boundary)
	Open            float64    // First price of the candle
	High            float64    // Highest price during the candle's duration
	Low             float64    // Lowest price during the candle's duration
	Close           float64    // Last price of the candle
	Volume          float64    // Sum of LastTradedQuantity within the candle's duration
	TradeCount      uint32     // Number of ticks/trades processed for this candle
	LastTickTime    time.Time  // Timestamp of the very last tick that updated this candle
	mu              sync.Mutex // Mutex to protect concurrent access to this specific CandleData instance
}

// CandleGenerator aggregates real-time ticks into OHLCV candles for various intervals.
// It listens to a Redis Pub/Sub channel for incoming market data ticks and
// maintains a collection of "open" (currently forming) candles.
type CandleGenerator struct {
	dbClient        *db.DBClient
	redisClient     *cache.RedisClient
	appCfg          *utils.AppConfig
	openCandles     map[uint32]map[string]*CandleData
	openCandlesMu   sync.Mutex
	candleWsClients *sync.Map
	marketLoc       *time.Location
	// indicatorManagerInputCh is the channel to send completed candles to the IndicatorsManager.
	indicatorManagerInputCh chan<- indicators.Candle // NEW: Channel to send candles to IndicatorsManager
}

// NewCandleGenerator creates and returns a new instance of CandleGenerator.
// It takes dependencies for database interaction, Redis Pub/Sub, and application configuration.
// It now also accepts a channel for sending completed candles to an IndicatorsManager.
func NewCandleGenerator(
	dbC *db.DBClient,
	rC *cache.RedisClient,
	cfg *utils.AppConfig,
	wsClients *sync.Map,
	indicatorManagerInputCh chan<- indicators.Candle, // NEW: Input channel for IndicatorsManager
) *CandleGenerator {
	loc, err := time.LoadLocation(marketTimezone)
	if err != nil {
		zap.L().Error("Failed to load market timezone, defaulting to UTC. Market time-based candle alignment may be incorrect.",
			zap.String("timezone", marketTimezone), zap.Error(err))
		loc = time.UTC // Fallback to UTC if timezone cannot be loaded
	}

	return &CandleGenerator{
		dbClient:                dbC,
		redisClient:             rC,
		appCfg:                  cfg,
		openCandles:             make(map[uint32]map[string]*CandleData),
		candleWsClients:         wsClients,
		marketLoc:               loc,
		indicatorManagerInputCh: indicatorManagerInputCh, // Assign the channel
	}
}

// StartCandleGeneration subscribes to Redis ticks and processes them into candles.
// This function runs in a separate goroutine and listens for incoming market data.
// It also handles Redis connection resilience (reconnection attempts).
// The context (`ctx`) is used for graceful shutdown.
func (cg *CandleGenerator) StartCandleGeneration(ctx context.Context) {
	pubsub := cg.redisClient.Subscribe(ctx, api.RedisMarketDataChannel)
	if pubsub == nil {
		zap.L().Fatal("Failed to subscribe to Redis PubSub for candle generation. Exiting.")
		return
	}
	defer func() {
		if err := pubsub.Close(); err != nil {
			zap.L().Error("Failed to close Redis PubSub connection for candle generator", zap.Error(err))
		}
		zap.L().Info("Redis PubSub subscriber for candle generator closed.")
	}()

	zap.L().Info("✅ Candle generator subscribed to Redis market data channel", zap.String("channel", api.RedisMarketDataChannel))

	ch := pubsub.Channel()
	for {
		select {
		case msg, ok := <-ch:
			if !ok {
				zap.L().Warn("Redis PubSub channel for candle generator closed. Attempting reconnect in 5 seconds...")
				time.Sleep(5 * time.Second)
				pubsub = cg.redisClient.Subscribe(ctx, api.RedisMarketDataChannel)
				if pubsub == nil {
					zap.L().Fatal("Failed to resubscribe to Redis PubSub for candle generation. Exiting.")
					return
				}
				ch = pubsub.Channel()
				zap.L().Info("Successfully reconnected to Redis PubSub for candle generation.")
				continue
			}

			var enrichedTick struct {
				Symbol           string          `json:"symbol"`
				ProcessedAtNanos int64           `json:"processed_at_nanos"`
				Tick             json.RawMessage `json:"tick"`
			}
			if err := json.Unmarshal([]byte(msg.Payload), &enrichedTick); err != nil {
				zap.L().Error("Failed to unmarshal Redis message payload outer structure for candle generation",
					zap.Error(err),
					zap.String("payload_sample", string(msg.Payload[:min(len(msg.Payload), 200)])))
				continue
			}

			var kiteTick kitemodels.Tick
			if err := json.Unmarshal(enrichedTick.Tick, &kiteTick); err != nil {
				zap.L().Error("Failed to unmarshal raw tick for candle generation",
					zap.Error(err),
					zap.String("tick_payload_sample", string(enrichedTick.Tick[:min(len(enrichedTick.Tick), 100)])))
				continue
			}

			cg.processTickForCandles(kiteTick)

		case <-ctx.Done():
			zap.L().Info("Context cancelled, stopping candle generator Redis subscriber.")
			cg.flushAllOpenCandles()
			return
		}
	}
}

// isMarketOpen checks if the given time falls within the defined market hours.
// Times are converted to the market timezone for comparison.
func (cg *CandleGenerator) isMarketOpen(t time.Time) bool {
	// Convert the given time to the market's local time for accurate comparison
	marketTime := t.In(cg.marketLoc)

	// Define today's market open and close times in the market's local timezone
	marketOpenToday := time.Date(marketTime.Year(), marketTime.Month(), marketTime.Day(),
		marketOpenHour, marketOpenMinute, 0, 0, cg.marketLoc)
	marketCloseToday := time.Date(marketTime.Year(), marketTime.Month(), marketTime.Day(),
		marketCloseHour, marketCloseMinute, 0, 0, cg.marketLoc)

	return !marketTime.Before(marketOpenToday) && !marketTime.After(marketCloseToday)
}

// getCandleStartTime aligns a given tick time to the appropriate candle start time
// based on the interval and market hours. This handles the 9:15 AM market open.
func (cg *CandleGenerator) getCandleStartTime(tickTime time.Time, intervalDuration time.Duration) time.Time {
	marketTime := tickTime.In(cg.marketLoc)

	// Define market open time for the current day
	marketOpenToday := time.Date(marketTime.Year(), marketTime.Month(), marketTime.Day(),
		marketOpenHour, marketOpenMinute, 0, 0, cg.marketLoc)

	// If the tick is before market open, it should not form a candle in real-time.
	if marketTime.Before(marketOpenToday) {
		return time.Time{} // Return zero time, indicating invalid candle start
	}

	if intervalDuration == time.Hour {
		// Calculate minutes since market open for the current day
		minutesSinceMarketOpen := marketTime.Sub(marketOpenToday).Minutes()

		if minutesSinceMarketOpen < 0 {
			return time.Time{}
		}

		// Number of complete 1-hour candles passed since market open
		hourOffset := int(minutesSinceMarketOpen / 60)

		candleStartTime := marketOpenToday.Add(time.Duration(hourOffset) * time.Hour)

		return candleStartTime.In(time.UTC) // Store and use UTC internally for consistency
	} else if intervalDuration == 24*time.Hour { // For "1d" candles
		// Daily candle always starts at market open (9:15 AM) of the current day
		candleStartTime := marketOpenToday
		return candleStartTime.In(time.UTC) // Store and use UTC internally
	}

	// For other intervals (1m, 5m, 15m), simply truncate to the interval boundary.
	// Ensure truncation is done in the market's local time first, then convert to UTC.
	truncatedLocalTime := marketTime.Truncate(intervalDuration)
	return truncatedLocalTime.In(time.UTC) // Store and use UTC internally
}

// processTickForCandles processes an incoming market data tick to update or create
// OHLCV candles for all configured time intervals.
// This function is now fully protected by `cg.openCandlesMu` to ensure thread safety
// across all map accesses and modifications.
func (cg *CandleGenerator) processTickForCandles(tick kitemodels.Tick) {
	instrumentToken := tick.InstrumentToken
	tickTime := tick.Timestamp.Time // This is the exchange's timestamp for the tick (should be in IST from Zerodha)

	// Filter out ticks outside market hours (9:15 AM - 3:30 PM IST)
	if !cg.isMarketOpen(tickTime) {
		zap.L().Debug("Skipping tick outside market hours",
			zap.Uint32("token", instrumentToken),
			zap.Time("tick_time", tickTime.In(cg.marketLoc)))
		return
	}

	cg.openCandlesMu.Lock()
	defer cg.openCandlesMu.Unlock()

	instrumentCandles, ok := cg.openCandles[instrumentToken]
	if !ok {
		instrumentCandles = make(map[string]*CandleData)
		cg.openCandles[instrumentToken] = instrumentCandles
	}

	for _, intervalStr := range cg.appCfg.Candles.Intervals {
		intervalDuration, err := parseInterval(intervalStr)
		if err != nil {
			zap.L().Error("Invalid candle interval configured in app.yaml, skipping candle generation for this interval",
				zap.String("interval", intervalStr),
				zap.Error(err))
			continue
		}

		// Get the precise start time for the candle to which this tick belongs, considering market hours
		candleStartTime := cg.getCandleStartTime(tickTime, intervalDuration)
		if candleStartTime.IsZero() {
			zap.L().Warn("Could not determine valid candle start time for tick, skipping",
				zap.Uint32("token", instrumentToken),
				zap.String("interval", intervalStr),
				zap.Time("tick_time", tickTime.In(cg.marketLoc)))
			continue
		}

		currentCandle, candleExists := instrumentCandles[intervalStr]

		// Check if a new candle period has started or if this is the very first tick for this interval.
		// Note: The `candleStartTime` is already adjusted for market hours.
		if !candleExists || currentCandle.Timestamp.Before(candleStartTime) {
			// A new candle period has started.
			// If an old candle existed for this interval, flush it *before* creating the new one.
			if candleExists && currentCandle.Timestamp.Before(candleStartTime) {
				tempCandleToFlush := *currentCandle
				cg.flushCandle(&tempCandleToFlush) // This will now also send to IndicatorsManager
				zap.L().Debug("Flushed completed candle",
					zap.Uint32("token", tempCandleToFlush.InstrumentToken),
					zap.String("interval", tempCandleToFlush.Interval),
					zap.Time("timestamp", tempCandleToFlush.Timestamp.In(cg.marketLoc)), // Log in market time for clarity
					zap.Float64("close", tempCandleToFlush.Close))
			}

			newCandle := &CandleData{
				InstrumentToken: instrumentToken,
				Interval:        intervalStr,
				Timestamp:       candleStartTime, // Use the market-aligned start time
				Open:            tick.LastPrice,
				High:            tick.LastPrice,
				Low:             tick.LastPrice,
				Close:           tick.LastPrice,
				Volume:          float64(tick.LastTradedQuantity),
				TradeCount:      1,
				LastTickTime:    tickTime,
			}
			instrumentCandles[intervalStr] = newCandle
			zap.L().Debug("Created new candle",
				zap.Uint32("token", instrumentToken),
				zap.String("interval", intervalStr),
				zap.Time("timestamp", candleStartTime.In(cg.marketLoc)), // Log in market time for clarity
				zap.Float64("open", tick.LastPrice),
				zap.Time("tick_time", tickTime.In(cg.marketLoc)))
		} else {
			// The tick belongs to the currently open candle. Update its High, Low, Close, Volume, and TradeCount.
			if tick.LastPrice > currentCandle.High {
				currentCandle.High = tick.LastPrice
			}
			if tick.LastPrice < currentCandle.Low {
				currentCandle.Low = tick.LastPrice
			}
			currentCandle.Close = tick.LastPrice
			currentCandle.Volume += float64(tick.LastTradedQuantity)
			currentCandle.TradeCount++
			currentCandle.LastTickTime = tickTime
			zap.L().Debug("Updated existing candle",
				zap.Uint32("token", instrumentToken),
				zap.String("interval", intervalStr),
				zap.Time("timestamp", currentCandle.Timestamp.In(cg.marketLoc)), // Log in market time
				zap.Float64("close", currentCandle.Close),
				zap.Float64("volume", currentCandle.Volume))
		}
	}
}

// flushCandle saves a completed CandleData to the database and sends it for indicator calculation.
func (cg *CandleGenerator) flushCandle(cd *CandleData) {
	if cd == nil {
		zap.L().Debug("Attempted to flush a nil candle, skipping.")
		return
	}

	cd.mu.Lock()
	defer cd.mu.Unlock()

	ohlcvCandle := db.OHLCVCandle{
		InstrumentToken: cd.InstrumentToken,
		Interval:        cd.Interval,
		Timestamp:       cd.Timestamp,
		Open:            cd.Open,
		High:            cd.High,
		Low:             cd.Low,
		Close:           cd.Close,
		Volume:          cd.Volume,
		TradeCount:      cd.TradeCount,
	}

	result := cg.dbClient.DB.Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "timestamp"}},
		DoUpdates: clause.AssignmentColumns([]string{"high", "low", "close", "volume", "trade_count", "updated_at"}),
	}).Create(&ohlcvCandle)

	if result.Error != nil {
		zap.L().Error("❌ Failed to save/update OHLCVCandle to DB",
			zap.Error(result.Error),
			zap.Uint32("instrument_token", cd.InstrumentToken),
			zap.String("interval", cd.Interval),
			zap.Time("timestamp", cd.Timestamp))
	} else if result.RowsAffected == 0 {
		zap.L().Debug("OHLCVCandle already up-to-date or conflict resolved with no changes",
			zap.Uint32("instrument_token", cd.InstrumentToken),
			zap.String("interval", cd.Interval),
			zap.Time("timestamp", cd.Timestamp))
	} else {
		zap.L().Info("✅ Saved/Updated OHLCVCandle to DB",
			zap.Uint32("instrument_token", cd.InstrumentToken),
			zap.String("interval", cd.Interval),
			zap.Time("timestamp", cd.Timestamp),
			zap.Float64("close", cd.Close),
			zap.Float64("volume", cd.Volume),
			zap.Uint32("trade_count", cd.TradeCount))

		// Broadcast the candle after successful DB operation
		cg.broadcastCandle(cd)

		zap.L().Debug("INDICATORMANAGERINPUTCH type", zap.String("type", fmt.Sprintf("%T", cg.indicatorManagerInputCh)))
		// NEW: Send the completed candle to the IndicatorsManager for calculation.
		// It's important to send a copy or a value to avoid concurrent modification issues.
		if cg.indicatorManagerInputCh != nil {
			select {
			case cg.indicatorManagerInputCh <- indicators.Candle{ // Convert CandleData to indicators.Candle
				InstrumentToken: cd.InstrumentToken,
				Interval:        cd.Interval,
				Timestamp:       cd.Timestamp,
				Open:            cd.Open,
				High:            cd.High,
				Low:             cd.Low,
				Close:           cd.Close,
				Volume:          cd.Volume,
				TradeCount:      cd.TradeCount,
			}:
				zap.L().Debug("Sent completed candle to IndicatorsManager",
					zap.Uint32("token", cd.InstrumentToken),
					zap.String("interval", cd.Interval),
					zap.Time("timestamp", cd.Timestamp.In(cg.marketLoc)))
			default:
				zap.L().Warn("IndicatorsManager input channel is full; dropping candle for indicator calculation.",
					zap.Uint32("token", cd.InstrumentToken),
					zap.String("interval", cd.Interval),
					zap.Time("timestamp", cd.Timestamp.In(cg.marketLoc)))
			}
		}
	}
}

// broadcastCandle marshals and sends the candle data to all connected WebSocket clients.
func (cg *CandleGenerator) broadcastCandle(cd *CandleData) {
	broadcastData := struct {
		InstrumentToken uint32    `json:"instrument_token"`
		Interval        string    `json:"interval"`
		Timestamp       time.Time `json:"timestamp"`
		Open            float64   `json:"open"`
		High            float64   `json:"high"`
		Low             float64   `json:"low"`
		Close           float64   `json:"close"`
		Volume          float64   `json:"volume"`
		TradeCount      uint32    `json:"trade_count"`
	}{
		InstrumentToken: cd.InstrumentToken,
		Interval:        cd.Interval,
		Timestamp:       cd.Timestamp,
		Open:            cd.Open,
		High:            cd.High,
		Low:             cd.Low,
		Close:           cd.Close,
		Volume:          cd.Volume,
		TradeCount:      cd.TradeCount,
	}

	jsonMessage, err := json.Marshal(broadcastData)
	if err != nil {
		zap.L().Error("Failed to marshal candle data for WebSocket broadcast", zap.Error(err), zap.Uint32("token", cd.InstrumentToken))
		return
	}

	cg.candleWsClients.Range(func(key, value interface{}) bool {
		conn, ok := value.(*websocket.Conn)
		if !ok {
			zap.L().Error("Invalid type in candleWsClients map for key", zap.Any("key", key))
			cg.candleWsClients.Delete(key)
			return true
		}

		if err := conn.WriteMessage(websocket.TextMessage, jsonMessage); err != nil {
			zap.L().Error("Failed to write candle message to WebSocket client, removing client",
				zap.Error(err),
				zap.String("remote_addr", conn.RemoteAddr().String()),
				zap.Uint32("token", cd.InstrumentToken))
			cg.candleWsClients.Delete(key)
		}
		return true
	})
	zap.L().Debug("Broadcasted candle to WebSocket clients", zap.Uint32("token", cd.InstrumentToken), zap.String("interval", cd.Interval))
}

// parseInterval converts a string representation of a time interval (e.g., "1m", "5m", "1h")
// into a `time.Duration` type.
func parseInterval(interval string) (time.Duration, error) {
	switch interval {
	case "1m":
		return time.Minute, nil
	case "5m":
		return 5 * time.Minute, nil
	case "15m":
		return 15 * time.Minute, nil
	case "1h":
		return time.Hour, nil
	case "1d":
		return 24 * time.Hour, nil
	default:
		return 0, fmt.Errorf("unsupported interval: %s", interval)
	}
}

// flushAllOpenCandles iterates through all currently "open" (forming) candles and flushes them
// to the database. This function is typically called during application shutdown
// to ensure that no in-memory candle data is lost.
func (cg *CandleGenerator) flushAllOpenCandles() {
	zap.L().Info("Flushing all remaining open candles during graceful shutdown...")
	cg.openCandlesMu.Lock()
	defer cg.openCandlesMu.Unlock()

	for _, instrumentCandles := range cg.openCandles {
		for _, candle := range instrumentCandles {
			tempCandleToFlush := *candle
			cg.flushCandle(&tempCandleToFlush) // This will also send to IndicatorsManager
		}
	}
	zap.L().Info("All open candles flushed successfully.")
}
