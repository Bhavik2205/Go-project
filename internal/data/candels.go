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
	"github.com/gorilla/websocket" // NEW: Import for WebSocket
	kitemodels "github.com/zerodha/gokiteconnect/v4/models"
	"go.uber.org/zap"
	"gorm.io/gorm/clause"
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
	dbClient    *db.DBClient
	redisClient *cache.RedisClient
	appCfg      *utils.AppConfig
	// openCandles stores currently forming candles.
	// Structure: map[InstrumentToken]map[Interval]*CandleData
	// Example: openCandles[12345]["1m"] -> points to the CandleData for instrument 12345, 1-minute interval.
	openCandles map[uint32]map[string]*CandleData
	// openCandlesMu protects concurrent access to the openCandles map itself.
	// It now protects all read/write access to the `openCandles` map and its nested maps within `processTickForCandles`.
	openCandlesMu sync.Mutex // Changed from RWMutex to simple Mutex for simpler, safer full-function locking.
	// NEW: candleWsClients stores active WebSocket connections to broadcast candle data.
	candleWsClients *sync.Map // Added for broadcasting candles
}

// NewCandleGenerator creates and returns a new instance of CandleGenerator.
// It takes dependencies for database interaction, Redis Pub/Sub, and application configuration.
// NEW: Added wsClients for candle broadcasting.
func NewCandleGenerator(dbC *db.DBClient, rC *cache.RedisClient, cfg *utils.AppConfig, wsClients *sync.Map) *CandleGenerator {
	return &CandleGenerator{
		dbClient:        dbC,
		redisClient:     rC,
		appCfg:          cfg,
		openCandles:     make(map[uint32]map[string]*CandleData),
		candleWsClients: wsClients, // Assign the passed WebSocket clients map
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
		// Ensure the Redis PubSub connection is closed when the function exits.
		if err := pubsub.Close(); err != nil {
			zap.L().Error("Failed to close Redis PubSub connection for candle generator", zap.Error(err))
		}
		zap.L().Info("Redis PubSub subscriber for candle generator closed.")
	}()

	zap.L().Info("✅ Candle generator subscribed to Redis market data channel", zap.String("channel", api.RedisMarketDataChannel))

	ch := pubsub.Channel() // Get the channel for messages
	for {
		select {
		case msg, ok := <-ch:
			// Check if the channel is still open. If not, it means the Redis connection might have broken.
			if !ok {
				zap.L().Warn("Redis PubSub channel for candle generator closed. Attempting reconnect in 5 seconds...")
				time.Sleep(5 * time.Second)                                        // Wait before attempting reconnect
				pubsub = cg.redisClient.Subscribe(ctx, api.RedisMarketDataChannel) // Attempt to resubscribe
				if pubsub == nil {
					zap.L().Fatal("Failed to resubscribe to Redis PubSub for candle generation. Exiting.")
					return // Fatal error, exit the goroutine
				}
				ch = pubsub.Channel() // Get the new channel from the resubscribed client
				zap.L().Info("Successfully reconnected to Redis PubSub for candle generation.")
				continue // Continue to the next iteration to process messages from the new channel
			}

			// Unmarshal the incoming Redis message payload.
			// The payload is expected to be an enriched tick, which contains the original Kite tick.
			var enrichedTick struct {
				Symbol           string          `json:"symbol"`
				ProcessedAtNanos int64           `json:"processed_at_nanos"`
				Tick             json.RawMessage `json:"tick"` // `json.RawMessage` to defer unmarshalling of the nested tick
			}
			if err := json.Unmarshal([]byte(msg.Payload), &enrichedTick); err != nil {
				zap.L().Error("Failed to unmarshal Redis message payload outer structure for candle generation",
					zap.Error(err),
					zap.String("payload_sample", string(msg.Payload[:min(len(msg.Payload), 200)]))) // Log a sample for debugging
				continue
			}

			// Unmarshal the actual Kite tick from the raw message.
			var kiteTick kitemodels.Tick
			if err := json.Unmarshal(enrichedTick.Tick, &kiteTick); err != nil {
				zap.L().Error("Failed to unmarshal raw tick for candle generation",
					zap.Error(err),
					zap.String("tick_payload_sample", string(enrichedTick.Tick[:min(len(enrichedTick.Tick), 100)])))
				continue
			}

			// Process the unmarshalled tick to update/create candles.
			cg.processTickForCandles(kiteTick)

		case <-ctx.Done():
			// Context cancelled, indicating a graceful shutdown request.
			zap.L().Info("Context cancelled, stopping candle generator Redis subscriber.")
			// On graceful shutdown, flush any remaining open candles to prevent data loss.
			cg.flushAllOpenCandles()
			return // Exit the goroutine
		}
	}
}

// processTickForCandles processes an incoming market data tick to update or create
// OHLCV candles for all configured time intervals.
// This function is now fully protected by `cg.openCandlesMu` to ensure thread safety
// across all map accesses and modifications.
func (cg *CandleGenerator) processTickForCandles(tick kitemodels.Tick) {
	instrumentToken := tick.InstrumentToken
	tickTime := tick.Timestamp.Time // This is the exchange's timestamp for the tick

	// Acquire a write lock on the top-level map for the entire duration of iterating through intervals
	// and modifying the `openCandles` map (or its nested maps).
	cg.openCandlesMu.Lock()
	defer cg.openCandlesMu.Unlock() // Ensure this lock is released at function exit

	instrumentCandles, ok := cg.openCandles[instrumentToken]
	if !ok {
		// If the instrument's candle map doesn't exist, create it.
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

		// Calculate the precise start time for the candle to which this tick belongs.
		// Truncate ensures the timestamp aligns to the beginning of the interval
		// (e.g., 10:00:30 with 1m interval becomes 10:00:00).
		candleStartTime := tickTime.Truncate(intervalDuration)

		// Get the current "open" candle for this instrument and interval.
		currentCandle, candleExists := instrumentCandles[intervalStr]

		// Check if a new candle period has started or if this is the very first tick for this interval.
		if !candleExists || currentCandle.Timestamp.Before(candleStartTime) {
			// A new candle period has started.
			// If an old candle existed for this interval, flush it *before* creating the new one.
			if candleExists && currentCandle.Timestamp.Before(candleStartTime) {
				// The old candle is now complete. Flush it.
				// Make a copy of the completed candle data. This is crucial because
				// `flushCandle` will acquire `currentCandle.mu`. Copying ensures that
				// the state being flushed is exactly what was completed.
				tempCandleToFlush := *currentCandle // Copy the value to ensure thread safety during flush
				cg.flushCandle(&tempCandleToFlush)  // Flush the copy to the database
				zap.L().Debug("Flushed completed candle",
					zap.Uint32("token", tempCandleToFlush.InstrumentToken),
					zap.String("interval", tempCandleToFlush.Interval),
					zap.Time("timestamp", tempCandleToFlush.Timestamp),
					zap.Float64("close", tempCandleToFlush.Close))
			}

			// Create the new candle for the current interval.
			newCandle := &CandleData{
				InstrumentToken: instrumentToken,
				Interval:        intervalStr,
				Timestamp:       candleStartTime,
				Open:            tick.LastPrice,
				High:            tick.LastPrice, // Start with current price, will update with true max
				Low:             tick.LastPrice, // Start with current price, will update with true min
				Close:           tick.LastPrice,
				Volume:          float64(tick.LastTradedQuantity),
				TradeCount:      1,
				LastTickTime:    tickTime,
			}
			instrumentCandles[intervalStr] = newCandle // Store the newly created candle
			zap.L().Debug("Created new candle",
				zap.Uint32("token", instrumentToken),
				zap.String("interval", intervalStr),
				zap.Time("timestamp", candleStartTime),
				zap.Float64("open", tick.LastPrice),
				zap.Time("tick_time", tickTime))

		} else {
			// The tick belongs to the currently open candle. Update its High, Low, Close, Volume, and TradeCount.
			// Since `cg.openCandlesMu` is held, this block is already protected from concurrent access to
			// `currentCandle` from other `processTickForCandles` calls for the same instrument/interval.
			// Thus, the `currentCandle.mu` is not strictly necessary *within this update logic*,
			// but it remains crucial for `flushCandle` or any external readers/writers.
			// The individual `currentCandle.mu` is not acquired/released here to avoid redundant locking
			// while `cg.openCandlesMu` already provides protection.

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
				zap.Time("timestamp", currentCandle.Timestamp),
				zap.Float64("close", currentCandle.Close),
				zap.Float64("volume", currentCandle.Volume))
		}
	}
}

// flushCandle saves a completed CandleData to the database.
// It uses GORM's `OnConflict` clause to perform an "upsert" operation:
// if a candle with the same primary key (InstrumentToken, Interval, Timestamp) exists,
// it updates the existing record; otherwise, it inserts a new one.
// This function is designed to be called with a copy of CandleData or a pointer
// to CandleData that is NOT being actively modified by other goroutines.
// It acquires `cd.mu` to protect its internal fields during the DB operation.
func (cg *CandleGenerator) flushCandle(cd *CandleData) {
	if cd == nil {
		zap.L().Debug("Attempted to flush a nil candle, skipping.")
		return
	}

	// Acquire the individual candle's mutex to protect its data during database serialization and writing.
	cd.mu.Lock()
	defer cd.mu.Unlock()

	// Convert the internal CandleData struct to the database model db.OHLCVCandle.
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
		// GORM will automatically handle `CreatedAt` and `UpdatedAt` fields.
	}

	// Perform the upsert operation.
	// `Columns` specifies the unique columns that identify a conflict.
	// `DoUpdates` specifies which columns to update if a conflict occurs.
	//
	// FIX: Removed "open" from DoUpdates. The Open price should be fixed once the candle starts,
	// and not updated on subsequent conflicts.
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
		// This can happen if the record already exists and no columns were actually changed,
		// or if a conflict occurred but `DoNothing` was implied (which is not the case here due to `DoUpdates`).
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

		// NEW: Broadcast the candle after successful DB operation
		cg.broadcastCandle(cd)
	}
}

// broadcastCandle marshals and sends the candle data to all connected WebSocket clients.
func (cg *CandleGenerator) broadcastCandle(cd *CandleData) {
	// We only broadcast data that is safe for public consumption.
	// The internal CandleData `mu` is held by `flushCandle` at this point.
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

	// Iterate through connected WebSocket clients and send the message
	cg.candleWsClients.Range(func(key, value interface{}) bool {
		conn, ok := value.(*websocket.Conn)
		if !ok {
			zap.L().Error("Invalid type in candleWsClients map for key", zap.Any("key", key))
			// Remove bad entry
			cg.candleWsClients.Delete(key)
			return true // continue iteration
		}

		if err := conn.WriteMessage(websocket.TextMessage, jsonMessage); err != nil {
			zap.L().Error("Failed to write candle message to WebSocket client, removing client",
				zap.Error(err),
				zap.String("remote_addr", conn.RemoteAddr().String()),
				zap.Uint32("token", cd.InstrumentToken))
			cg.candleWsClients.Delete(key) // Remove the client if write fails
		}
		return true // continue iteration
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
		// For "1d" (one day) candles, direct `time.Duration` might be problematic
		// due to daylight saving changes. However, for simple truncation, 24 hours is a common proxy.
		// For strict daily candles, a more sophisticated day-boundary detection might be needed.
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
	cg.openCandlesMu.Lock()         // Acquire the global lock to ensure no other goroutine is modifying openCandles
	defer cg.openCandlesMu.Unlock() // Release the lock when function exits

	// Iterate through instrument tokens
	for _, instrumentCandles := range cg.openCandles {
		// Iterate through intervals for each instrument
		for _, candle := range instrumentCandles {
			// Make a copy to flush safely, as cg.openCandlesMu is held.
			// This ensures the data is stable for the DB operation.
			tempCandleToFlush := *candle
			cg.flushCandle(&tempCandleToFlush)
		}
	}
	zap.L().Info("All open candles flushed successfully.")
}
