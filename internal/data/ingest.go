// internal/data/ingest.go
package data

import (
	"context"
	"encoding/json"
	"sync"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/cache" // ADDED: Import config package
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/utils" // Import your config package
	"github.com/gorilla/websocket"                // For broadcasting to WebSockets
	kitemodels "github.com/zerodha/gokiteconnect/v4/models"
	"go.uber.org/zap"
	"gorm.io/gorm/clause"
)

// MarketDataIngestor holds dependencies for market data ingestion and broadcasting.
type MarketDataIngestor struct {
	dbClient    *db.DBClient
	redisClient *cache.RedisClient
	wsClients   *sync.Map // Use sync.Map for concurrent access to WebSocket clients
	// Buffer for batching market data
	marketDataBuffer []db.MarketData
	bufferLock       sync.Mutex
	lastFlushTime    time.Time
	// NEW: For managing tick_sequence_id
	// Stores the next available sequence ID for a given (instrument_token, actual_tick_timestamp_from_broker)
	// map[instrument_token]map[time.Time]next_sequence_id  <-- Key for inner map is full precision time.Time
	tickSequenceCounters map[uint]map[time.Time]int
	sequenceMux          sync.Mutex // Protects tickSequenceCounters

	// NEW: To manage cleanup of old sequence counters
	lastCleanupTime time.Time // Keep track of when last cleanup happened

	broadcastChannel chan []byte // Channel for sending data to WebSocket handler
	livePrices       sync.Map    // Store latest prices (JSON []byte) for new WS connections
	// ADDED: Configuration for ingestion
	cfg *utils.AppConfig // Configuration for the ingestor, if needed
}

// NewMarketDataIngestor creates and returns a new instance of MarketDataIngestor.
func NewMarketDataIngestor(dbC *db.DBClient, rC *cache.RedisClient, wsClients *sync.Map, cfg *utils.AppConfig) *MarketDataIngestor {
	ingestor := &MarketDataIngestor{
		dbClient:             dbC,
		redisClient:          rC,
		wsClients:            wsClients,
		marketDataBuffer:     make([]db.MarketData, 0, cfg.Ingestion.MarketDataBatchSize),
		lastFlushTime:        time.Now(),
		tickSequenceCounters: make(map[uint]map[time.Time]int), // Initialize the map
		lastCleanupTime:      time.Now(),
		broadcastChannel:     make(chan []byte, 1000), // Buffered channel to avoid blocking
		livePrices:           sync.Map{},
		cfg:                  cfg,
	}
	// Initialize tick sequence counters from the database on startup
	ingestor.loadInitialTickSequenceCounters()
	return ingestor
}

// StartIngestionAndBroadcast kicks off the Redis subscription, DB ingestion, and WebSocket broadcasting.
func (m *MarketDataIngestor) StartIngestionAndBroadcast(ctx context.Context) {
	// Goroutine to subscribe to Redis and process ticks
	go m.subscribeAndProcessRedis(ctx)

	// Goroutine to handle batching and flushing to DB
	go m.startDBFlusher(ctx)

	// NEW: Goroutine to periodically clean up old sequence counters
	go m.startSequenceCounterCleanup(ctx)

	// Goroutine to handle broadcasting to WebSocket clients
	go m.startWebSocketBroadcaster(ctx)

	zap.L().Info("🚀 Market data ingestion and broadcasting started.")
}

// subscribeAndProcessRedis subscribes to the Redis market data channel and unmarshals ticks.
func (m *MarketDataIngestor) subscribeAndProcessRedis(ctx context.Context) {
	pubsub := m.redisClient.Subscribe(ctx, api.RedisMarketDataChannel)
	defer func() {
		if err := pubsub.Close(); err != nil {
			zap.L().Error("Failed to close Redis PubSub connection", zap.Error(err))
		}
		zap.L().Info("Redis PubSub subscriber closed.")
	}()

	zap.L().Info("✅ Subscribed to Redis market data channel", zap.String("channel", api.RedisMarketDataChannel))

	ch := pubsub.Channel()
	for {
		select {
		case msg, ok := <-ch:
			if !ok {
				zap.L().Warn("Redis PubSub channel closed, attempting reconnect in 5 seconds...")
				time.Sleep(5 * time.Second)
				pubsub = m.redisClient.Subscribe(ctx, api.RedisMarketDataChannel)
				if pubsub == nil {
					zap.L().Fatal("Failed to resubscribe to Redis PubSub, exiting.")
					return
				}
				ch = pubsub.Channel()
				zap.L().Info("Successfully reconnected to Redis PubSub.")
				continue
			}

			// --- FIX FOR "unknown time format" STARTS HERE ---
			var rawPayload map[string]json.RawMessage
			if err := json.Unmarshal([]byte(msg.Payload), &rawPayload); err != nil {
				zap.L().Error("Failed to unmarshal Redis message payload into raw map", zap.Error(err), zap.String("payload_sample", string(msg.Payload[:min(len(msg.Payload), 200)])))
				continue
			}

			var enrichedTick struct {
				Symbol           string `json:"symbol"`
				ProcessedAtNanos int64  `json:"processed_at_nanos"`
				// We will unmarshal 'Tick' manually to handle the Timestamp
				Tick json.RawMessage `json:"tick"`
			}
			// Unmarshal the outer structure first
			if err := json.Unmarshal([]byte(msg.Payload), &enrichedTick); err != nil {
				zap.L().Error("Failed to unmarshal Redis message payload outer structure", zap.Error(err), zap.String("payload_sample", string(msg.Payload[:min(len(msg.Payload), 200)])))
				continue
			}

			// Now, unmarshal the 'Tick' field separately to parse its Timestamp
			var kiteTick kitemodels.Tick
			if err := json.Unmarshal(enrichedTick.Tick, &kiteTick); err != nil {
				// This is where the time format error likely originates.
				// Instead of failing here, we'll try to parse the timestamp manually if `kiteTick` fails to parse it.
				var tempTick map[string]interface{}
				if errTemp := json.Unmarshal(enrichedTick.Tick, &tempTick); errTemp != nil {
					zap.L().Error("Failed to unmarshal raw tick for manual timestamp parsing", zap.Error(errTemp), zap.String("tick_payload_sample", string(enrichedTick.Tick[:min(len(enrichedTick.Tick), 100)])))
					continue
				}

				timestampStr, ok := tempTick["Timestamp"].(string)
				if !ok {
					zap.L().Error("Timestamp field not found or not a string in tick payload", zap.String("tick_payload_sample", string(enrichedTick.Tick[:min(len(enrichedTick.Tick), 100)])))
					continue
				}

				parsedTime, errParse := time.Parse(time.RFC3339Nano, timestampStr) // RFC3339Nano is safer for various precision
				if errParse != nil {
					// Fallback to simpler RFC3339 if Nano fails
					parsedTime, errParse = time.Parse(time.RFC3339, timestampStr)
					if errParse != nil {
						zap.L().Error("Failed to parse Timestamp from tick payload with RFC3339/RFC3339Nano",
							zap.Error(errParse),
							zap.String("timestamp_string", timestampStr),
							zap.String("payload_sample", string(msg.Payload[:min(len(msg.Payload), 200)])))
						continue
					}
				}
				kiteTick.Timestamp = kitemodels.Time{Time: parsedTime} // Assign the manually parsed time as models.Time
				// Now, manually unmarshal other fields into kiteTick if you need them and they are not affected by timestamp issue.
				// This part assumes that only Timestamp parsing is the problem for kitemodels.Tick.
				// If other fields also fail, you might need a custom struct to mirror kitemodels.Tick and unmarshal everything manually.
				// For now, let's assume `kiteTick` has its other fields correctly populated after the first `json.Unmarshal(enrichedTick.Tick, &kiteTick)`
				// and only its `Timestamp` is problematic for it. If `json.Unmarshal(enrichedTick.Tick, &kiteTick)` fails entirely,
				// you might need to copy over all scalar fields from `tempTick` to `kiteTick`
				// This is more complex, so let's try the simpler approach first.
			}

			// Now that kiteTick has its Timestamp parsed (either by default unmarshal or manually),
			// create a new struct to pass to processTick with the correctly unmarshaled kitemodels.Tick
			processedEnrichedTick := struct {
				Symbol           string
				ProcessedAtNanos int64
				Tick             kitemodels.Tick
			}{
				Symbol:           enrichedTick.Symbol,
				ProcessedAtNanos: enrichedTick.ProcessedAtNanos,
				Tick:             kiteTick,
			}

			m.processTick(processedEnrichedTick)
			// --- FIX FOR "unknown time format" ENDS HERE ---

		case <-ctx.Done():
			zap.L().Info("Context cancelled, stopping Redis PubSub subscriber.")
			return
		}
	}
}

// min helper function for payload logging
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// processTick converts the enriched tick to MarketData and adds it to the buffer.
func (m *MarketDataIngestor) processTick(enrichedTick struct {
	Symbol           string
	ProcessedAtNanos int64
	Tick             kitemodels.Tick
}) {
	tick := enrichedTick.Tick

	// --- ADDED: Generate tick_sequence_id ---
	m.sequenceMux.Lock()
	defer m.sequenceMux.Unlock()

	// Ensure the inner map exists for this instrument
	if _, ok := m.tickSequenceCounters[uint(tick.InstrumentToken)]; !ok {
		m.tickSequenceCounters[uint(tick.InstrumentToken)] = make(map[time.Time]int)
	}

	// Get the last sequence ID for this instrument and timestamp
	// Preserve the full precision of the timestamp for the map key and DB storage.
	// This ensures consistency with how timestamps are loaded from the database.
	normalizedTimestamp := tick.Timestamp.Time // REMOVED .Truncate(time.Second)

	currentSequenceID := m.tickSequenceCounters[uint(tick.InstrumentToken)][normalizedTimestamp] + 1
	m.tickSequenceCounters[uint(tick.InstrumentToken)][normalizedTimestamp] = currentSequenceID
	// --- END ADDED: Generate tick_sequence_id ---

	// Convert Kite tick to your MarketData model
	// Ensure InstrumentToken exists in your db.Instrument table before inserting MarketData
	// Convert Kite tick to your MarketData model
	md := db.MarketData{
		InstrumentToken:    tick.InstrumentToken, // UPDATED: Directly assign uint32
		Timestamp:          normalizedTimestamp,
		TickSequenceID:     currentSequenceID,
		LastPrice:          tick.LastPrice,
		LastTradedQuantity: tick.LastTradedQuantity, // ADDED: From kitemodels.Tick
		Volume:             tick.VolumeTraded,       // UPDATED: Maps to VolumeTraded
		AverageTradePrice:  tick.AverageTradePrice,  // ADDED: From kitemodels.Tick
		NetChange:          tick.NetChange,          // ADDED: From kitemodels.Tick
		Open:               tick.OHLC.Open,
		High:               tick.OHLC.High,
		Low:                tick.OHLC.Low,
		Close:              tick.OHLC.Close,
		OpenInterest:       tick.OI, // ADDED: From kitemodels.Tick

		// --- ADDED: Market Depth Fields (from kitemodels.Tick.Depth) ---
		BidPrice1:    tick.Depth.Buy[0].Price,
		BidQuantity1: tick.Depth.Buy[0].Quantity,
		BidOrders1:   tick.Depth.Buy[0].Orders,
		BidPrice2:    tick.Depth.Buy[1].Price,
		BidQuantity2: tick.Depth.Buy[1].Quantity,
		BidOrders2:   tick.Depth.Buy[1].Orders,
		BidPrice3:    tick.Depth.Buy[2].Price,
		BidQuantity3: tick.Depth.Buy[2].Quantity,
		BidOrders3:   tick.Depth.Buy[2].Orders,
		BidPrice4:    tick.Depth.Buy[3].Price,
		BidQuantity4: tick.Depth.Buy[3].Quantity,
		BidOrders4:   tick.Depth.Buy[3].Orders,
		BidPrice5:    tick.Depth.Buy[4].Price,
		BidQuantity5: tick.Depth.Buy[4].Quantity,
		BidOrders5:   tick.Depth.Buy[4].Orders,

		AskPrice1:    tick.Depth.Sell[0].Price,
		AskQuantity1: tick.Depth.Sell[0].Quantity,
		AskOrders1:   tick.Depth.Sell[0].Orders,
		AskPrice2:    tick.Depth.Sell[1].Price,
		AskQuantity2: tick.Depth.Sell[1].Quantity,
		AskOrders2:   tick.Depth.Sell[1].Orders,
		AskPrice3:    tick.Depth.Sell[2].Price,
		AskQuantity3: tick.Depth.Sell[2].Quantity,
		AskOrders3:   tick.Depth.Sell[2].Orders,
		AskPrice4:    tick.Depth.Sell[3].Price,
		AskQuantity4: tick.Depth.Sell[3].Quantity,
		AskOrders4:   tick.Depth.Sell[3].Orders,
		AskPrice5:    tick.Depth.Sell[4].Price,
		AskQuantity5: tick.Depth.Sell[4].Quantity,
		AskOrders5:   tick.Depth.Sell[4].Orders,

		TotalBuyQuantity:  tick.TotalBuyQuantity,  // ADDED: From kitemodels.Tick
		TotalSellQuantity: tick.TotalSellQuantity, // ADDED: From kitemodels.Tick
	}

	m.bufferLock.Lock()
	m.marketDataBuffer = append(m.marketDataBuffer, md)
	// UPDATED: Use config value for MarketDataBatchSize
	if len(m.marketDataBuffer) >= m.cfg.Ingestion.MarketDataBatchSize {
		m.flushBuffer()
	}
	m.bufferLock.Unlock()

	// Push the data for frontend broadcast
	frontendData, err := json.Marshal(map[string]interface{}{
		"symbol": enrichedTick.Symbol,
		"tick":   tick, // Send the raw tick structure to frontend
	})
	if err != nil {
		zap.L().Error("Failed to marshal data for frontend broadcast", zap.Error(err))
	} else {
		select {
		case m.broadcastChannel <- frontendData:
			// Store the latest price for this symbol for new WebSocket connections
			m.livePrices.Store(enrichedTick.Symbol, frontendData)
		default:
			zap.L().Warn("Dropping WebSocket broadcast message: channel is full. Consider increasing buffer size.")
		}
	}
}

// startDBFlusher periodically flushes the buffered market data to the database.
func (m *MarketDataIngestor) startDBFlusher(ctx context.Context) {
	flushInterval := time.Duration(m.cfg.Ingestion.MarketDataFlushIntervalMS) * time.Millisecond
	if flushInterval <= 0 {
		zap.L().Fatal("MarketDataFlushIntervalMS must be a positive duration in app.yaml", zap.Int("MarketDataFlushIntervalMS", m.cfg.Ingestion.MarketDataFlushIntervalMS))
	}
	ticker := time.NewTicker(flushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if time.Since(m.lastFlushTime) >= flushInterval {
				m.flushBuffer()
				m.lastFlushTime = time.Now()
			}
		case <-ctx.Done():
			zap.L().Info("Context cancelled, flushing remaining buffer and stopping DB flusher.")
			m.flushBuffer() // Flush any remaining data before exiting
			return
		}
	}
}

// flushBuffer performs a batch insert of the buffered market data into the database.
func (m *MarketDataIngestor) flushBuffer() {
	m.bufferLock.Lock()
	defer m.bufferLock.Unlock()

	if len(m.marketDataBuffer) == 0 {
		return
	}

	// Take a copy of the buffer to allow new ticks to be added while flushing
	dataToFlush := m.marketDataBuffer
	m.marketDataBuffer = make([]db.MarketData, 0, m.cfg.Ingestion.MarketDataBatchSize) // Reset buffer immediately

	zap.L().Debug("Flushing market data buffer to DB", zap.Int("count", len(dataToFlush)))

	// Perform batch insert using GORM's CreateInBatches.
	// We rely on the `Instrument` table being pre-populated.
	result := m.dbClient.DB.Clauses(clause.OnConflict{
		// On conflict for the primary key (instrument_token, timestamp, tick_sequence_id),
		// do nothing. This means existing duplicates are simply ignored.
		Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "timestamp"}, {Name: "tick_sequence_id"}},
		DoNothing: true,
	}).CreateInBatches(dataToFlush, m.cfg.Ingestion.MarketDataBatchSize)

	if result.Error != nil {
		zap.L().Error("❌ Failed to batch insert market data with ON CONFLICT DO NOTHING", zap.Error(result.Error), zap.Int("batch_size", len(dataToFlush)))
	} else {
		skippedCount := len(dataToFlush) - int(result.RowsAffected) // Cast to int for arithmetic
		if skippedCount > 0 {
			zap.L().Warn("⚠️ Flushed market data to DB with skipped duplicates",
				zap.Int("total_attempted", len(dataToFlush)),
				zap.Int64("rows_inserted", result.RowsAffected),
				zap.Int("rows_skipped", skippedCount))
		} else {
			zap.L().Info("✅ Successfully flushed market data to DB", zap.Int64("count", result.RowsAffected))
		}
	}
}

// startSequenceCounterCleanup periodically cleans up old entries from tickSequenceCounters.
func (m *MarketDataIngestor) startSequenceCounterCleanup(ctx context.Context) {
	ticker := time.NewTicker(time.Duration(m.cfg.Ingestion.TickSequenceCleanupInterval) * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			m.cleanupOldSequenceCounters()
		case <-ctx.Done():
			zap.L().Info("Context cancelled, stopping sequence counter cleanup.")
			return
		}
	}
}

// cleanupOldSequenceCounters iterates through tickSequenceCounters and removes old timestamps.
func (m *MarketDataIngestor) cleanupOldSequenceCounters() {
	m.sequenceMux.Lock()
	defer m.sequenceMux.Unlock()

	now := time.Now()
	for instToken, timestampMap := range m.tickSequenceCounters {
		for ts := range timestampMap { // `ts` will be the full precision time.Time
			// If the timestamp is older than our cleanup duration, delete it
			if now.Sub(ts) > time.Duration(m.cfg.Ingestion.MaxTickSequenceCacheDuration)*time.Second {
				delete(timestampMap, ts)
			}
		}
		// If an instrument has no more active timestamps, delete its entry to free memory
		if len(timestampMap) == 0 {
			delete(m.tickSequenceCounters, instToken)
		}
	}
	zap.L().Debug("Cleaned up old tick sequence counters", zap.Duration("duration", time.Duration(m.cfg.Ingestion.MaxTickSequenceCacheDuration)*time.Second))
}

// startWebSocketBroadcaster listens on the broadcastChannel and sends messages to connected WS clients.
func (m *MarketDataIngestor) startWebSocketBroadcaster(ctx context.Context) {
	for {
		select {
		case msg := <-m.broadcastChannel:
			m.wsClients.Range(func(key, value interface{}) bool {
				conn, ok := key.(*websocket.Conn)
				if !ok {
					m.wsClients.Delete(key) // Clean up invalid entries
					return true
				}
				err := conn.WriteMessage(websocket.TextMessage, msg)
				if err != nil {
					if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
						zap.L().Debug("WebSocket write error (unexpected close), removing client", zap.Error(err), zap.String("remote_addr", conn.RemoteAddr().String()))
					} else {
						zap.L().Info("WebSocket write error (graceful close/other), removing client", zap.Error(err), zap.String("remote_addr", conn.RemoteAddr().String()))
					}
					conn.Close()
					m.wsClients.Delete(key)
				}
				return true // Continue iteration
			})
		case <-ctx.Done():
			zap.L().Info("Context cancelled, stopping WebSocket broadcaster.")
			return
		}
	}
}

// RegisterWebSocketClient adds a new WebSocket client for broadcasting.
func (m *MarketDataIngestor) RegisterWebSocketClient(conn *websocket.Conn) {
	m.wsClients.Store(conn, true)
	zap.L().Info("🧑‍💻 New WebSocket client connected", zap.String("remote_addr", conn.RemoteAddr().String()))

	// Send initial live prices to the newly connected client
	m.livePrices.Range(func(key, value interface{}) bool {
		symbol := key.(string)
		data := value.([]byte)
		err := conn.WriteMessage(websocket.TextMessage, data)
		if err != nil {
			zap.L().Warn("Failed to send initial live price to new WS client",
				zap.String("symbol", symbol),
				zap.String("remote_addr", conn.RemoteAddr().String()),
				zap.Error(err))
			// If sending initial data fails, no need to keep this client
			conn.Close()
			m.wsClients.Delete(conn)
			return false // Stop iterating for this client
		}
		return true // Continue to next live price
	})
}

// UnregisterWebSocketClient removes a WebSocket client.
func (m *MarketDataIngestor) UnregisterWebSocketClient(conn *websocket.Conn) {
	m.wsClients.Delete(conn)
	conn.Close() // Ensure the connection is closed
	zap.L().Info("🔌 WebSocket client disconnected", zap.String("remote_addr", conn.RemoteAddr().String()))
}

// loadInitialTickSequenceCounters loads the max tick sequence IDs from DB on startup.
// loadInitialTickSequenceCounters loads the max tick sequence IDs from DB on startup.
func (m *MarketDataIngestor) loadInitialTickSequenceCounters() {
	m.sequenceMux.Lock()
	defer m.sequenceMux.Unlock()

	type Result struct {
		InstrumentToken uint32    `gorm:"column:instrument_token"`
		Timestamp       time.Time `gorm:"column:timestamp"` // Will receive the full precision from DB
		MaxSequenceID   int       `gorm:"column:max_sequence_id"`
	}

	var results []Result

	loc, err := time.LoadLocation("Asia/Kolkata")
	if err != nil {
		zap.L().Error("Failed to load Asia/Kolkata location, using UTC.", zap.Error(err))
		loc = time.UTC
	}

	now := time.Now().In(loc)
	todayStart := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, loc)

	query := `
        SELECT instrument_token, timestamp, MAX(tick_sequence_id) as max_sequence_id
        FROM market_data
        WHERE timestamp >= ?
        GROUP BY instrument_token, timestamp;
    `
	err = m.dbClient.DB.Raw(query, todayStart).Scan(&results).Error
	if err != nil {
		zap.L().Error("❌ Failed to load initial tick sequence counters from DB", zap.Error(err))
		return
	}

	for _, r := range results {
		if _, ok := m.tickSequenceCounters[uint(r.InstrumentToken)]; !ok {
			m.tickSequenceCounters[uint(r.InstrumentToken)] = make(map[time.Time]int)
		}
		// DO NOT TRUNCATE HERE if you want to preserve full precision.
		// The key in the map MUST match the precision stored in the DB and generated by processTickMessage.
		m.tickSequenceCounters[uint(r.InstrumentToken)][r.Timestamp] = r.MaxSequenceID
	}
	zap.L().Info("✅ Loaded initial tick sequence counters from DB", zap.Int("count", len(results)), zap.String("from_timestamp", todayStart.String()))
}
