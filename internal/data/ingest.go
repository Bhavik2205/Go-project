// package data

// import (
// 	"context"
// 	"encoding/json"
// 	"sync"
// 	"time"

// 	"github.com/Bhavik2205/ML-Bot/internal/api"
// 	"github.com/Bhavik2205/ML-Bot/internal/cache"
// 	"github.com/Bhavik2205/ML-Bot/internal/db"
// 	"github.com/Bhavik2205/ML-Bot/internal/utils"
// 	"github.com/gorilla/websocket"
// 	kitemodels "github.com/zerodha/gokiteconnect/v4/models"
// 	"go.uber.org/zap"
// 	"gorm.io/gorm/clause"
// )

// // MarketDataIngestor holds dependencies for market data ingestion and broadcasting.
// type MarketDataIngestor struct {
// 	dbClient             *db.DBClient
// 	redisClient          *cache.RedisClient
// 	wsClients            *sync.Map
// 	marketDataBuffer     []db.MarketData
// 	bufferLock           sync.Mutex
// 	lastFlushTime        time.Time
// 	tickSequenceCounters map[uint]map[time.Time]int
// 	sequenceMux          sync.Mutex
// 	lastCleanupTime      time.Time
// 	broadcastChannel     chan []byte // Channel for sending data to WebSocket broadcasters
// 	livePrices           sync.Map
// 	cfg                  *utils.AppConfig

// 	// New: Channel for sending buffered market data to DB workers
// 	dbFlushCh chan []db.MarketData
// 	// New: Number of goroutines to handle DB writes
// 	dbWorkerCount int
// 	// New: Number of goroutines to handle WebSocket broadcasting
// 	wsBroadcastWorkerCount int
// }

// // NewMarketDataIngestor creates and returns a new instance of MarketDataIngestor.
// func NewMarketDataIngestor(dbC *db.DBClient, rC *cache.RedisClient, wsClients *sync.Map, cfg *utils.AppConfig, _ *utils.IndicatorsConfig) *MarketDataIngestor {
// 	ingestor := &MarketDataIngestor{
// 		dbClient:             dbC,
// 		redisClient:          rC,
// 		wsClients:            wsClients,
// 		marketDataBuffer:     make([]db.MarketData, 0, cfg.Ingestion.MarketDataBatchSize),
// 		lastFlushTime:        time.Now(),
// 		tickSequenceCounters: make(map[uint]map[time.Time]int),
// 		sequenceMux:          sync.Mutex{},
// 		lastCleanupTime:      time.Now(),
// 		broadcastChannel:     make(chan []byte, 10000), // Increased buffer for high-frequency
// 		livePrices:           sync.Map{},
// 		cfg:                  cfg,

// 		// Initializing new fields
// 		dbFlushCh:              make(chan []db.MarketData, 100), // Buffer for DB batches
// 		dbWorkerCount:          4,                               // Tunable: Number of goroutines for DB writes
// 		wsBroadcastWorkerCount: 8,                               // Tunable: Number of goroutines for WS broadcasts
// 	}
// 	ingestor.loadInitialTickSequenceCounters()
// 	return ingestor
// }

// // StartIngestionAndBroadcast kicks off the Redis subscription, DB ingestion, and WebSocket broadcasting.
// func (m *MarketDataIngestor) StartIngestionAndBroadcast(ctx context.Context) {
// 	// Start workers that handle database flushing
// 	m.startDBWorkers(ctx)
// 	// Start workers that handle WebSocket broadcasting
// 	m.startWebSocketBroadcasterWorkers(ctx)

// 	// These remain as primary consumers/dispatchers
// 	go m.subscribeAndProcessRedis(ctx)
// 	go m.startDBFlusher(ctx) // This flusher will now send batches to dbFlushCh
// 	go m.startSequenceCounterCleanup(ctx)

// 	zap.L().Info("🚀 Market data ingestion and broadcasting started.")
// }

// // subscribeAndProcessRedis subscribes to the Redis market data channel and unmarshals ticks.
// func (m *MarketDataIngestor) subscribeAndProcessRedis(ctx context.Context) {
// 	pubsub := m.redisClient.Subscribe(ctx, api.RedisMarketDataChannel)
// 	defer func() {
// 		if err := pubsub.Close(); err != nil {
// 			zap.L().Error("Failed to close Redis PubSub connection", zap.Error(err))
// 		}
// 		zap.L().Info("Redis PubSub subscriber closed.")
// 	}()

// 	zap.L().Info("✅ Subscribed to Redis market data channel", zap.String("channel", api.RedisMarketDataChannel))

// 	ch := pubsub.Channel()
// 	for {
// 		select {
// 		case msg, ok := <-ch:
// 			if !ok {
// 				zap.L().Warn("Redis PubSub channel closed, attempting reconnect in 5 seconds...")
// 				time.Sleep(5 * time.Second) // Simple reconnect logic
// 				pubsub = m.redisClient.Subscribe(ctx, api.RedisMarketDataChannel)
// 				if pubsub == nil {
// 					zap.L().Fatal("Failed to resubscribe to Redis PubSub, exiting.")
// 					// Close the dbFlushCh and broadcastChannel to signal workers to stop
// 					close(m.dbFlushCh)
// 					close(m.broadcastChannel)
// 					return
// 				}
// 				ch = pubsub.Channel()
// 				zap.L().Info("Successfully reconnected to Redis PubSub.")
// 				continue
// 			}

// 			// Efficiently unmarshal by delaying full tick unmarshaling if possible
// 			// This structure directly maps to what ticker.go publishes
// 			var enrichedTick struct {
// 				Symbol           string          `json:"symbol"`
// 				ProcessedAtNanos int64           `json:"processed_at_nanos"`
// 				Tick             json.RawMessage `json:"tick"` // Keep as RawMessage initially
// 			}
// 			if err := json.Unmarshal([]byte(msg.Payload), &enrichedTick); err != nil {
// 				zap.L().Error("Failed to unmarshal Redis message payload outer structure", zap.Error(err), zap.String("payload_sample", string(msg.Payload[:min(len(msg.Payload), 200)])))
// 				continue
// 			}

// 			var kiteTick kitemodels.Tick
// 			// Only unmarshal the full Kite tick if needed for processing
// 			if err := json.Unmarshal(enrichedTick.Tick, &kiteTick); err != nil {
// 				// Fallback for timestamp parsing if direct unmarshal fails
// 				var tempTick map[string]interface{}
// 				if errTemp := json.Unmarshal(enrichedTick.Tick, &tempTick); errTemp != nil {
// 					zap.L().Error("Failed to unmarshal raw tick for manual timestamp parsing", zap.Error(errTemp), zap.String("tick_payload_sample", string(enrichedTick.Tick[:min(len(enrichedTick.Tick), 100)])))
// 					continue
// 				}
// 				timestampStr, ok := tempTick["Timestamp"].(string)
// 				if !ok {
// 					zap.L().Error("Timestamp field not found or not a string in tick payload", zap.String("tick_payload_sample", string(enrichedTick.Tick[:min(len(enrichedTick.Tick), 100)])))
// 					continue
// 				}
// 				parsedTime, errParse := time.Parse(time.RFC3339Nano, timestampStr)
// 				if errParse != nil {
// 					parsedTime, errParse = time.Parse(time.RFC3339, timestampStr)
// 					if errParse != nil {
// 						zap.L().Error("Failed to parse Timestamp from tick payload with RFC3339/RFC3339Nano",
// 							zap.Error(errParse),
// 							zap.String("timestamp_string", timestampStr),
// 							zap.String("payload_sample", string(msg.Payload[:min(len(msg.Payload), 200)])))
// 						continue
// 					}
// 				}
// 				kiteTick.Timestamp = kitemodels.Time{Time: parsedTime}
// 			}

// 			processedEnrichedTick := struct {
// 				Symbol           string
// 				ProcessedAtNanos int64
// 				Tick             kitemodels.Tick
// 			}{
// 				Symbol:           enrichedTick.Symbol,
// 				ProcessedAtNanos: enrichedTick.ProcessedAtNanos,
// 				Tick:             kiteTick,
// 			}

// 			// This function will now be very fast, primarily appending to a buffer and sending to channels
// 			m.processTick(processedEnrichedTick)

// 		case <-ctx.Done():
// 			zap.L().Info("Context cancelled, stopping Redis PubSub subscriber.")
// 			// Close channels when context is done to signal workers to stop
// 			close(m.dbFlushCh)
// 			close(m.broadcastChannel)
// 			return
// 		}
// 	}
// }

// func min(a, b int) int {
// 	if a < b {
// 		return a
// 	}
// 	return b
// }

// // processTick converts the enriched tick to MarketData and adds it to the buffer.
// // It is now non-blocking for database writes and WebSocket broadcasts.
// func (m *MarketDataIngestor) processTick(enrichedTick struct {
// 	Symbol           string
// 	ProcessedAtNanos int64
// 	Tick             kitemodels.Tick
// }) {
// 	tick := enrichedTick.Tick

// 	// Sequence counter update (still needs mutex, but typically fast)
// 	m.sequenceMux.Lock()
// 	if _, ok := m.tickSequenceCounters[uint(tick.InstrumentToken)]; !ok {
// 		m.tickSequenceCounters[uint(tick.InstrumentToken)] = make(map[time.Time]int)
// 	}
// 	normalizedTimestamp := tick.Timestamp.Time
// 	currentSequenceID := m.tickSequenceCounters[uint(tick.InstrumentToken)][normalizedTimestamp] + 1
// 	m.tickSequenceCounters[uint(tick.InstrumentToken)][normalizedTimestamp] = currentSequenceID
// 	m.sequenceMux.Unlock()

// 	// Prepare MarketData for DB buffer
// 	md := db.MarketData{
// 		InstrumentToken:    tick.InstrumentToken,
// 		Timestamp:          normalizedTimestamp,
// 		TickSequenceID:     currentSequenceID,
// 		LastPrice:          tick.LastPrice,
// 		LastTradedQuantity: tick.LastTradedQuantity,
// 		Volume:             tick.VolumeTraded,
// 		AverageTradePrice:  tick.AverageTradePrice,
// 		NetChange:          tick.NetChange,
// 		Open:               tick.OHLC.Open,
// 		High:               tick.OHLC.High,
// 		Low:                tick.OHLC.Low,
// 		Close:              tick.OHLC.Close,
// 		OpenInterest:       tick.OI,
// 		BidPrice1:          tick.Depth.Buy[0].Price,
// 		BidQuantity1:       tick.Depth.Buy[0].Quantity,
// 		BidOrders1:         tick.Depth.Buy[0].Orders,
// 		BidPrice2:          tick.Depth.Buy[1].Price,
// 		BidQuantity2:       tick.Depth.Buy[1].Quantity,
// 		BidOrders2:         tick.Depth.Buy[1].Orders,
// 		BidPrice3:          tick.Depth.Buy[2].Price,
// 		BidQuantity3:       tick.Depth.Buy[2].Quantity,
// 		BidOrders3:         tick.Depth.Buy[2].Orders,
// 		BidPrice4:          tick.Depth.Buy[3].Price,
// 		BidQuantity4:       tick.Depth.Buy[3].Quantity,
// 		BidOrders4:         tick.Depth.Buy[3].Orders,
// 		BidPrice5:          tick.Depth.Buy[4].Price,
// 		BidQuantity5:       tick.Depth.Buy[4].Quantity,
// 		BidOrders5:         tick.Depth.Buy[4].Orders,
// 		AskPrice1:          tick.Depth.Sell[0].Price,
// 		AskQuantity1:       tick.Depth.Sell[0].Quantity,
// 		AskOrders1:         tick.Depth.Sell[0].Orders,
// 		AskPrice2:          tick.Depth.Sell[1].Price,
// 		AskQuantity2:       tick.Depth.Sell[1].Quantity,
// 		AskOrders2:         tick.Depth.Sell[1].Orders,
// 		AskPrice3:          tick.Depth.Sell[2].Price,
// 		AskQuantity3:       tick.Depth.Sell[2].Quantity,
// 		AskOrders3:         tick.Depth.Sell[2].Orders,
// 		AskPrice4:          tick.Depth.Sell[3].Price,
// 		AskQuantity4:       tick.Depth.Sell[3].Quantity,
// 		AskOrders4:         tick.Depth.Sell[3].Orders,
// 		AskPrice5:          tick.Depth.Sell[4].Price,
// 		AskQuantity5:       tick.Depth.Sell[4].Quantity,
// 		AskOrders5:         tick.Depth.Sell[4].Orders,
// 		TotalBuyQuantity:   tick.TotalBuyQuantity,
// 		TotalSellQuantity:  tick.TotalSellQuantity,
// 	}

// 	// Buffer market data for batch DB insertion
// 	m.bufferLock.Lock()
// 	m.marketDataBuffer = append(m.marketDataBuffer, md)
// 	if len(m.marketDataBuffer) >= m.cfg.Ingestion.MarketDataBatchSize {
// 		// Send a copy of the buffer to the DB flusher channel
// 		dataToFlush := make([]db.MarketData, len(m.marketDataBuffer))
// 		copy(dataToFlush, m.marketDataBuffer)
// 		select {
// 		case m.dbFlushCh <- dataToFlush:
// 			// Successfully sent batch, clear buffer
// 			m.marketDataBuffer = make([]db.MarketData, 0, m.cfg.Ingestion.MarketDataBatchSize)
// 		default:
// 			// If DB channel is full, log and continue. Data might be lost or processed later.
// 			// For HFT, this is a sign of backpressure or overload.
// 			zap.L().Warn("Dropping DB write batch: DB flush channel is full. Consider increasing buffer size or DB worker count.",
// 				zap.Int("batch_size", len(dataToFlush)))
// 			// Keep data in buffer for next flush attempt if channel is full, to avoid data loss.
// 			// Or, for extreme HFT where loss is acceptable for speed, you might clear buffer here.
// 			// Current behavior: keep in buffer, try again on next flush interval.
// 		}
// 	}
// 	m.bufferLock.Unlock()

// 	// Prepare data for real-time WebSocket broadcast
// 	frontendData, err := json.Marshal(map[string]interface{}{
// 		"symbol": enrichedTick.Symbol,
// 		"tick":   tick, // Full tick data for frontend
// 	})
// 	if err != nil {
// 		zap.L().Error("Failed to marshal data for frontend broadcast", zap.Error(err))
// 	} else {
// 		select {
// 		case m.broadcastChannel <- frontendData:
// 			m.livePrices.Store(enrichedTick.Symbol, frontendData) // Store latest price for new WS clients
// 		default:
// 			zap.L().Warn("Dropping WebSocket broadcast message: broadcast channel is full. Consider increasing buffer size or WS worker count.",
// 				zap.String("symbol", enrichedTick.Symbol))
// 		}
// 	}
// }

// // startDBFlusher periodically checks if the buffer needs flushing based on time.
// // It now sends batches to the dbFlushCh.
// func (m *MarketDataIngestor) startDBFlusher(ctx context.Context) {
// 	flushInterval := time.Duration(m.cfg.Ingestion.MarketDataFlushIntervalMS) * time.Millisecond
// 	if flushInterval <= 0 {
// 		zap.L().Fatal("MarketDataFlushIntervalMS must be a positive duration in app.yaml", zap.Int("MarketDataFlushIntervalMS", m.cfg.Ingestion.MarketDataFlushIntervalMS))
// 	}
// 	ticker := time.NewTicker(flushInterval)
// 	defer ticker.Stop()

// 	for {
// 		select {
// 		case <-ticker.C:
// 			m.bufferLock.Lock()
// 			if len(m.marketDataBuffer) > 0 && time.Since(m.lastFlushTime) >= flushInterval {
// 				// Send a copy of the buffer to the DB flusher channel
// 				dataToFlush := make([]db.MarketData, len(m.marketDataBuffer))
// 				copy(dataToFlush, m.marketDataBuffer)
// 				select {
// 				case m.dbFlushCh <- dataToFlush:
// 					// Successfully sent batch, clear buffer
// 					m.marketDataBuffer = make([]db.MarketData, 0, m.cfg.Ingestion.MarketDataBatchSize)
// 					m.lastFlushTime = time.Now() // Only update time if batch was successfully sent
// 				default:
// 					// If DB channel is full, log and keep data in buffer for next attempt
// 					zap.L().Warn("DB flush skipped: DB flush channel is full during timed flush. Data will be re-attempted.",
// 						zap.Int("buffered_count", len(dataToFlush)))
// 				}
// 			}
// 			m.bufferLock.Unlock()
// 		case <-ctx.Done():
// 			zap.L().Info("Context cancelled, attempting to flush remaining buffer to DB workers before stopping DB flusher.")
// 			m.bufferLock.Lock()
// 			if len(m.marketDataBuffer) > 0 {
// 				dataToFlush := make([]db.MarketData, len(m.marketDataBuffer))
// 				copy(dataToFlush, m.marketDataBuffer)
// 				select {
// 				case m.dbFlushCh <- dataToFlush:
// 					zap.L().Info("Successfully sent final buffer to DB workers.")
// 				default:
// 					zap.L().Error("Failed to send final buffer to DB workers: channel full. Data might be lost.",
// 						zap.Int("buffered_count", len(dataToFlush)))
// 				}
// 			}
// 			m.bufferLock.Unlock()
// 			return
// 		}
// 	}
// }

// // startDBWorkers starts a pool of goroutines to consume from dbFlushCh and perform batch inserts.
// func (m *MarketDataIngestor) startDBWorkers(ctx context.Context) {
// 	for i := 0; i < m.dbWorkerCount; i++ {
// 		go func(workerID int) {
// 			zap.L().Info("📦 DB worker started", zap.Int("worker_id", workerID))
// 			for {
// 				select {
// 				case dataToFlush, ok := <-m.dbFlushCh:
// 					if !ok {
// 						zap.L().Info("📦 DB flush channel closed, worker stopping", zap.Int("worker_id", workerID))
// 						return
// 					}
// 					// Perform the actual blocking DB insert
// 					result := m.dbClient.DB.Clauses(clause.OnConflict{
// 						Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "timestamp"}, {Name: "tick_sequence_id"}},
// 						DoNothing: true, // Prevents inserting duplicates
// 					}).CreateInBatches(dataToFlush, m.cfg.Ingestion.MarketDataBatchSize) // Batch size is a hint for GORM, actual batch is dataToFlush

// 					if result.Error != nil {
// 						zap.L().Error("❌ DB worker failed to batch insert market data",
// 							zap.Error(result.Error),
// 							zap.Int("worker_id", workerID),
// 							zap.Int("batch_size", len(dataToFlush)))
// 					} else {
// 						skippedCount := len(dataToFlush) - int(result.RowsAffected)
// 						if skippedCount > 0 {
// 							zap.L().Warn("⚠️ DB worker flushed market data with skipped duplicates",
// 								zap.Int("worker_id", workerID),
// 								zap.Int("total_attempted", len(dataToFlush)),
// 								zap.Int64("rows_inserted", result.RowsAffected),
// 								zap.Int("rows_skipped", skippedCount))
// 						} else {
// 							zap.L().Debug("✅ DB worker successfully flushed market data",
// 								zap.Int("worker_id", workerID),
// 								zap.Int64("count", result.RowsAffected))
// 						}
// 					}
// 				case <-ctx.Done():
// 					zap.L().Info("📦 Context cancelled, DB worker stopping", zap.Int("worker_id", workerID))
// 					return
// 				}
// 			}
// 		}(i)
// 	}
// }

// // startSequenceCounterCleanup periodically cleans up old entries from tickSequenceCounters.
// func (m *MarketDataIngestor) startSequenceCounterCleanup(ctx context.Context) {
// 	ticker := time.NewTicker(time.Duration(m.cfg.Ingestion.TickSequenceCleanupInterval) * time.Second)
// 	defer ticker.Stop()

// 	for {
// 		select {
// 		case <-ticker.C:
// 			m.cleanupOldSequenceCounters()
// 		case <-ctx.Done():
// 			zap.L().Info("Context cancelled, stopping sequence counter cleanup.")
// 			return
// 		}
// 	}
// }

// // cleanupOldSequenceCounters iterates through tickSequenceCounters and removes old timestamps.
// func (m *MarketDataIngestor) cleanupOldSequenceCounters() {
// 	m.sequenceMux.Lock()
// 	defer m.sequenceMux.Unlock()

// 	now := time.Now()
// 	for instToken, timestampMap := range m.tickSequenceCounters {
// 		for ts := range timestampMap {
// 			if now.Sub(ts) > time.Duration(m.cfg.Ingestion.MaxTickSequenceCacheDuration)*time.Second {
// 				delete(timestampMap, ts)
// 			}
// 		}
// 		if len(timestampMap) == 0 {
// 			delete(m.tickSequenceCounters, instToken)
// 		}
// 	}
// 	zap.L().Debug("Cleaned up old tick sequence counters", zap.Duration("duration", time.Duration(m.cfg.Ingestion.MaxTickSequenceCacheDuration)*time.Second))
// }

// // startWebSocketBroadcasterWorkers starts a pool of goroutines to consume from broadcastChannel and send to WS clients.
// func (m *MarketDataIngestor) startWebSocketBroadcasterWorkers(ctx context.Context) {
// 	for i := 0; i < m.wsBroadcastWorkerCount; i++ {
// 		go func(workerID int) {
// 			zap.L().Info("🌐 WS broadcaster worker started", zap.Int("worker_id", workerID))
// 			for {
// 				select {
// 				case msg, ok := <-m.broadcastChannel:
// 					if !ok {
// 						zap.L().Info("🌐 WS broadcast channel closed, worker stopping", zap.Int("worker_id", workerID))
// 						return
// 					}
// 					// Each worker iterates over all clients and sends the message
// 					// This is a fan-out pattern within the worker.
// 					// For extreme scale, this 'Range' might still be a bottleneck;
// 					// a per-client goroutine might be needed for very high client counts.
// 					m.wsClients.Range(func(key, value interface{}) bool {
// 						conn, ok := key.(*websocket.Conn)
// 						if !ok {
// 							m.wsClients.Delete(key)
// 							return true
// 						}
// 						// Non-blocking write to WebSocket, but actual network I/O might still block briefly
// 						err := conn.WriteMessage(websocket.TextMessage, msg)
// 						if err != nil {
// 							if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
// 								zap.L().Debug("WebSocket write error (unexpected close), removing client", zap.Error(err), zap.String("remote_addr", conn.RemoteAddr().String()))
// 							} else {
// 								zap.L().Info("WebSocket write error (graceful close/other), removing client", zap.Error(err), zap.String("remote_addr", conn.RemoteAddr().String()))
// 							}
// 							conn.Close()
// 							m.wsClients.Delete(key)
// 							return true // Continue iterating other clients
// 						}
// 						return true // Continue iterating other clients
// 					})
// 				case <-ctx.Done():
// 					zap.L().Info("🌐 Context cancelled, WS broadcaster worker stopping", zap.Int("worker_id", workerID))
// 					return
// 				}
// 			}
// 		}(i)
// 	}
// }

// // RegisterWebSocketClient adds a new WebSocket client for broadcasting.
// func (m *MarketDataIngestor) RegisterWebSocketClient(conn *websocket.Conn) {
// 	m.wsClients.Store(conn, true)
// 	zap.L().Info("🧑‍💻 New WebSocket client connected", zap.String("remote_addr", conn.RemoteAddr().String()))

// 	// Send initial live prices to the newly connected client
// 	m.livePrices.Range(func(key, value interface{}) bool {
// 		// key is symbol (string), value is []byte (marshaled tick)
// 		data := value.([]byte)
// 		err := conn.WriteMessage(websocket.TextMessage, data)
// 		if err != nil {
// 			zap.L().Warn("Failed to send initial live price to new WS client",
// 				zap.String("symbol", key.(string)),
// 				zap.String("remote_addr", conn.RemoteAddr().String()),
// 				zap.Error(err))
// 			// If sending initial data fails, close the connection and remove it.
// 			conn.Close()
// 			m.wsClients.Delete(conn)
// 			return false // Stop iterating
// 		}
// 		return true // Continue iterating
// 	})
// }

// // UnregisterWebSocketClient removes a WebSocket client.
// func (m *MarketDataIngestor) UnregisterWebSocketClient(conn *websocket.Conn) {
// 	m.wsClients.Delete(conn)
// 	conn.Close() // Ensure connection is closed
// 	zap.L().Info("🔌 WebSocket client disconnected", zap.String("remote_addr", conn.RemoteAddr().String()))
// }

// // loadInitialTickSequenceCounters loads the max tick sequence IDs from DB on startup.
// func (m *MarketDataIngestor) loadInitialTickSequenceCounters() {
// 	m.sequenceMux.Lock()
// 	defer m.sequenceMux.Unlock()

// 	type Result struct {
// 		InstrumentToken uint32    `gorm:"column:instrument_token"`
// 		Timestamp       time.Time `gorm:"column:timestamp"`
// 		MaxSequenceID   int       `gorm:"column:max_sequence_id"`
// 	}

// 	var results []Result

// 	loc, err := time.LoadLocation("Asia/Kolkata")
// 	if err != nil {
// 		zap.L().Error("Failed to load Asia/Kolkata location, using UTC.", zap.Error(err))
// 		loc = time.UTC
// 	}

// 	now := time.Now().In(loc)
// 	todayStart := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, loc)

// 	// Query to get the max sequence ID for each instrument and timestamp since today's start
// 	query := `
//         SELECT instrument_token, timestamp, MAX(tick_sequence_id) as max_sequence_id
//         FROM market_data
//         WHERE timestamp >= ?
//         GROUP BY instrument_token, timestamp;
//     `
// 	err = m.dbClient.DB.Raw(query, todayStart).Scan(&results).Error
// 	if err != nil {
// 		zap.L().Error("❌ Failed to load initial tick sequence counters from DB", zap.Error(err))
// 		return
// 	}

//		for _, r := range results {
//			if _, ok := m.tickSequenceCounters[uint(r.InstrumentToken)]; !ok {
//				m.tickSequenceCounters[uint(r.InstrumentToken)] = make(map[time.Time]int)
//			}
//			m.tickSequenceCounters[uint(r.InstrumentToken)][r.Timestamp] = r.MaxSequenceID
//		}
//		zap.L().Info("✅ Loaded initial tick sequence counters from DB", zap.Int("count", len(results)), zap.String("from_timestamp", todayStart.String()))
//	}
package data

import (
	"context"
	"encoding/json"
	"math" // Needed for math.Pow in exponential backoff
	"sync" // Needed for sync.Map and sync.Mutex
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/cache"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"github.com/gorilla/websocket"
	kitemodels "github.com/zerodha/gokiteconnect/v4/models"
	"go.uber.org/zap"
	"gorm.io/gorm/clause"

	redis "github.com/redis/go-redis/v9" // Explicitly import redis client
)

// MarketDataIngestor holds dependencies for market data ingestion and broadcasting.
type MarketDataIngestor struct {
	dbClient             *db.DBClient
	redisClient          *cache.RedisClient
	wsClients            *sync.Map // maps *websocket.Conn to chan []byte (the write channel for that client)
	marketDataBuffer     []db.MarketData
	bufferLock           sync.Mutex
	lastFlushTime        time.Time
	tickSequenceCounters map[uint]map[time.Time]int
	sequenceMux          sync.Mutex
	lastCleanupTime      time.Time
	broadcastChannel     chan []byte // Channel for sending data to WebSocket broadcasters (now dispatchers)
	livePrices           *sync.Map   // Should be a pointer to sync.Map if it's the global instance
	cfg                  *utils.AppConfig

	dbFlushCh              chan []db.MarketData
	dbWorkerCount          int
	wsBroadcastWorkerCount int // Number of goroutines to dispatch WS messages
}

// NewMarketDataIngestor creates and returns a new instance of MarketDataIngestor.
func NewMarketDataIngestor(dbC *db.DBClient, rC *cache.RedisClient, wsClients *sync.Map, cfg *utils.AppConfig, _ *utils.IndicatorsConfig) *MarketDataIngestor {
	ingestor := &MarketDataIngestor{
		dbClient:             dbC,
		redisClient:          rC,
		wsClients:            wsClients,
		marketDataBuffer:     make([]db.MarketData, 0, cfg.Ingestion.MarketDataBatchSize),
		lastFlushTime:        time.Now(),
		tickSequenceCounters: make(map[uint]map[time.Time]int),
		sequenceMux:          sync.Mutex{},
		lastCleanupTime:      time.Now(),
		broadcastChannel:     make(chan []byte, cfg.Ingestion.WSBroadcastChannelSize), // Use configurable buffer size
		livePrices:           &sync.Map{},                                             // Initialize as pointer
		cfg:                  cfg,

		dbFlushCh:              make(chan []db.MarketData, cfg.Ingestion.DBFlushChannelSize), // Use configurable buffer size
		dbWorkerCount:          cfg.Ingestion.DBWorkerCount,
		wsBroadcastWorkerCount: cfg.Ingestion.WSBroadcastWorkerCount,
	}
	ingestor.loadInitialTickSequenceCounters()
	return ingestor
}

// StartIngestionAndBroadcast kicks off the Redis subscription, DB ingestion, and WebSocket broadcasting.
func (m *MarketDataIngestor) StartIngestionAndBroadcast(ctx context.Context) {
	// Start workers that handle database flushing
	m.startDBWorkers(ctx)
	// Start workers that act as dispatchers for WebSocket broadcasting
	m.startWebSocketBroadcasterWorkers(ctx)

	// These remain as primary consumers/dispatchers
	go m.subscribeAndProcessRedis(ctx)
	go m.startDBFlusher(ctx) // This flusher will now send batches to dbFlushCh
	go m.startSequenceCounterCleanup(ctx)

	zap.L().Info("🚀 Market data ingestion and broadcasting started.")
}

// subscribeAndProcessRedis subscribes to the Redis market data channel and unmarshals ticks.
func (m *MarketDataIngestor) subscribeAndProcessRedis(ctx context.Context) {
	var pubsub *redis.PubSub
	// var err error // Removed: 'err' cannot be assigned from m.redisClient.Subscribe if it returns only one value

	initialDelay := time.Duration(m.cfg.Ingestion.RedisReconnectInitialDelayMs) * time.Millisecond
	maxDelay := time.Duration(m.cfg.Ingestion.RedisReconnectMaxDelayMs) * time.Millisecond
	maxAttempts := m.cfg.Ingestion.RedisReconnectMaxAttempts

	// Initial subscription loop with exponential backoff
	for attempt := 0; attempt < maxAttempts; attempt++ {
		select {
		case <-ctx.Done():
			zap.L().Info("Context cancelled, stopping Redis PubSub subscriber before initial subscription.")
			return
		default:
			// Attempt to subscribe. Assuming m.redisClient.Subscribe returns only *redis.PubSub.
			pubsub = m.redisClient.Subscribe(ctx, api.RedisMarketDataChannel) // <-- FIXED: Removed 'err' from assignment

			// Check if pubsub is successfully obtained
			if pubsub != nil { // <-- FIXED: Removed 'err == nil' check
				ch := pubsub.Channel()
				if ch != nil {
					zap.L().Info("✅ Subscribed to Redis market data channel",
						zap.String("channel", api.RedisMarketDataChannel),
						zap.Int("attempt", attempt+1))
					// Now start processing messages from the channel
					m.processRedisMessages(ctx, ch, pubsub) // Pass pubsub for closing
					return                                  // Exit this retry loop and subscribeAndProcessRedis as processing loop is now running
				}
			}

			// If subscription or channel retrieval failed, wait and retry
			logFields := []zap.Field{
				zap.Int("attempt", attempt+1),
				zap.Duration("initial_delay", initialDelay),
				zap.Duration("max_delay", maxDelay),
			}
			// Since m.redisClient.Subscribe doesn't return an error directly,
			// any issue here is likely an internal problem with the RedisClient setup
			// or an underlying network issue. We log a generic reason.
			logFields = append(logFields, zap.String("reason", "pubsub client or channel is nil after subscribe call (possible Redis connection issue or internal RedisClient error)"))

			delay := initialDelay * time.Duration(math.Pow(2, float64(attempt)))
			if delay > maxDelay {
				delay = maxDelay
			}
			logFields = append(logFields, zap.Duration("delay", delay))
			zap.L().Warn("Failed to obtain Redis PubSub client or channel, retrying...", logFields...)
			time.Sleep(delay)
		}

		if attempt == maxAttempts-1 { // If last attempt fails
			zap.L().Fatal("❌ Failed to subscribe to Redis PubSub after multiple attempts, exiting.",
				zap.Int("max_attempts", maxAttempts))
			// Close channels if we are fatally exiting without having started processing
			close(m.dbFlushCh)
			close(m.broadcastChannel)
			return
		}
	}
}

// processRedisMessages consumes messages from the Redis PubSub channel.
func (m *MarketDataIngestor) processRedisMessages(ctx context.Context, ch <-chan *redis.Message, pubsub *redis.PubSub) {
	defer func() {
		if pubsub != nil {
			if err := pubsub.Close(); err != nil {
				zap.L().Error("Failed to close Redis PubSub connection during shutdown", zap.Error(err))
			}
		}
		// Close the main producer channels when this processing goroutine exits.
		// This signals to downstream workers (DB, WS) that no more data is coming.
		close(m.dbFlushCh)
		close(m.broadcastChannel)
		zap.L().Info("Redis PubSub subscriber, DB flush channel, and WS broadcast channel closed.")
	}()

	for {
		select {
		case msg, ok := <-ch:
			if !ok {
				zap.L().Warn("Redis PubSub channel closed, attempting reconnect...")
				// The outer function (subscribeAndProcessRedis) handles reconnection logic.
				// This goroutine should now exit and let the outer loop re-establish.
				return // Exit this message processing loop, triggers defer
			}

			var enrichedTick struct {
				Symbol           string          `json:"symbol"`
				ProcessedAtNanos int64           `json:"processed_at_nanos"`
				Tick             json.RawMessage `json:"tick"` // Keep as RawMessage initially
			}
			if err := json.Unmarshal([]byte(msg.Payload), &enrichedTick); err != nil {
				zap.L().Error("Failed to unmarshal Redis message payload outer structure", zap.Error(err), zap.String("payload_sample", string(msg.Payload[:min(len(msg.Payload), 200)])))
				continue
			}

			var kiteTick kitemodels.Tick
			if err := json.Unmarshal(enrichedTick.Tick, &kiteTick); err != nil {
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
				parsedTime, errParse := time.Parse(time.RFC3339Nano, timestampStr)
				if errParse != nil {
					parsedTime, errParse = time.Parse(time.RFC3339, timestampStr)
					if errParse != nil {
						zap.L().Error("Failed to parse Timestamp from tick payload with RFC3339/RFC3339Nano",
							zap.Error(errParse),
							zap.String("timestamp_string", timestampStr),
							zap.String("payload_sample", string(msg.Payload[:min(len(msg.Payload), 200)])))
						continue
					}
				}
				kiteTick.Timestamp = kitemodels.Time{Time: parsedTime}
			}

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

		case <-ctx.Done():
			zap.L().Info("Context cancelled, stopping Redis PubSub message processor.")
			return // Exit this message processing loop, triggers defer
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// processTick converts the enriched tick to MarketData and adds it to the buffer.
// It is now non-blocking for database writes and WebSocket broadcasts.
func (m *MarketDataIngestor) processTick(enrichedTick struct {
	Symbol           string
	ProcessedAtNanos int64
	Tick             kitemodels.Tick
}) {
	tick := enrichedTick.Tick

	m.sequenceMux.Lock()
	if _, ok := m.tickSequenceCounters[uint(tick.InstrumentToken)]; !ok {
		m.tickSequenceCounters[uint(tick.InstrumentToken)] = make(map[time.Time]int)
	}
	normalizedTimestamp := tick.Timestamp.Time
	currentSequenceID := m.tickSequenceCounters[uint(tick.InstrumentToken)][normalizedTimestamp] + 1
	m.tickSequenceCounters[uint(tick.InstrumentToken)][normalizedTimestamp] = currentSequenceID
	m.sequenceMux.Unlock()

	md := db.MarketData{
		InstrumentToken:    tick.InstrumentToken,
		Timestamp:          normalizedTimestamp,
		TickSequenceID:     currentSequenceID,
		LastPrice:          tick.LastPrice,
		LastTradedQuantity: tick.LastTradedQuantity,
		Volume:             tick.VolumeTraded,
		AverageTradePrice:  tick.AverageTradePrice,
		NetChange:          tick.NetChange,
		Open:               tick.OHLC.Open,
		High:               tick.OHLC.High,
		Low:                tick.OHLC.Low,
		Close:              tick.OHLC.Close,
		OpenInterest:       tick.OI,
		BidPrice1:          tick.Depth.Buy[0].Price,
		BidQuantity1:       tick.Depth.Buy[0].Quantity,
		BidOrders1:         tick.Depth.Buy[0].Orders,
		BidPrice2:          tick.Depth.Buy[1].Price,
		BidQuantity2:       tick.Depth.Buy[1].Quantity,
		BidOrders2:         tick.Depth.Buy[1].Orders,
		BidPrice3:          tick.Depth.Buy[2].Price,
		BidQuantity3:       tick.Depth.Buy[2].Quantity,
		BidOrders3:         tick.Depth.Buy[2].Orders,
		BidPrice4:          tick.Depth.Buy[3].Price,
		BidQuantity4:       tick.Depth.Buy[3].Quantity,
		BidOrders4:         tick.Depth.Buy[3].Orders,
		BidPrice5:          tick.Depth.Buy[4].Price,
		BidQuantity5:       tick.Depth.Buy[4].Quantity,
		BidOrders5:         tick.Depth.Buy[4].Orders,
		AskPrice1:          tick.Depth.Sell[0].Price,
		AskQuantity1:       tick.Depth.Sell[0].Quantity,
		AskOrders1:         tick.Depth.Sell[0].Orders,
		AskPrice2:          tick.Depth.Sell[1].Price,
		AskQuantity2:       tick.Depth.Sell[1].Quantity,
		AskOrders2:         tick.Depth.Sell[1].Orders,
		AskPrice3:          tick.Depth.Sell[2].Price,
		AskQuantity3:       tick.Depth.Sell[2].Quantity,
		AskOrders3:         tick.Depth.Sell[2].Orders,
		AskPrice4:          tick.Depth.Sell[3].Price,
		AskQuantity4:       tick.Depth.Sell[3].Quantity,
		AskOrders4:         tick.Depth.Sell[3].Orders,
		AskPrice5:          tick.Depth.Sell[4].Price,
		AskQuantity5:       tick.Depth.Sell[4].Quantity,
		AskOrders5:         tick.Depth.Sell[4].Orders,
		TotalBuyQuantity:   tick.TotalBuyQuantity,
		TotalSellQuantity:  tick.TotalSellQuantity,
	}

	m.bufferLock.Lock()
	m.marketDataBuffer = append(m.marketDataBuffer, md)
	if len(m.marketDataBuffer) >= m.cfg.Ingestion.MarketDataBatchSize {
		dataToFlush := make([]db.MarketData, len(m.marketDataBuffer))
		copy(dataToFlush, m.marketDataBuffer)
		select {
		case m.dbFlushCh <- dataToFlush:
			m.marketDataBuffer = make([]db.MarketData, 0, m.cfg.Ingestion.MarketDataBatchSize)
		default:
			zap.L().Warn("Dropping DB write batch: DB flush channel is full. Consider increasing buffer size or DB worker count.",
				zap.Int("batch_size", len(dataToFlush)),
				zap.String("instrument", enrichedTick.Symbol))
		}
	}
	m.bufferLock.Unlock()

	frontendData, err := json.Marshal(map[string]interface{}{
		"symbol": enrichedTick.Symbol,
		"tick":   tick,
	})
	if err != nil {
		zap.L().Error("Failed to marshal data for frontend broadcast", zap.Error(err))
	} else {
		select {
		case m.broadcastChannel <- frontendData:
			m.livePrices.Store(enrichedTick.Symbol, frontendData)
		default:
			zap.L().Warn("Dropping WebSocket broadcast message: broadcast channel is full. Consider increasing buffer size or WS worker count.",
				zap.String("symbol", enrichedTick.Symbol))
		}
	}
}

// startDBFlusher periodically checks if the buffer needs flushing based on time.
// It now sends batches to the dbFlushCh.
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
			m.bufferLock.Lock()
			if len(m.marketDataBuffer) > 0 && time.Since(m.lastFlushTime) >= flushInterval {
				dataToFlush := make([]db.MarketData, len(m.marketDataBuffer))
				copy(dataToFlush, m.marketDataBuffer)
				select {
				case m.dbFlushCh <- dataToFlush:
					m.marketDataBuffer = make([]db.MarketData, 0, m.cfg.Ingestion.MarketDataBatchSize)
					m.lastFlushTime = time.Now()
					zap.L().Debug("Timed flush: sent batch to DB worker channel", zap.Int("count", len(dataToFlush)))
				default:
					zap.L().Warn("DB timed flush skipped: DB flush channel is full. Data will be re-attempted next interval.",
						zap.Int("buffered_count", len(dataToFlush)))
				}
			}
			m.bufferLock.Unlock()
		case <-ctx.Done():
			zap.L().Info("Context cancelled, attempting to flush remaining buffer to DB workers before stopping DB flusher.")
			m.bufferLock.Lock()
			if len(m.marketDataBuffer) > 0 {
				dataToFlush := make([]db.MarketData, len(m.marketDataBuffer))
				copy(dataToFlush, m.marketDataBuffer)
				select {
				case m.dbFlushCh <- dataToFlush:
					zap.L().Info("Successfully sent final buffer to DB workers.")
				default:
					zap.L().Error("Failed to send final buffer to DB workers: channel full. Data might be lost.",
						zap.Int("buffered_count", len(dataToFlush)))
				}
			}
			m.bufferLock.Unlock()
			return
		}
	}
}

// startDBWorkers starts a pool of goroutines to consume from dbFlushCh and perform batch inserts.
func (m *MarketDataIngestor) startDBWorkers(ctx context.Context) {
	for i := 0; i < m.dbWorkerCount; i++ {
		go func(workerID int) {
			zap.L().Info("📦 DB worker started", zap.Int("worker_id", workerID))
			for {
				select {
				case dataToFlush, ok := <-m.dbFlushCh:
					if !ok {
						zap.L().Info("📦 DB flush channel closed, worker stopping", zap.Int("worker_id", workerID))
						return
					}
					result := m.dbClient.DB.Clauses(clause.OnConflict{
						Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "timestamp"}, {Name: "tick_sequence_id"}},
						DoNothing: true,
					}).CreateInBatches(dataToFlush, m.cfg.Ingestion.MarketDataBatchSize)

					if result.Error != nil {
						zap.L().Error("❌ DB worker failed to batch insert market data",
							zap.Error(result.Error),
							zap.Int("worker_id", workerID),
							zap.Int("batch_size", len(dataToFlush)))
					} else {
						skippedCount := len(dataToFlush) - int(result.RowsAffected)
						if skippedCount > 0 {
							zap.L().Warn("⚠️ DB worker flushed market data with skipped duplicates",
								zap.Int("worker_id", workerID),
								zap.Int("total_attempted", len(dataToFlush)),
								zap.Int64("rows_inserted", result.RowsAffected),
								zap.Int("rows_skipped", skippedCount))
						} else {
							zap.L().Debug("✅ DB worker successfully flushed market data",
								zap.Int("worker_id", workerID),
								zap.Int64("count", result.RowsAffected))
						}
					}
				case <-ctx.Done():
					zap.L().Info("📦 Context cancelled, DB worker stopping", zap.Int("worker_id", workerID))
					return
				}
			}
		}(i)
	}
}

// startWebSocketBroadcasterWorkers starts a pool of goroutines to consume from broadcastChannel
// and dispatch messages to individual client write channels.
func (m *MarketDataIngestor) startWebSocketBroadcasterWorkers(ctx context.Context) {
	for i := 0; i < m.wsBroadcastWorkerCount; i++ {
		go func(workerID int) {
			zap.L().Info("🌐 WS dispatcher worker started", zap.Int("worker_id", workerID))
			for {
				select {
				case msg, ok := <-m.broadcastChannel:
					if !ok {
						zap.L().Info("🌐 WS dispatch channel closed, worker stopping", zap.Int("worker_id", workerID))
						return
					}
					// Iterate over all connected WebSocket clients and send the message
					m.wsClients.Range(func(key, value interface{}) bool {
						conn, ok := key.(*websocket.Conn)
						if !ok {
							zap.L().Warn("Found non-websocket.Conn in wsClients map, deleting.", zap.Any("key", key))
							m.wsClients.Delete(key) // Clean up bad entry
							return true
						}
						// Retrieve the dedicated write channel for this client
						clientWriteCh, ok := value.(chan []byte)
						if !ok {
							zap.L().Error("Value in wsClients map is not a chan []byte, deleting.", zap.Any("key", key))
							m.wsClients.Delete(key) // Clean up bad entry
							return true
						}

						// Attempt to send the message to the client's dedicated write channel
						select {
						case clientWriteCh <- msg:
							// Message successfully queued for this client
						default:
							// If client's individual channel is full, drop message for this client
							zap.L().Warn("Dropping WebSocket message for client: client's write channel is full.",
								zap.String("remote_addr", conn.RemoteAddr().String()),
								zap.Int("worker_id", workerID))
						}
						return true // Continue iterating other clients
					})
				case <-ctx.Done():
					zap.L().Info("🌐 Context cancelled, WS dispatcher worker stopping", zap.Int("worker_id", workerID))
					return
				}
			}
		}(i)
	}
}

// RegisterWebSocketClient adds a new WebSocket client and starts its dedicated write pump.
func (m *MarketDataIngestor) RegisterWebSocketClient(conn *websocket.Conn) {
	// Create a buffered channel for this specific client's writes
	clientWriteCh := make(chan []byte, 16) // Small buffer per client to avoid blocking the dispatcher

	// Store the connection and its write channel
	m.wsClients.Store(conn, clientWriteCh)
	zap.L().Info("🧑‍💻 New WebSocket client connected", zap.String("remote_addr", conn.RemoteAddr().String()))

	// Start a dedicated goroutine (write pump) for this client
	go m.writePump(conn, clientWriteCh)

	// Send initial live prices to the newly connected client's dedicated channel
	m.livePrices.Range(func(key, value interface{}) bool {
		data := value.([]byte)
		select {
		case clientWriteCh <- data:
			// Message successfully queued
		default:
			zap.L().Warn("Failed to send initial live price to new WS client (channel full during init)",
				zap.String("symbol", key.(string)),
				zap.String("remote_addr", conn.RemoteAddr().String()))
			// If initial send fails, it's problematic. Close connection and remove.
			conn.Close()
			m.wsClients.Delete(conn)
			return false // Stop iterating and mark this registration as failed
		}
		return true // Continue iterating
	})
}

// UnregisterWebSocketClient removes a WebSocket client and signals its write pump to stop.
func (m *MarketDataIngestor) UnregisterWebSocketClient(conn *websocket.Conn) {
	if clientWriteCh, ok := m.wsClients.Load(conn); ok {
		// Attempt to close the channel safely, only if it hasn't been closed by writePump already
		// This prevents panics if `UnregisterWebSocketClient` is called after `writePump` has already exited
		// and closed the channel.
		select {
		case <-clientWriteCh.(chan []byte): // Try to read to see if it's already closed
			// Channel is already closed or has data, proceed to close
			zap.L().Debug("Client write channel already closed or being read from, proceeding with unregister.", zap.String("remote_addr", conn.RemoteAddr().String()))
		default:
			// If not closed, close it now
			close(clientWriteCh.(chan []byte)) // Signal the writePump to exit
		}

		m.wsClients.Delete(conn)
		conn.Close() // Ensure connection is closed
		zap.L().Info("🔌 WebSocket client disconnected", zap.String("remote_addr", conn.RemoteAddr().String()))
	} else {
		zap.L().Warn("Attempted to unregister a WebSocket client that was not found.", zap.String("remote_addr", conn.RemoteAddr().String()))
	}
}

// writePump reads messages from a client's dedicated channel and writes them to the WebSocket connection.
// This ensures that only one goroutine ever writes to a specific websocket.Conn.
func (m *MarketDataIngestor) writePump(conn *websocket.Conn, clientWriteCh <-chan []byte) {
	defer func() {
		// Ensure cleanup if writePump exits prematurely
		// It's safe to call UnregisterWebSocketClient, as it handles idempotent deletion and closing.
		m.UnregisterWebSocketClient(conn)
		zap.L().Info("🚽 WebSocket write pump stopped for client", zap.String("remote_addr", conn.RemoteAddr().String()))
	}()

	for {
		select {
		case message, ok := <-clientWriteCh:
			if !ok { // Channel closed, signaling shutdown
				zap.L().Info("WebSocket client write channel closed, write pump exiting gracefully", zap.String("remote_addr", conn.RemoteAddr().String()))
				return
			}

			// Set a write deadline to prevent writes from blocking indefinitely
			// This can help detect broken connections faster
			// conn.SetWriteDeadline(time.Now().Add(m.cfg.Server.WriteDeadline)) // Assuming you add WriteDeadline to config
			// For now, let's keep it simple without a specific deadline from config

			err := conn.WriteMessage(websocket.TextMessage, message)
			if err != nil {
				// Log the error and allow the deferred UnregisterWebSocketClient to handle cleanup
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					zap.L().Error("WebSocket write error (unexpected close) in write pump", zap.Error(err), zap.String("remote_addr", conn.RemoteAddr().String()))
				} else {
					zap.L().Info("WebSocket write error (graceful close/other) in write pump", zap.Error(err), zap.String("remote_addr", conn.RemoteAddr().String()))
				}
				return // Exit the write pump, trigger defer
			}
			// If using a context for the write pump, you'd add a case for ctx.Done() here.
			// For simplicity, relying on channel closure for graceful shutdown.
		}
	}
}

// startSequenceCounterCleanup periodically cleans up old entries in tickSequenceCounters to prevent memory leaks.
func (m *MarketDataIngestor) startSequenceCounterCleanup(ctx context.Context) {
	// Use config value for cleanup interval if available, otherwise default
	cleanupInterval := time.Duration(m.cfg.Ingestion.TickSequenceCleanupInterval) * time.Second
	if cleanupInterval <= 0 {
		cleanupInterval = 10 * time.Minute // Fallback if config is invalid
		zap.L().Warn("Invalid TickSequenceCleanupInterval in config, defaulting to 10 minutes", zap.Int("configured_value", m.cfg.Ingestion.TickSequenceCleanupInterval))
	}

	expiryDuration := time.Duration(m.cfg.Ingestion.MaxTickSequenceCacheDuration) * time.Second
	if expiryDuration <= 0 {
		expiryDuration = 24 * time.Hour // Fallback if config is invalid
		zap.L().Warn("Invalid MaxTickSequenceCacheDuration in config, defaulting to 24 hours", zap.Int("configured_value", m.cfg.Ingestion.MaxTickSequenceCacheDuration))
	}

	ticker := time.NewTicker(cleanupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			m.sequenceMux.Lock()
			now := time.Now()
			for instrument, tsMap := range m.tickSequenceCounters {
				for ts := range tsMap {
					if now.Sub(ts) > expiryDuration {
						delete(tsMap, ts)
					}
				}
				if len(tsMap) == 0 {
					delete(m.tickSequenceCounters, instrument)
				}
			}
			m.sequenceMux.Unlock()
			m.lastCleanupTime = time.Now()
			zap.L().Debug("Cleaned up old tick sequence counters",
				zap.Time("cleanup_time", m.lastCleanupTime),
				zap.Duration("removed_older_than", expiryDuration),
			)
		case <-ctx.Done():
			zap.L().Info("Context cancelled, stopping sequence counter cleanup goroutine.")
			return
		}
	}
}

// loadInitialTickSequenceCounters loads the max tick sequence IDs from DB on startup.
func (m *MarketDataIngestor) loadInitialTickSequenceCounters() {
	m.sequenceMux.Lock()
	defer m.sequenceMux.Unlock()

	type Result struct {
		InstrumentToken uint32    `gorm:"column:instrument_token"`
		Timestamp       time.Time `gorm:"column:timestamp"`
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
		m.tickSequenceCounters[uint(r.InstrumentToken)][r.Timestamp] = r.MaxSequenceID
	}
	zap.L().Info("✅ Loaded initial tick sequence counters from DB", zap.Int("count", len(results)), zap.String("from_timestamp", todayStart.String()))
}
