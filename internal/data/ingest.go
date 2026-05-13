package data

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/cache"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"github.com/gorilla/websocket"
	kitemodels "github.com/zerodha/gokiteconnect/v4/models"
	"go.uber.org/zap"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"

	redis "github.com/redis/go-redis/v9"
)

const (
	wsPingPeriod = 45 * time.Second
	wsWriteWait  = 10 * time.Second
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
	wsBroadcastWorkerCount int

	// Monitoring metrics
	dbErrors      uint64
	wsDrops       uint64
	wsBroadcasted uint64
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
		broadcastChannel:     make(chan []byte, cfg.Ingestion.WSBroadcastChannelSize),
		livePrices:           &sync.Map{},
		cfg:                  cfg,

		dbFlushCh:              make(chan []db.MarketData, cfg.Ingestion.DBFlushChannelSize),
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

	// Start monitoring goroutine for metrics
	go m.startMonitoring(ctx)

	// These remain as primary consumers/dispatchers
	go m.subscribeAndProcessRedis(ctx)
	go m.startDBFlusher(ctx)
	go m.startSequenceCounterCleanup(ctx)

	zap.L().Info("🚀 Market data ingestion and broadcasting started.")
}

// Monitoring goroutine for DB errors, WS drops, and broadcast rate.
func (m *MarketDataIngestor) startMonitoring(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	var lastBroadcasted uint64
	for {
		select {
		case <-ticker.C:
			dbErrs := atomic.LoadUint64(&m.dbErrors)
			wsDrops := atomic.LoadUint64(&m.wsDrops)
			broadcasted := atomic.LoadUint64(&m.wsBroadcasted)
			rate := broadcasted - lastBroadcasted
			lastBroadcasted = broadcasted

			zap.L().Info("MarketDataIngestor monitoring",
				zap.Uint64("db_errors", dbErrs),
				zap.Uint64("ws_drops", wsDrops),
				zap.Uint64("broadcast_rate_per_5s", rate),
			)
			if dbErrs > 10 {
				zap.L().Warn("High DB error count", zap.Uint64("db_errors", dbErrs))
			}
			if wsDrops > 10 {
				zap.L().Warn("High WebSocket drop count", zap.Uint64("ws_drops", wsDrops))
			}
			if rate < 10 {
				zap.L().Warn("Low broadcast rate", zap.Uint64("broadcast_rate_per_5s", rate))
			}
		case <-ctx.Done():
			return
		}
	}
}

// subscribeAndProcessRedis subscribes to the Redis market data channel and unmarshals ticks.
func (m *MarketDataIngestor) subscribeAndProcessRedis(ctx context.Context) {
	var pubsub *redis.PubSub

	initialDelay := time.Duration(m.cfg.Ingestion.RedisReconnectInitialDelayMs) * time.Millisecond
	maxDelay := time.Duration(m.cfg.Ingestion.RedisReconnectMaxDelayMs) * time.Millisecond
	maxAttempts := m.cfg.Ingestion.RedisReconnectMaxAttempts

	for attempt := 0; attempt < maxAttempts; attempt++ {
		select {
		case <-ctx.Done():
			zap.L().Info("Context cancelled, stopping Redis PubSub subscriber before initial subscription.")
			return
		default:
			pubsub = m.redisClient.Subscribe(ctx, api.RedisMarketDataChannel)
			if pubsub != nil {
				ch := pubsub.Channel()
				if ch != nil {
					zap.L().Info("✅ Subscribed to Redis market data channel",
						zap.String("channel", api.RedisMarketDataChannel),
						zap.Int("attempt", attempt+1))
					m.processRedisMessages(ctx, ch, pubsub)
					return
				}
			}
			logFields := []zap.Field{
				zap.Int("attempt", attempt+1),
				zap.Duration("initial_delay", initialDelay),
				zap.Duration("max_delay", maxDelay),
			}
			logFields = append(logFields, zap.String("reason", "pubsub client or channel is nil after subscribe call (possible Redis connection issue or internal RedisClient error)"))
			delay := initialDelay * time.Duration(math.Pow(2, float64(attempt)))
			if delay > maxDelay {
				delay = maxDelay
			}
			logFields = append(logFields, zap.Duration("delay", delay))
			zap.L().Warn("Failed to obtain Redis PubSub client or channel, retrying...", logFields...)
			time.Sleep(delay)
		}

		if attempt == maxAttempts-1 {
			zap.L().Error("Failed to subscribe to Redis PubSub after max attempts, stopping ingestor.",
				zap.Int("max_attempts", maxAttempts))
			return
		}
	}
}

// processRedisMessages consumes messages from the Redis PubSub channel.
func (m *MarketDataIngestor) processRedisMessages(ctx context.Context, ch <-chan *redis.Message, pubsub *redis.PubSub) {
	defer func() {
		if pubsub != nil {
			if err := pubsub.Close(); err != nil {
				// "use of closed network connection" is expected during shutdown — suppress it.
				if !isClosedNetworkError(err) {
					zap.L().Error("Failed to close Redis PubSub connection during shutdown", zap.Error(err))
				}
			}
		}
		close(m.dbFlushCh)
		close(m.broadcastChannel)
		zap.L().Info("Redis PubSub subscriber, DB flush channel, and WS broadcast channel closed.")
	}()

	for {
		select {
		case msg, ok := <-ch:
			if !ok {
				zap.L().Warn("Redis PubSub channel closed, attempting reconnect...")
				return
			}

			var enrichedTick struct {
				Symbol           string          `json:"symbol"`
				ProcessedAtNanos int64           `json:"processed_at_nanos"`
				Tick             json.RawMessage `json:"tick"`
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
			return
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// isClosedNetworkError returns true for the benign "use of closed network connection"
// error that occurs when a TCP connection is closed before we call Close() on it.
func isClosedNetworkError(err error) bool {
	return err != nil && strings.Contains(err.Error(), "use of closed network connection")
}

// processTick converts the enriched tick to MarketData and adds it to the buffer.
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
	// Pad depth to 5 levels; tick.Depth.Buy/Sell are fixed [5]DepthItem arrays.
	buyDepth := tick.Depth.Buy
	sellDepth := tick.Depth.Sell
	zap.L().Debug("Tick received",
		zap.String("symbol", enrichedTick.Symbol),
		zap.Float64("ltp", tick.LastPrice),
		zap.Float64("bid", buyDepth[0].Price),
		zap.Float64("ask", sellDepth[0].Price),
		zap.Int("volume", int(tick.VolumeTraded)),
		zap.Float64("prev_close", tick.OHLC.Close),
	)
	GetMarketHeatmap().Update(
		enrichedTick.Symbol,
		tick.LastPrice,
		buyDepth[0].Price,
		sellDepth[0].Price,
		int64(buyDepth[0].Quantity),
		int64(sellDepth[0].Quantity),
		int64(tick.VolumeTraded),
		tick.LastPrice,
		tick.OHLC.Close,
	)
	zap.L().Debug("Heatmap updated", zap.String("symbol", enrichedTick.Symbol))
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
		BidPrice1:          buyDepth[0].Price,
		BidQuantity1:       buyDepth[0].Quantity,
		BidOrders1:         buyDepth[0].Orders,
		BidPrice2:          buyDepth[1].Price,
		BidQuantity2:       buyDepth[1].Quantity,
		BidOrders2:         buyDepth[1].Orders,
		BidPrice3:          buyDepth[2].Price,
		BidQuantity3:       buyDepth[2].Quantity,
		BidOrders3:         buyDepth[2].Orders,
		BidPrice4:          buyDepth[3].Price,
		BidQuantity4:       buyDepth[3].Quantity,
		BidOrders4:         buyDepth[3].Orders,
		BidPrice5:          buyDepth[4].Price,
		BidQuantity5:       buyDepth[4].Quantity,
		BidOrders5:         buyDepth[4].Orders,
		AskPrice1:          sellDepth[0].Price,
		AskQuantity1:       sellDepth[0].Quantity,
		AskOrders1:         sellDepth[0].Orders,
		AskPrice2:          sellDepth[1].Price,
		AskQuantity2:       sellDepth[1].Quantity,
		AskOrders2:         sellDepth[1].Orders,
		AskPrice3:          sellDepth[2].Price,
		AskQuantity3:       sellDepth[2].Quantity,
		AskOrders3:         sellDepth[2].Orders,
		AskPrice4:          sellDepth[3].Price,
		AskQuantity4:       sellDepth[3].Quantity,
		AskOrders4:         sellDepth[3].Orders,
		AskPrice5:          sellDepth[4].Price,
		AskQuantity5:       sellDepth[4].Quantity,
		AskOrders5:         sellDepth[4].Orders,
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
			atomic.AddUint64(&m.dbErrors, 1)
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
			atomic.AddUint64(&m.wsBroadcasted, 1)
			m.livePrices.Store(enrichedTick.Symbol, frontendData)
		default:
			atomic.AddUint64(&m.wsDrops, 1)
			zap.L().Warn("Dropping WebSocket broadcast message: broadcast channel is full. Consider increasing buffer size or WS worker count.",
				zap.String("symbol", enrichedTick.Symbol))
		}
	}
}

// startDBFlusher periodically checks if the buffer needs flushing based on time.
func (m *MarketDataIngestor) startDBFlusher(ctx context.Context) {
	flushInterval := time.Duration(m.cfg.Ingestion.MarketDataFlushIntervalMS) * time.Millisecond
	if flushInterval <= 0 {
		zap.L().Error("MarketDataFlushIntervalMS must be positive in app.yaml, defaulting to 500ms")
		flushInterval = 500 * time.Millisecond
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
					atomic.AddUint64(&m.dbErrors, 1)
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
					atomic.AddUint64(&m.dbErrors, 1)
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
			defer func() {
				if r := recover(); r != nil {
					zap.L().Error("Panic recovered in DB worker", zap.Int("worker_id", workerID), zap.Any("recover", r))
				}
			}()
			zap.L().Info("📦 DB worker started", zap.Int("worker_id", workerID))
			for {
				select {
				case dataToFlush, ok := <-m.dbFlushCh:
					if !ok {
						zap.L().Info("📦 DB flush channel closed, worker stopping", zap.Int("worker_id", workerID))
						return
					}
					start := time.Now()

					seen := make(map[string]int)
					for _, row := range dataToFlush {
						key := fmt.Sprintf("%d_%s_%d", row.InstrumentToken, row.Timestamp.Format(time.RFC3339Nano), row.TickSequenceID)
						seen[key]++
					}
					for k, v := range seen {
						if v > 1 {
							zap.L().Warn("Duplicate in batch before DB insert", zap.String("key", k), zap.Int("count", v))
						}
					}

					maxRetries := 3
					var result *gorm.DB
					var err error
					for attempt := 1; attempt <= maxRetries; attempt++ {
						result = m.dbClient.DB.Clauses(clause.OnConflict{
							Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "timestamp"}, {Name: "tick_sequence_id"}},
							DoNothing: true,
						}).CreateInBatches(dataToFlush, m.cfg.Ingestion.MarketDataBatchSize)
						err = result.Error
						if err != nil && err.Error() != "" &&
							(strings.Contains(err.Error(), "deadlock detected") || strings.Contains(err.Error(), "SQLSTATE 40P01")) {
							zap.L().Warn("DB deadlock detected, retrying batch insert",
								zap.Int("worker_id", workerID),
								zap.Int("attempt", attempt),
								zap.Error(err),
							)
							time.Sleep(100 * time.Millisecond)
							continue
						}
						break
					}
					duration := time.Since(start)
					if err != nil {
						atomic.AddUint64(&m.dbErrors, 1)
						zap.L().Error("❌ DB worker failed to batch insert market data",
							zap.Error(err),
							zap.Int("worker_id", workerID),
							zap.Int("batch_size", len(dataToFlush)),
							zap.Duration("duration", duration))
						if duration > time.Second {
							zap.L().Warn("DB write took too long", zap.Duration("duration", duration))
						}
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
			defer func() {
				if r := recover(); r != nil {
					zap.L().Error("Panic recovered in WS dispatcher worker", zap.Int("worker_id", workerID), zap.Any("recover", r))
				}
			}()
			zap.L().Info("🌐 WS dispatcher worker started", zap.Int("worker_id", workerID))
			for {
				select {
				case msg, ok := <-m.broadcastChannel:
					if !ok {
						zap.L().Info("🌐 WS dispatch channel closed, worker stopping", zap.Int("worker_id", workerID))
						return
					}
					m.wsClients.Range(func(key, value interface{}) bool {
						conn, ok := key.(*websocket.Conn)
						if !ok {
							zap.L().Warn("Found non-websocket.Conn in wsClients map, deleting.", zap.Any("key", key))
							m.wsClients.Delete(key)
							return true
						}
						clientWriteCh, ok := value.(chan []byte)
						if !ok {
							zap.L().Error("Value in wsClients map is not a chan []byte, deleting.", zap.Any("key", key))
							m.wsClients.Delete(key)
							return true
						}
						select {
						case clientWriteCh <- msg:
							// Message successfully queued for this client
						default:
							atomic.AddUint64(&m.wsDrops, 1)
							zap.L().Warn("Dropping WebSocket message for client: client's write channel is full.",
								zap.String("remote_addr", conn.RemoteAddr().String()),
								zap.Int("worker_id", workerID))
						}
						return true
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
	clientWriteCh := make(chan []byte, 1024)
	m.wsClients.Store(conn, clientWriteCh)
	zap.L().Info("🧑‍💻 New WebSocket client connected", zap.String("remote_addr", conn.RemoteAddr().String()))
	go m.writePump(conn, clientWriteCh)
	m.livePrices.Range(func(key, value interface{}) bool {
		data := value.([]byte)
		select {
		case clientWriteCh <- data:
		default:
			zap.L().Warn("Failed to send initial live price to new WS client (channel full during init)",
				zap.String("symbol", key.(string)),
				zap.String("remote_addr", conn.RemoteAddr().String()))
			conn.Close()
			m.wsClients.Delete(conn)
			return false
		}
		return true
	})
}

// UnregisterWebSocketClient removes a WebSocket client and signals its write pump to stop.
func (m *MarketDataIngestor) UnregisterWebSocketClient(conn *websocket.Conn) {
	if clientWriteCh, ok := m.wsClients.LoadAndDelete(conn); ok {
		close(clientWriteCh.(chan []byte))
		conn.Close()
		zap.L().Info("🔌 WebSocket client disconnected", zap.String("remote_addr", conn.RemoteAddr().String()))
	}
}

// writePump reads messages from a client's dedicated channel and writes them to the WebSocket connection.
// It also sends periodic pings to detect stale connections.
func (m *MarketDataIngestor) writePump(conn *websocket.Conn, clientWriteCh <-chan []byte) {
	defer zap.L().Info("WebSocket write pump stopped", zap.String("remote_addr", conn.RemoteAddr().String()))
	pingTicker := time.NewTicker(wsPingPeriod)
	defer pingTicker.Stop()
	for {
		select {
		case message, ok := <-clientWriteCh:
			if !ok {
				_ = conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
				_ = conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
				return
			}
			_ = conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
			if err := conn.WriteMessage(websocket.TextMessage, message); err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					zap.L().Error("WebSocket write error", zap.Error(err), zap.String("remote_addr", conn.RemoteAddr().String()))
				}
				return
			}
		case <-pingTicker.C:
			_ = conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
			if err := conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// startSequenceCounterCleanup periodically cleans up old entries in tickSequenceCounters to prevent memory leaks.
func (m *MarketDataIngestor) startSequenceCounterCleanup(ctx context.Context) {
	cleanupInterval := time.Duration(m.cfg.Ingestion.TickSequenceCleanupInterval) * time.Second
	if cleanupInterval <= 0 {
		cleanupInterval = 10 * time.Minute
		zap.L().Warn("Invalid TickSequenceCleanupInterval in config, defaulting to 10 minutes", zap.Int("configured_value", m.cfg.Ingestion.TickSequenceCleanupInterval))
	}

	expiryDuration := time.Duration(m.cfg.Ingestion.MaxTickSequenceCacheDuration) * time.Second
	if expiryDuration <= 0 {
		expiryDuration = 24 * time.Hour
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
