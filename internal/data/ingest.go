package data

import (
	"context"
	"encoding/json"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/cache"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/marketdata"
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
	wsClients            *sync.Map
	marketDataBuffer     []db.MarketData
	bufferLock           sync.Mutex
	lastFlushTime        time.Time
	tickSequenceCounters map[uint]map[time.Time]int
	sequenceMux          sync.Mutex
	lastCleanupTime      time.Time
	broadcastChannel     chan []byte
	livePrices           *sync.Map
	cfg                  *utils.AppConfig

	dbFlushCh              chan []db.MarketData
	dbWorkerCount          int
	wsBroadcastWorkerCount int
	tickIngestionTimeout   time.Duration // used only for WebSocket broadcast
	dbFlushTimeout         time.Duration // used for DB flush blocking sends

	// Monitoring metrics
	dbErrors      uint64 // DB write errors (not drops)
	wsDrops       uint64 // frontend WebSocket drops
	wsBroadcasted uint64
	droppedTicks  uint64 // only counts frontend drops (not DB)
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
		tickIngestionTimeout:   time.Duration(cfg.Ingestion.TickIngestionTimeoutMs) * time.Millisecond,
	}
	dbFlushTimeout := time.Duration(cfg.Ingestion.DBFlushTimeoutMs) * time.Millisecond
	if dbFlushTimeout <= 0 {
		dbFlushTimeout = 5 * time.Second
	}
	ingestor.dbFlushTimeout = dbFlushTimeout
	ingestor.loadInitialTickSequenceCounters()
	return ingestor
}

// StartIngestionAndBroadcast kicks off the Redis subscription, DB ingestion, and WebSocket broadcasting.
func (m *MarketDataIngestor) StartIngestionAndBroadcast(ctx context.Context) {
	m.startDBWorkers(ctx)
	m.startWebSocketBroadcasterWorkers(ctx)
	go m.startMonitoring(ctx)
	go m.subscribeAndProcessRedis(ctx)
	go m.startDBFlusher(ctx)
	go m.startSequenceCounterCleanup(ctx)
	zap.L().Info("🚀 Market data ingestion and broadcasting started.")
}

// startMonitoring logs metrics every 5s.
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
			dropTicks := atomic.LoadUint64(&m.droppedTicks)
			rate := broadcasted - lastBroadcasted
			lastBroadcasted = broadcasted

			zap.L().Info("MarketDataIngestor monitoring",
				zap.Uint64("db_errors", dbErrs),
				zap.Uint64("ws_drops", wsDrops),
				zap.Uint64("broadcast_rate_per_5s", rate),
				zap.Uint64("frontend_dropped_ticks", dropTicks),
			)
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
			delay := initialDelay * time.Duration(math.Pow(2, float64(attempt)))
			if delay > maxDelay {
				delay = maxDelay
			}
			zap.L().Warn("Failed to obtain Redis PubSub client or channel, retrying...",
				zap.Int("attempt", attempt+1), zap.Duration("delay", delay))
			time.Sleep(delay)
		}
	}
	zap.L().Error("Failed to subscribe to Redis PubSub after max attempts, stopping ingestor.",
		zap.Int("max_attempts", maxAttempts))
}

// convertNormalizedToKiteTick converts a NormalizedTick to a kitemodels.Tick for downstream compatibility.
func convertNormalizedToKiteTick(nt marketdata.NormalizedTick) kitemodels.Tick {
	return kitemodels.Tick{
		InstrumentToken:    nt.InstrumentToken,
		Timestamp:          kitemodels.Time{Time: nt.EventTime},
		LastPrice:          nt.LastPrice,
		LastTradedQuantity: nt.LastTradedQuantity,
		VolumeTraded:       nt.Volume,
		AverageTradePrice:  nt.AverageTradePrice,
		NetChange:          nt.NetChange,
		OHLC:               nt.OHLC,
		Depth:              nt.Depth,
		TotalBuyQuantity:   nt.TotalBuyQuantity,
		TotalSellQuantity:  nt.TotalSellQuantity,
		OI:                 nt.OpenInterest,
	}
}

// processRedisMessages consumes messages from the Redis PubSub channel.
func (m *MarketDataIngestor) processRedisMessages(ctx context.Context, ch <-chan *redis.Message, pubsub *redis.PubSub) {
	defer func() {
		if pubsub != nil {
			if err := pubsub.Close(); err != nil && !isClosedNetworkError(err) {
				zap.L().Error("Failed to close Redis PubSub connection", zap.Error(err))
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
				Symbol           string                    `json:"symbol"`
				ProcessedAtNanos int64                     `json:"processed_at_nanos"`
				Tick             marketdata.NormalizedTick `json:"tick"`
			}
			if err := json.Unmarshal([]byte(msg.Payload), &enrichedTick); err != nil {
				zap.L().Error("Failed to unmarshal Redis message payload", zap.Error(err))
				continue
			}

			// Convert to old tick format for downstream processing (candles, indicators, etc.)
			kiteTick := convertNormalizedToKiteTick(enrichedTick.Tick)
			m.processTick(struct {
				Symbol           string
				ProcessedAtNanos int64
				Tick             kitemodels.Tick
			}{
				Symbol:           enrichedTick.Symbol,
				ProcessedAtNanos: enrichedTick.ProcessedAtNanos,
				Tick:             kiteTick,
			})
		case <-ctx.Done():
			zap.L().Info("Context cancelled, stopping Redis PubSub message processor.")
			return
		}
	}
}

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

	// Sequence counter
	m.sequenceMux.Lock()
	if _, ok := m.tickSequenceCounters[uint(tick.InstrumentToken)]; !ok {
		m.tickSequenceCounters[uint(tick.InstrumentToken)] = make(map[time.Time]int)
	}
	normalizedTimestamp := tick.Timestamp.Time
	currentSequenceID := m.tickSequenceCounters[uint(tick.InstrumentToken)][normalizedTimestamp] + 1
	m.tickSequenceCounters[uint(tick.InstrumentToken)][normalizedTimestamp] = currentSequenceID
	m.sequenceMux.Unlock()

	buyDepth := tick.Depth.Buy
	sellDepth := tick.Depth.Sell
	zap.L().Debug("Tick received",
		zap.String("symbol", enrichedTick.Symbol),
		zap.Float64("ltp", tick.LastPrice),
		zap.Float64("bid", buyDepth[0].Price),
		zap.Float64("ask", sellDepth[0].Price),
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
		BidPrice1:          buyDepth[0].Price, BidQuantity1: buyDepth[0].Quantity, BidOrders1: buyDepth[0].Orders,
		BidPrice2: buyDepth[1].Price, BidQuantity2: buyDepth[1].Quantity, BidOrders2: buyDepth[1].Orders,
		BidPrice3: buyDepth[2].Price, BidQuantity3: buyDepth[2].Quantity, BidOrders3: buyDepth[2].Orders,
		BidPrice4: buyDepth[3].Price, BidQuantity4: buyDepth[3].Quantity, BidOrders4: buyDepth[3].Orders,
		BidPrice5: buyDepth[4].Price, BidQuantity5: buyDepth[4].Quantity, BidOrders5: buyDepth[4].Orders,
		AskPrice1: sellDepth[0].Price, AskQuantity1: sellDepth[0].Quantity, AskOrders1: sellDepth[0].Orders,
		AskPrice2: sellDepth[1].Price, AskQuantity2: sellDepth[1].Quantity, AskOrders2: sellDepth[1].Orders,
		AskPrice3: sellDepth[2].Price, AskQuantity3: sellDepth[2].Quantity, AskOrders3: sellDepth[2].Orders,
		AskPrice4: sellDepth[3].Price, AskQuantity4: sellDepth[3].Quantity, AskOrders4: sellDepth[3].Orders,
		AskPrice5: sellDepth[4].Price, AskQuantity5: sellDepth[4].Quantity, AskOrders5: sellDepth[4].Orders,
		TotalBuyQuantity:  tick.TotalBuyQuantity,
		TotalSellQuantity: tick.TotalSellQuantity,
	}

	m.bufferLock.Lock()
	m.marketDataBuffer = append(m.marketDataBuffer, md)
	batchSize := m.cfg.Ingestion.MarketDataBatchSize
	flushNow := len(m.marketDataBuffer) >= batchSize
	var dataToFlush []db.MarketData
	if flushNow {
		dataToFlush = make([]db.MarketData, len(m.marketDataBuffer))
		copy(dataToFlush, m.marketDataBuffer)
	}

	// ---- DB PATH: BLOCKING WITH TIMEOUT (FATAL ON TIMEOUT) ----
	if flushNow {
		select {
		case m.dbFlushCh <- dataToFlush:
			// success
		case <-time.After(m.dbFlushTimeout):
			atomic.AddUint64(&m.dbErrors, 1)
			zap.L().Fatal("DB flush timeout – cannot persist market data. Exiting.",
				zap.Duration("timeout", m.dbFlushTimeout),
				zap.Int("batch_size", len(dataToFlush)))
		}
		m.marketDataBuffer = make([]db.MarketData, 0, batchSize)
		m.lastFlushTime = time.Now()
	}
	m.bufferLock.Unlock()

	// ---- FRONTEND BROADCAST: TIMEOUT (drops allowed) ----
	// Keep the original broadcast format (kitemodels.Tick) to avoid breaking the frontend.
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
		case <-time.After(m.tickIngestionTimeout):
			atomic.AddUint64(&m.droppedTicks, 1)
			zap.L().Warn("Frontend broadcast timeout, dropping tick (acceptable)",
				zap.Duration("timeout", m.tickIngestionTimeout),
				zap.String("symbol", enrichedTick.Symbol))
		}
	}
}

// startDBFlusher periodically checks if the buffer needs flushing based on time.
func (m *MarketDataIngestor) startDBFlusher(ctx context.Context) {
	flushInterval := time.Duration(m.cfg.Ingestion.MarketDataFlushIntervalMS) * time.Millisecond
	if flushInterval <= 0 {
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

				// ---- DB FLUSH WITH TIMEOUT (FATAL ON TIMEOUT) ----
				select {
				case m.dbFlushCh <- dataToFlush:
					// success
				case <-time.After(m.dbFlushTimeout):
					atomic.AddUint64(&m.dbErrors, 1)
					zap.L().Fatal("Timed DB flush timeout – cannot persist market data. Exiting.",
						zap.Duration("timeout", m.dbFlushTimeout),
						zap.Int("batch_size", len(dataToFlush)))
				}

				m.marketDataBuffer = make([]db.MarketData, 0, m.cfg.Ingestion.MarketDataBatchSize)
				m.lastFlushTime = time.Now()
				zap.L().Debug("Timed flush: sent batch to DB worker channel", zap.Int("count", len(dataToFlush)))
			}
			m.bufferLock.Unlock()

		case <-ctx.Done():
			// Final flush on shutdown – use a short timeout to avoid hanging
			m.bufferLock.Lock()
			if len(m.marketDataBuffer) > 0 {
				dataToFlush := make([]db.MarketData, len(m.marketDataBuffer))
				copy(dataToFlush, m.marketDataBuffer)

				shutdownTimeout := 1 * time.Second
				select {
				case m.dbFlushCh <- dataToFlush:
					zap.L().Info("Sent final buffer to DB workers on shutdown.")
				case <-time.After(shutdownTimeout):
					atomic.AddUint64(&m.dbErrors, 1)
					zap.L().Error("Final DB flush timeout on shutdown – some ticks may be lost.",
						zap.Duration("timeout", shutdownTimeout),
						zap.Int("batch_size", len(dataToFlush)))
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
					maxRetries := 3
					var result *gorm.DB
					var err error
					for attempt := 1; attempt <= maxRetries; attempt++ {
						result = m.dbClient.DB.Clauses(clause.OnConflict{
							Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "timestamp"}, {Name: "tick_sequence_id"}},
							DoNothing: true,
						}).CreateInBatches(dataToFlush, m.cfg.Ingestion.MarketDataBatchSize)
						err = result.Error
						if err != nil && strings.Contains(err.Error(), "deadlock") {
							zap.L().Warn("DB deadlock, retrying", zap.Int("attempt", attempt), zap.Error(err))
							time.Sleep(100 * time.Millisecond)
							continue
						}
						break
					}
					duration := time.Since(start)
					if err != nil {
						atomic.AddUint64(&m.dbErrors, 1)
						zap.L().Error("❌ DB worker failed to batch insert market data",
							zap.Error(err), zap.Int("batch_size", len(dataToFlush)), zap.Duration("duration", duration))
					} else {
						zap.L().Debug("✅ DB worker inserted batch", zap.Int64("rows", result.RowsAffected), zap.Duration("duration", duration))
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
							m.wsClients.Delete(key)
							return true
						}
						clientWriteCh, ok := value.(chan []byte)
						if !ok {
							m.wsClients.Delete(key)
							return true
						}
						select {
						case clientWriteCh <- msg:
						default:
							atomic.AddUint64(&m.wsDrops, 1)
							zap.L().Warn("Dropping WebSocket message for client: client's write channel is full",
								zap.String("remote_addr", conn.RemoteAddr().String()))
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
				zap.String("symbol", key.(string)))
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
					zap.L().Error("WebSocket write error", zap.Error(err))
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

// startSequenceCounterCleanup periodically cleans up old entries in tickSequenceCounters.
func (m *MarketDataIngestor) startSequenceCounterCleanup(ctx context.Context) {
	cleanupInterval := time.Duration(m.cfg.Ingestion.TickSequenceCleanupInterval) * time.Second
	if cleanupInterval <= 0 {
		cleanupInterval = 10 * time.Minute
	}
	expiryDuration := time.Duration(m.cfg.Ingestion.MaxTickSequenceCacheDuration) * time.Second
	if expiryDuration <= 0 {
		expiryDuration = 24 * time.Hour
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
		case <-ctx.Done():
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
	loc, _ := time.LoadLocation("Asia/Kolkata")
	now := time.Now().In(loc)
	todayStart := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, loc)

	query := `
        SELECT instrument_token, timestamp, MAX(tick_sequence_id) as max_sequence_id
        FROM market_data
        WHERE timestamp >= ?
        GROUP BY instrument_token, timestamp;
    `
	err := m.dbClient.DB.Raw(query, todayStart).Scan(&results).Error
	if err != nil {
		zap.L().Error("Failed to load initial tick sequence counters", zap.Error(err))
		return
	}
	for _, r := range results {
		if _, ok := m.tickSequenceCounters[uint(r.InstrumentToken)]; !ok {
			m.tickSequenceCounters[uint(r.InstrumentToken)] = make(map[time.Time]int)
		}
		m.tickSequenceCounters[uint(r.InstrumentToken)][r.Timestamp] = r.MaxSequenceID
	}
	zap.L().Info("Loaded initial tick sequence counters", zap.Int("count", len(results)))
}

// GetWebSocketClientCount returns the number of currently connected WebSocket clients for ticks.
func (m *MarketDataIngestor) GetWebSocketClientCount() int {
	count := 0
	m.wsClients.Range(func(key, value interface{}) bool {
		count++
		return true
	})
	return count
}
