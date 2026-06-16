package data

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/marketdata"
	"github.com/Bhavik2205/ML-Bot/internal/marketdata/tickbus"
	"github.com/Bhavik2205/ML-Bot/internal/observability"
	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"github.com/gorilla/websocket"
	kitemodels "github.com/zerodha/gokiteconnect/v4/models"
	"go.uber.org/zap"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

const (
	wsPingPeriod = 45 * time.Second
	wsWriteWait  = 10 * time.Second
)

// MarketDataIngestor holds dependencies for market data ingestion and broadcasting.
type MarketDataIngestor struct {
	dbClient             *db.DBClient
	tickBus              tickbus.TickBus
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
	dbFlushdrops           uint64        // counts how many times we had to drop a DB flush due to timeout (should be 0 ideally)

	// Monitoring metrics
	dbErrors      uint64 // DB write errors (not drops)
	wsDrops       uint64 // frontend WebSocket drops
	wsBroadcasted uint64
	droppedTicks  uint64 // only counts frontend drops (not DB)
}

// NewMarketDataIngestor creates and returns a new instance of MarketDataIngestor.
func NewMarketDataIngestor(dbC *db.DBClient, tb tickbus.TickBus, wsClients *sync.Map, cfg *utils.AppConfig, _ *utils.IndicatorsConfig) *MarketDataIngestor {
	ingestor := &MarketDataIngestor{
		dbClient:             dbC,
		tickBus:              tb,
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
	go m.startTickSubscription(ctx)
	go m.startDBFlusher(ctx)
	go m.startSequenceCounterCleanup(ctx)
	zap.L().Info("🚀 Market data ingestion and broadcasting started.")
}

// startMonitoring logs metrics every 5s.
func (m *MarketDataIngestor) startMonitoring(ctx context.Context) {
	defer observability.RecoverPanic("ingestor-monitoring")
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
			dbFlushDrops := atomic.LoadUint64(&m.dbFlushdrops)
			rate := broadcasted - lastBroadcasted
			lastBroadcasted = broadcasted

			// NOTE: Prometheus counters (DBErrors, TicksDropped, etc.) are incremented
			// at event time via Inc()/Add(). Do NOT add cumulative totals here — that
			// would double-count on every monitoring cycle.

			// Update queue depth gauges only (gauges are safe to Set repeatedly).
			observability.TickQueueDepth.Set(float64(len(m.broadcastChannel)))
			observability.DBFlushQueueDepth.Set(float64(len(m.dbFlushCh)))

			zap.L().Info("MarketDataIngestor monitoring",
				zap.Uint64("db_errors", dbErrs),
				zap.Uint64("ws_drops", wsDrops),
				zap.Uint64("broadcast_rate_per_5s", rate),
				zap.Uint64("frontend_dropped_ticks", dropTicks),
				zap.Uint64("db_flush_drops", dbFlushDrops),
			)
		case <-ctx.Done():
			return
		}
	}
}

// QueueDepthProvider implementation — satisfies observability.QueueDepthProvider.
func (m *MarketDataIngestor) TickQueueLen() int      { return len(m.broadcastChannel) }
func (m *MarketDataIngestor) TickQueueCap() int      { return cap(m.broadcastChannel) }
func (m *MarketDataIngestor) DBFlushQueueLen() int   { return len(m.dbFlushCh) }
func (m *MarketDataIngestor) DBFlushQueueCap() int   { return cap(m.dbFlushCh) }
func (m *MarketDataIngestor) CandleQueueLen() int    { return 0 } // set via RegisterQueueDepthProvider override
func (m *MarketDataIngestor) CandleQueueCap() int    { return 0 }
func (m *MarketDataIngestor) IndicatorQueueLen() int { return 0 }
func (m *MarketDataIngestor) IndicatorQueueCap() int { return 0 }

// startTickSubscription subscribes to the TickBus and processes incoming ticks.
func (m *MarketDataIngestor) startTickSubscription(ctx context.Context) {
	defer observability.RecoverPanic("ingestor-tick-subscription")
	tickCh, err := m.tickBus.Subscribe(ctx)
	if err != nil {
		zap.L().Fatal("Failed to subscribe to tick bus", zap.Error(err))
	}
	for {
		select {
		case tick, ok := <-tickCh:
			if !ok {
				zap.L().Warn("TickBus channel closed, stopping ingestion")
				return
			}
			m.processTick(tick)
		case <-ctx.Done():
			return
		}
	}
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
		Mode:               nt.Mode,
	}
}

// processTick converts the enriched tick to MarketData and adds it to the buffer.
func (m *MarketDataIngestor) processTick(tick marketdata.NormalizedTick) {
	// Sequence counter
	m.sequenceMux.Lock()
	token := uint(tick.InstrumentToken)
	if _, ok := m.tickSequenceCounters[token]; !ok {
		m.tickSequenceCounters[token] = make(map[time.Time]int)
	}
	// Truncate to second so the per-timestamp key space doesn't explode at sub-second tick rates.
	normalizedTimestamp := tick.EventTime.Truncate(time.Second)
	currentSequenceID := m.tickSequenceCounters[token][normalizedTimestamp] + 1
	m.tickSequenceCounters[token][normalizedTimestamp] = currentSequenceID
	m.sequenceMux.Unlock()

	buyDepth := tick.Depth.Buy
	sellDepth := tick.Depth.Sell

	// Helper to safely get depth item from array (or slice, works for both)
	safeDepthItem := func(depthArray interface{}, idx int) (price float64, quantity int64, orders int) {
		// Handle array of DepthItem
		switch d := depthArray.(type) {
		case [5]kitemodels.DepthItem:
			if idx >= 0 && idx < len(d) {
				return d[idx].Price, int64(d[idx].Quantity), int(d[idx].Orders)
			}
		case []kitemodels.DepthItem:
			if idx >= 0 && idx < len(d) {
				return d[idx].Price, int64(d[idx].Quantity), int(d[idx].Orders)
			}
		}
		return 0, 0, 0
	}

	// Log depth warning if the first level is zero (indicating ModeLTP/Quote)
	if buyDepth[0].Price == 0 && sellDepth[0].Price == 0 {
		zap.L().Warn("Market depth appears empty – tick may be in ModeLTP/Quote",
			zap.String("symbol", tick.Symbol))
	}

	// Safely get first level for logging and heatmap
	bestBidPrice, bestBidQty, _ := safeDepthItem(buyDepth, 0)
	bestAskPrice, bestAskQty, _ := safeDepthItem(sellDepth, 0)

	zap.L().Debug("Tick received",
		zap.String("symbol", tick.Symbol),
		zap.Float64("ltp", tick.LastPrice),
		zap.Float64("bid", buyDepth[0].Price),
		zap.Float64("ask", sellDepth[0].Price),
	)

	GetMarketHeatmap().Update(
		tick.Symbol,
		tick.LastPrice,
		bestBidPrice,
		bestAskPrice,
		bestBidQty,
		bestAskQty,
		int64(tick.Volume),
		tick.LastPrice,
		tick.OHLC.Close,
	)

	// Get all 5 depth levels (always safe because arrays have length 5)
	bid1Price, bid1Qty, bid1Orders := safeDepthItem(buyDepth, 0)
	bid2Price, bid2Qty, bid2Orders := safeDepthItem(buyDepth, 1)
	bid3Price, bid3Qty, bid3Orders := safeDepthItem(buyDepth, 2)
	bid4Price, bid4Qty, bid4Orders := safeDepthItem(buyDepth, 3)
	bid5Price, bid5Qty, bid5Orders := safeDepthItem(buyDepth, 4)

	ask1Price, ask1Qty, ask1Orders := safeDepthItem(sellDepth, 0)
	ask2Price, ask2Qty, ask2Orders := safeDepthItem(sellDepth, 1)
	ask3Price, ask3Qty, ask3Orders := safeDepthItem(sellDepth, 2)
	ask4Price, ask4Qty, ask4Orders := safeDepthItem(sellDepth, 3)
	ask5Price, ask5Qty, ask5Orders := safeDepthItem(sellDepth, 4)

	md := db.MarketData{
		InstrumentToken:    tick.InstrumentToken,
		Timestamp:          normalizedTimestamp,
		TickSequenceID:     currentSequenceID,
		LastPrice:          tick.LastPrice,
		LastTradedQuantity: tick.LastTradedQuantity,
		Volume:             tick.Volume,
		AverageTradePrice:  tick.AverageTradePrice,
		NetChange:          tick.NetChange,
		Open:               tick.OHLC.Open,
		High:               tick.OHLC.High,
		Low:                tick.OHLC.Low,
		Close:              tick.OHLC.Close,
		OpenInterest:       tick.OpenInterest,
		BidPrice1:          bid1Price, BidQuantity1: uint32(bid1Qty), BidOrders1: uint32(bid1Orders),
		BidPrice2: bid2Price, BidQuantity2: uint32(bid2Qty), BidOrders2: uint32(bid2Orders),
		BidPrice3: bid3Price, BidQuantity3: uint32(bid3Qty), BidOrders3: uint32(bid3Orders),
		BidPrice4: bid4Price, BidQuantity4: uint32(bid4Qty), BidOrders4: uint32(bid4Orders),
		BidPrice5: bid5Price, BidQuantity5: uint32(bid5Qty), BidOrders5: uint32(bid5Orders),
		AskPrice1: ask1Price, AskQuantity1: uint32(ask1Qty), AskOrders1: uint32(ask1Orders),
		AskPrice2: ask2Price, AskQuantity2: uint32(ask2Qty), AskOrders2: uint32(ask2Orders),
		AskPrice3: ask3Price, AskQuantity3: uint32(ask3Qty), AskOrders3: uint32(ask3Orders),
		AskPrice4: ask4Price, AskQuantity4: uint32(ask4Qty), AskOrders4: uint32(ask4Orders),
		AskPrice5: ask5Price, AskQuantity5: uint32(ask5Qty), AskOrders5: uint32(ask5Orders),
		TotalBuyQuantity:  tick.TotalBuyQuantity,
		TotalSellQuantity: tick.TotalSellQuantity,
		DataSource:        dataSourceFromConfig(m.cfg),
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
			observability.TicksProcessed.Add(float64(len(dataToFlush)))
		case <-time.After(m.dbFlushTimeout):
			atomic.AddUint64(&m.dbFlushdrops, 1)
			observability.DBFlushDrops.Inc()
			zap.L().Error("DB flush timeout – dropping batch.",
				zap.Duration("timeout", m.dbFlushTimeout),
				zap.Int("batch_size", len(dataToFlush)))
		}
		m.marketDataBuffer = make([]db.MarketData, 0, batchSize)
		m.lastFlushTime = time.Now()
	}
	m.bufferLock.Unlock()

	// ---- FRONTEND BROADCAST: TIMEOUT (drops allowed) ----
	// Keep the original broadcast format (kitemodels.Tick) to avoid breaking the frontend.
	kiteTick := convertNormalizedToKiteTick(tick)
	frontendData, err := json.Marshal(map[string]interface{}{
		"symbol": tick.Symbol,
		"tick":   kiteTick,
	})
	if err != nil {
		zap.L().Error("Failed to marshal data for frontend broadcast", zap.Error(err))
	} else {
		select {
		case m.broadcastChannel <- frontendData:
			atomic.AddUint64(&m.wsBroadcasted, 1)
			observability.WebSocketBroadcasted.Inc()
			m.livePrices.Store(tick.Symbol, frontendData)
		case <-time.After(m.tickIngestionTimeout):
			atomic.AddUint64(&m.droppedTicks, 1)
			observability.TicksDropped.Inc()
			zap.L().Warn("Frontend broadcast timeout, dropping tick (acceptable)",
				zap.Duration("timeout", m.tickIngestionTimeout),
				zap.String("symbol", tick.Symbol))
		}
	}
}

// startDBFlusher periodically checks if the buffer needs flushing based on time.
func (m *MarketDataIngestor) startDBFlusher(ctx context.Context) {
	defer observability.RecoverPanic("ingestor-db-flusher")
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
			observability.TicksProcessed.Add(float64(len(dataToFlush)))
					// success
				case <-time.After(m.dbFlushTimeout):
					atomic.AddUint64(&m.dbFlushdrops, 1)
					zap.L().Error("Timed DB flush timeout – dropping batch.",
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
			defer observability.RecoverPanic(fmt.Sprintf("ingestor-db-worker-%d", workerID))
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
			defer observability.RecoverPanic(fmt.Sprintf("ingestor-ws-worker-%d", workerID))
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
	defer observability.RecoverPanic("ingestor-seq-cleanup")
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
