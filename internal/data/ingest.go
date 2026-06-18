// internal/data/ingest.go
package data

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/marketdata"
	"github.com/Bhavik2205/ML-Bot/internal/marketdata/candles"
	"github.com/Bhavik2205/ML-Bot/internal/marketdata/tickbus"
	"github.com/Bhavik2205/ML-Bot/internal/observability"
	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"github.com/gorilla/websocket"
	kitemodels "github.com/zerodha/gokiteconnect/v4/models"
	"go.uber.org/zap"
)

const (
	wsPingPeriod = 45 * time.Second
	wsWriteWait  = 10 * time.Second
)

// scalePrice converts a float64 price to a scaled int64 using candles.PriceScale.
func scalePrice(p float64) int64 { return int64(p * candles.PriceScale) }

// sync.Pool for JSON encoding buffers (kept for now)
var encodePool = sync.Pool{
	New: func() interface{} {
		return new(bytes.Buffer)
	},
}

// MarketDataIngestor holds dependencies.
type MarketDataIngestor struct {
	dbClient         *db.DBClient
	tickBus          tickbus.TickBus
	wsClients        *sync.Map
	marketDataBuffer []db.MarketData
	bufferLock       sync.Mutex
	lastFlushTime    time.Time
	sequenceCounter  *SequenceCounter
	broadcastChannel chan []byte
	livePrices       *sync.Map
	cfg              *utils.AppConfig

	dbFlushCh              chan []db.MarketData
	dbWorkerCount          int
	wsBroadcastWorkerCount int
	dbFlushdrops           uint64

	// Monitoring metrics
	dbErrors               uint64
	wsDrops                uint64
	wsBroadcasted          uint64
	frontendBroadcastDrops uint64
	workerPool             *TickWorkerPool

	// Graceful shutdown
	shuttingDown int32
	wgDB         sync.WaitGroup
	closeOnce    sync.Once
}

// NewMarketDataIngestor creates and returns a new instance.
func NewMarketDataIngestor(dbC *db.DBClient, tb tickbus.TickBus, wsClients *sync.Map, cfg *utils.AppConfig, _ *utils.IndicatorsConfig) *MarketDataIngestor {
	ingestor := &MarketDataIngestor{
		dbClient:         dbC,
		tickBus:          tb,
		wsClients:        wsClients,
		marketDataBuffer: make([]db.MarketData, 0, cfg.Ingestion.MarketDataBatchSize),
		lastFlushTime:    time.Now(),
		sequenceCounter:  NewSequenceCounter(),
		broadcastChannel: make(chan []byte, cfg.Ingestion.WSBroadcastChannelSize),
		livePrices:       &sync.Map{},
		cfg:              cfg,

		dbFlushCh:              make(chan []db.MarketData, cfg.Ingestion.DBFlushChannelSize),
		dbWorkerCount:          cfg.Ingestion.DBWorkerCount,
		wsBroadcastWorkerCount: cfg.Ingestion.WSBroadcastWorkerCount,
	}

	return ingestor
}

// StartIngestionAndBroadcast starts all components.
func (m *MarketDataIngestor) StartIngestionAndBroadcast(ctx context.Context) {
	// Create worker pool with configurable worker count.
	workerCount := m.cfg.Ingestion.TickWorkerCount
	if workerCount <= 0 {
		workerCount = 16
	}
	m.workerPool = NewTickWorkerPool(
		ctx,
		workerCount,
		func(ctx context.Context, tick marketdata.NormalizedTick) {
			m.processTick(ctx, tick)
		},
	)

	m.startDBWorkers(ctx)
	m.startWebSocketBroadcasterWorkers(ctx)
	go m.startMonitoring(ctx)
	go m.startDBFlushQueueWatcher(ctx)
	go m.startTickSubscription(ctx)
	go m.startDBFlusher(ctx)

	// Graceful shutdown handler
	go func() {
		<-ctx.Done()
		m.shutdown()
	}()
	zap.L().Info("Market data ingestion and broadcasting started.")
}

// shutdown performs a graceful shutdown.
func (m *MarketDataIngestor) shutdown() {
	// 1. Stop accepting new ticks.
	atomic.StoreInt32(&m.shuttingDown, 1)

	// 2. Stop worker pool (this will cause workers to finish after their current tick).
	if m.workerPool != nil {
		m.workerPool.Close()
	}

	// 3. Flush any remaining data in the buffer.
	m.bufferLock.Lock()
	if len(m.marketDataBuffer) > 0 {
		dataToFlush := m.marketDataBuffer
		m.marketDataBuffer = make([]db.MarketData, 0, m.cfg.Ingestion.MarketDataBatchSize)
		m.bufferLock.Unlock()

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := m.enqueueDBFlush(ctx, dataToFlush); err != nil {
			atomic.AddUint64(&m.dbErrors, 1)
			zap.L().Error("Final flush enqueue failed", zap.Error(err))
		}
	} else {
		m.bufferLock.Unlock()
	}

	// 4. Close the DB flush channel so workers drain and exit.
	m.closeOnce.Do(func() {
		close(m.dbFlushCh)
	})

	// 5. Wait for all DB workers to finish processing queued batches.
	m.wgDB.Wait()

	zap.L().Info("Market data ingestor shutdown complete")
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
			frontendDrops := atomic.LoadUint64(&m.frontendBroadcastDrops)
			dbFlushDrops := atomic.LoadUint64(&m.dbFlushdrops)
			rate := broadcasted - lastBroadcasted
			lastBroadcasted = broadcasted

			observability.TickQueueDepth.Set(float64(len(m.broadcastChannel)))
			observability.DBFlushQueueDepth.Set(float64(len(m.dbFlushCh)))

			// Also log worker pool depth if available.
			if m.workerPool != nil {
				zap.L().Debug("Worker pool depth", zap.Int("queued", m.workerPool.TotalQueueDepth()))
			}

			zap.L().Info("MarketDataIngestor monitoring",
				zap.Uint64("db_errors", dbErrs),
				zap.Uint64("ws_drops", wsDrops),
				zap.Uint64("broadcast_rate_per_5s", rate),
				zap.Uint64("frontend_broadcast_drops", frontendDrops),
				zap.Uint64("db_flush_drops", dbFlushDrops),
			)
		case <-ctx.Done():
			return
		}
	}
}

// QueueDepthProvider methods...
func (m *MarketDataIngestor) TickQueueLen() int      { return len(m.broadcastChannel) }
func (m *MarketDataIngestor) TickQueueCap() int      { return cap(m.broadcastChannel) }
func (m *MarketDataIngestor) DBFlushQueueLen() int   { return len(m.dbFlushCh) }
func (m *MarketDataIngestor) DBFlushQueueCap() int   { return cap(m.dbFlushCh) }
func (m *MarketDataIngestor) CandleQueueLen() int    { return 0 }
func (m *MarketDataIngestor) CandleQueueCap() int    { return 0 }
func (m *MarketDataIngestor) IndicatorQueueLen() int { return 0 }
func (m *MarketDataIngestor) IndicatorQueueCap() int { return 0 }

// startTickSubscription subscribes to the TickBus and submits to worker pool.
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
			if atomic.LoadInt32(&m.shuttingDown) == 1 {
				continue
			}
			if err := m.workerPool.Submit(tick); err != nil {
				if err == context.Canceled {
					return
				}
				zap.L().Error("worker pool submit failed", zap.Error(err))
			}
		case <-ctx.Done():
			return
		}
	}
}

// tickBroadcastMsg...
type tickBroadcastMsg struct {
	Symbol string          `json:"symbol"`
	Tick   kitemodels.Tick `json:"tick"`
}

// convertNormalizedToKiteTick...
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

// safeDepthItem...
func safeDepthItem(depthArray interface{}, idx int) (price float64, quantity int64, orders int) {
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

// processTick handles a single tick.
func (m *MarketDataIngestor) processTick(ctx context.Context, tick marketdata.NormalizedTick) {
	// Generate sequence ID (lock‑free)
	normalizedTimestamp := tick.EventTime.Truncate(time.Second)
	currentSequenceID := int(
		m.sequenceCounter.Next(
			tick.InstrumentToken,
			normalizedTimestamp,
		),
	)

	buyDepth := tick.Depth.Buy
	sellDepth := tick.Depth.Sell

	if buyDepth[0].Price == 0 && sellDepth[0].Price == 0 {
		zap.L().Warn("Market depth appears empty; tick may be in ModeLTP/Quote",
			zap.String("symbol", tick.Symbol))
	}

	bestBidPrice, bestBidQty, _ := safeDepthItem(buyDepth, 0)
	bestAskPrice, bestAskQty, _ := safeDepthItem(sellDepth, 0)

	zap.L().Debug("Tick received",
		zap.String("symbol", tick.Symbol),
		zap.Float64("ltp", tick.LastPrice),
		zap.Float64("bid", buyDepth[0].Price),
		zap.Float64("ask", sellDepth[0].Price),
	)

	// Update heatmap (synchronous for now; later can be made async)
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
		LastPrice:          scalePrice(tick.LastPrice),
		LastTradedQuantity: tick.LastTradedQuantity,
		Volume:             tick.Volume,
		AverageTradePrice:  scalePrice(tick.AverageTradePrice),
		NetChange:          scalePrice(tick.NetChange),
		Open:               scalePrice(tick.OHLC.Open),
		High:               scalePrice(tick.OHLC.High),
		Low:                scalePrice(tick.OHLC.Low),
		Close:              scalePrice(tick.OHLC.Close),
		OpenInterest:       tick.OpenInterest,
		BidPrice1:          scalePrice(bid1Price), BidQuantity1: uint32(bid1Qty), BidOrders1: uint32(bid1Orders),
		BidPrice2: scalePrice(bid2Price), BidQuantity2: uint32(bid2Qty), BidOrders2: uint32(bid2Orders),
		BidPrice3: scalePrice(bid3Price), BidQuantity3: uint32(bid3Qty), BidOrders3: uint32(bid3Orders),
		BidPrice4: scalePrice(bid4Price), BidQuantity4: uint32(bid4Qty), BidOrders4: uint32(bid4Orders),
		BidPrice5: scalePrice(bid5Price), BidQuantity5: uint32(bid5Qty), BidOrders5: uint32(bid5Orders),
		AskPrice1: scalePrice(ask1Price), AskQuantity1: uint32(ask1Qty), AskOrders1: uint32(ask1Orders),
		AskPrice2: scalePrice(ask2Price), AskQuantity2: uint32(ask2Qty), AskOrders2: uint32(ask2Orders),
		AskPrice3: scalePrice(ask3Price), AskQuantity3: uint32(ask3Qty), AskOrders3: uint32(ask3Orders),
		AskPrice4: scalePrice(ask4Price), AskQuantity4: uint32(ask4Qty), AskOrders4: uint32(ask4Orders),
		AskPrice5: scalePrice(ask5Price), AskQuantity5: uint32(ask5Qty), AskOrders5: uint32(ask5Orders),
		TotalBuyQuantity:  tick.TotalBuyQuantity,
		TotalSellQuantity: tick.TotalSellQuantity,
		DataSource:        dataSourceFromConfig(m.cfg),
	}

	// Buffer management
	m.bufferLock.Lock()
	m.marketDataBuffer = append(m.marketDataBuffer, md)
	batchSize := m.cfg.Ingestion.MarketDataBatchSize
	if len(m.marketDataBuffer) >= batchSize {
		dataToFlush := m.marketDataBuffer
		m.marketDataBuffer = make([]db.MarketData, 0, batchSize)
		m.bufferLock.Unlock()
		if err := m.enqueueDBFlush(ctx, dataToFlush); err != nil {
			atomic.AddUint64(&m.dbErrors, 1)
			zap.L().Error("Failed to enqueue market_data batch for DB persistence",
				zap.Int("batch_size", len(dataToFlush)),
				zap.Error(err))
		}
		m.lastFlushTime = time.Now()
	} else {
		m.bufferLock.Unlock()
	}

	// Broadcast to frontend (JSON for now)
	kiteTick := convertNormalizedToKiteTick(tick)
	buf := encodePool.Get().(*bytes.Buffer)
	buf.Reset()
	defer encodePool.Put(buf)

	encoder := json.NewEncoder(buf)
	if err := encoder.Encode(tickBroadcastMsg{Symbol: tick.Symbol, Tick: kiteTick}); err != nil {
		zap.L().Error("Failed to marshal data for frontend broadcast", zap.Error(err))
		return
	}
	frontendData := make([]byte, buf.Len())
	copy(frontendData, buf.Bytes())

	select {
	case m.broadcastChannel <- frontendData:
		atomic.AddUint64(&m.wsBroadcasted, 1)
		observability.WebSocketBroadcasted.Inc()
		m.livePrices.Store(tick.Symbol, frontendData)
	default:
		atomic.AddUint64(&m.frontendBroadcastDrops, 1)
		observability.WebSocketBroadcastDrops.Inc()
	}
}

// enqueueDBFlush...
func (m *MarketDataIngestor) enqueueDBFlush(ctx context.Context, dataToFlush []db.MarketData) error {
	select {
	case m.dbFlushCh <- dataToFlush:
		m.wgDB.Add(1)
		observability.TicksProcessed.Add(float64(len(dataToFlush)))
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// startDBFlushQueueWatcher...
func (m *MarketDataIngestor) startDBFlushQueueWatcher(ctx context.Context) {
	defer observability.RecoverPanic("ingestor-db-flush-queue-watcher")
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	const warnThreshold = 0.80
	for {
		select {
		case <-ticker.C:
			capacity := cap(m.dbFlushCh)
			if capacity == 0 {
				continue
			}
			length := len(m.dbFlushCh)
			usage := float64(length) / float64(capacity)
			if usage >= warnThreshold {
				zap.L().Warn("db flush queue high usage",
					zap.Float64("usage_pct", usage*100),
					zap.Int("length", length),
					zap.Int("capacity", capacity))
			}
		case <-ctx.Done():
			return
		}
	}
}

// startDBFlusher...
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
				dataToFlush := m.marketDataBuffer
				m.marketDataBuffer = make([]db.MarketData, 0, m.cfg.Ingestion.MarketDataBatchSize)
				m.bufferLock.Unlock()
				if err := m.enqueueDBFlush(ctx, dataToFlush); err != nil {
					atomic.AddUint64(&m.dbErrors, 1)
					zap.L().Error("Timed flush failed to enqueue batch for DB persistence",
						zap.Int("batch_size", len(dataToFlush)),
						zap.Error(err))
				} else {
					zap.L().Debug("Timed flush enqueued batch for DB worker", zap.Int("count", len(dataToFlush)))
				}
				m.lastFlushTime = time.Now()
			} else {
				m.bufferLock.Unlock()
			}
		case <-ctx.Done():
			return
		}
	}
}

// startDBWorkers – each worker now holds a dedicated pgx.Conn.
func (m *MarketDataIngestor) startDBWorkers(ctx context.Context) {
	for i := 0; i < m.dbWorkerCount; i++ {
		go func(workerID int) {
			defer observability.RecoverPanic(fmt.Sprintf("ingestor-db-worker-%d", workerID))
			zap.L().Info("DB worker started", zap.Int("worker_id", workerID))

			stageName := fmt.Sprintf("market_data_stage_%d", workerID)

			conn, err := m.dbClient.Pool.Acquire(ctx)
			if err != nil {
				zap.L().Fatal("Failed to acquire database connection for worker",
					zap.Int("worker_id", workerID), zap.Error(err))
			}
			defer conn.Release()

			if err := m.dbClient.CreateTempStageTable(ctx, conn.Conn(), stageName); err != nil {
				zap.L().Fatal("Failed to create temp stage table",
					zap.Int("worker_id", workerID), zap.Error(err))
			}
			zap.L().Info("Temp stage table created", zap.Int("worker_id", workerID), zap.String("stage", stageName))

			for {
				select {
				case dataToFlush, ok := <-m.dbFlushCh:
					if !ok {
						zap.L().Info("DB flush channel closed, worker stopping", zap.Int("worker_id", workerID))
						return
					}
					start := time.Now()
					maxRetries := 3
					var err error
					for attempt := 1; attempt <= maxRetries; attempt++ {
						err = m.dbClient.CopyMarketDataBatchWithConn(ctx, conn.Conn(), stageName, dataToFlush)
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
						zap.L().Error("DB worker failed to batch insert market data",
							zap.Error(err), zap.Int("batch_size", len(dataToFlush)), zap.Duration("duration", duration))
					} else {
						zap.L().Debug("DB worker inserted batch", zap.Int("rows", len(dataToFlush)), zap.Duration("duration", duration))
					}
					m.wgDB.Done()
				case <-ctx.Done():
					zap.L().Info("Context cancelled, DB worker stopping", zap.Int("worker_id", workerID))
					return
				}
			}
		}(i)
	}
}

// startWebSocketBroadcasterWorkers...
func (m *MarketDataIngestor) startWebSocketBroadcasterWorkers(ctx context.Context) {
	for i := 0; i < m.wsBroadcastWorkerCount; i++ {
		go func(workerID int) {
			defer observability.RecoverPanic(fmt.Sprintf("ingestor-ws-worker-%d", workerID))
			zap.L().Info("WS dispatcher worker started", zap.Int("worker_id", workerID))
			for {
				select {
				case msg, ok := <-m.broadcastChannel:
					if !ok {
						zap.L().Info("WS dispatch channel closed, worker stopping", zap.Int("worker_id", workerID))
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
							zap.L().Warn("Dropping WebSocket message for client because write channel is full",
								zap.String("remote_addr", conn.RemoteAddr().String()))
						}
						return true
					})
				case <-ctx.Done():
					zap.L().Info("Context cancelled, WS dispatcher worker stopping", zap.Int("worker_id", workerID))
					return
				}
			}
		}(i)
	}
}

// RegisterWebSocketClient...
func (m *MarketDataIngestor) RegisterWebSocketClient(conn *websocket.Conn) {
	clientWriteCh := make(chan []byte, 1024)
	m.wsClients.Store(conn, clientWriteCh)
	zap.L().Info("New WebSocket client connected", zap.String("remote_addr", conn.RemoteAddr().String()))
	go m.writePump(conn, clientWriteCh)
	m.livePrices.Range(func(key, value interface{}) bool {
		data := value.([]byte)
		select {
		case clientWriteCh <- data:
		default:
			zap.L().Warn("Failed to send initial live price to new WS client because channel is full",
				zap.String("symbol", key.(string)))
			conn.Close()
			m.wsClients.Delete(conn)
			return false
		}
		return true
	})
}

// UnregisterWebSocketClient...
func (m *MarketDataIngestor) UnregisterWebSocketClient(conn *websocket.Conn) {
	if clientWriteCh, ok := m.wsClients.LoadAndDelete(conn); ok {
		close(clientWriteCh.(chan []byte))
		conn.Close()
		zap.L().Info("WebSocket client disconnected", zap.String("remote_addr", conn.RemoteAddr().String()))
	}
}

// writePump...
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

// GetWebSocketClientCount...
func (m *MarketDataIngestor) GetWebSocketClientCount() int {
	count := 0
	m.wsClients.Range(func(key, value interface{}) bool {
		count++
		return true
	})
	return count
}
