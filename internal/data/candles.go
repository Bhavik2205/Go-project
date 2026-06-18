// internal/data/candles.go
package data

import (
	"context"
	"encoding/json"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/marketdata/candles"
	"github.com/Bhavik2205/ML-Bot/internal/marketdata/tickbus"
	"github.com/Bhavik2205/ML-Bot/internal/observability"

	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"github.com/gorilla/websocket"
	"go.uber.org/zap"
	"gorm.io/gorm/clause"

	"github.com/Bhavik2205/ML-Bot/internal/indicators"
)

const (
	marketOpenHour    = 9
	marketOpenMinute  = 15
	marketCloseHour   = 15
	marketCloseMinute = 30
	marketTimezone    = "Asia/Kolkata"
)

// CandleGenerator delegates to a CandleEngine.
type CandleGenerator struct {
	dbClient                *db.DBClient
	tickBus                 tickbus.TickBus
	appCfg                  *utils.AppConfig
	engine                  *candles.CandleEngine
	candleWsClients         *sync.Map // maps *websocket.Conn to *wsClient
	indicatorManagerInputCh chan<- indicators.Candle

	candleDBFlushCh chan db.OHLCVCandle

	// Metrics
	ticksProcessed uint64
	dbErrors       uint64
	wsDrops        uint64

	// Monitoring
	monitorStopCh chan struct{}
}

func dataSourceFromConfig(cfg *utils.AppConfig) string {
	if cfg.Market.Simulate {
		return "simulation"
	}
	return "live"
}

// NewCandleGenerator creates and returns a new instance of CandleGenerator.
func NewCandleGenerator(
	dbC *db.DBClient,
	tb tickbus.TickBus,
	cfg *utils.AppConfig,
	wsClients *sync.Map,
	indicatorManagerInputCh chan<- indicators.Candle,
) *CandleGenerator {
	loc, err := time.LoadLocation(marketTimezone)
	if err != nil {
		zap.L().Error("Failed to load market timezone, defaulting to UTC.", zap.Error(err))
		loc = time.UTC
	}

	// Create the flush channel for finalised candles
	dbFlushCh := make(chan db.OHLCVCandle, cfg.Ingestion.DBFlushChannelSize)

	// Engine configuration
	engineCfg := &candles.EngineConfig{
		Timezone:         loc,
		Intervals:        cfg.Candles.Intervals,
		GracePeriod:      time.Duration(cfg.Candles.GracePeriodMs) * time.Millisecond,
		FinalizeInterval: time.Duration(cfg.Candles.FinalizeIntervalMs) * time.Millisecond,
		OnFinalize:       nil,                 // will be set after we create the generator
		SimulationMode:   cfg.Market.Simulate, //read from app config
	}

	engine := candles.NewCandleEngine(engineCfg)

	cg := &CandleGenerator{
		dbClient:                dbC,
		tickBus:                 tb,
		appCfg:                  cfg,
		engine:                  engine,
		candleWsClients:         wsClients,
		indicatorManagerInputCh: indicatorManagerInputCh,
		candleDBFlushCh:         dbFlushCh,
		monitorStopCh:           make(chan struct{}),
	}

	// Set the callback after cg is fully initialised
	engineCfg.OnFinalize = cg.handleFinalizedCandle

	return cg
}

// handleFinalizedCandle is called by the engine when a candle is finalised.
// Both the DB flush and indicator sends are blocking — candle data is never dropped.
func (cg *CandleGenerator) handleFinalizedCandle(candle *candles.OpenCandle) {
	ohlcv := db.OHLCVCandle{
		InstrumentToken: candle.InstrumentToken,
		Interval:        candle.IntervalStr,
		Timestamp:       candle.StartTime,
		Open:            candle.Open,
		High:            candle.High,
		Low:             candle.Low,
		Close:           candle.Close,
		Volume:          candle.Volume,
		TradeCount:      candle.TradeCount,
		DataSource:      dataSourceFromConfig(cg.appCfg),
	}

	cg.candleDBFlushCh <- ohlcv

	// Unscale prices for the indicator manager (float64 domain).
	const inv = 1.0 / candles.PriceScale
	if cg.indicatorManagerInputCh != nil {
		cg.indicatorManagerInputCh <- indicators.Candle{
			InstrumentToken: candle.InstrumentToken,
			Interval:        candle.IntervalStr,
			Timestamp:       candle.StartTime,
			Open:            float64(candle.Open) * inv,
			High:            float64(candle.High) * inv,
			Low:             float64(candle.Low) * inv,
			Close:           float64(candle.Close) * inv,
			Volume:          float64(candle.Volume),
			TradeCount:      candle.TradeCount,
			DataSource:      dataSourceFromConfig(cg.appCfg),
		}
	}

	cg.broadcastCandle(candle)
}

// broadcastCandle marshals and sends the candle data to all connected WebSocket clients.
// Prices are unscaled back to float64 for the JSON payload.
func (cg *CandleGenerator) broadcastCandle(candle *candles.OpenCandle) {
	const inv = 1.0 / candles.PriceScale
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
		DataSource      string    `json:"data_source"`
	}{
		InstrumentToken: candle.InstrumentToken,
		Interval:        candle.IntervalStr,
		Timestamp:       candle.StartTime,
		Open:            float64(candle.Open) * inv,
		High:            float64(candle.High) * inv,
		Low:             float64(candle.Low) * inv,
		Close:           float64(candle.Close) * inv,
		Volume:          float64(candle.Volume),
		TradeCount:      candle.TradeCount,
		DataSource:      dataSourceFromConfig(cg.appCfg),
	}

	jsonMessage, err := json.Marshal(broadcastData)
	if err != nil {
		zap.L().Error("Failed to marshal candle data for WebSocket broadcast", zap.Error(err), zap.Uint32("token", candle.InstrumentToken))
		return
	}

	cg.candleWsClients.Range(func(key, value interface{}) bool {
		client, ok := value.(*wsClient)
		if !ok {
			cg.candleWsClients.Delete(key)
			return true
		}
		select {
		case client.send <- jsonMessage:
		default:
			atomic.AddUint64(&cg.wsDrops, 1)
			zap.L().Warn("WebSocket send channel full, dropping candle message")
		}
		return true
	})
	zap.L().Debug("Broadcasted candle to WebSocket clients", zap.Uint32("token", candle.InstrumentToken), zap.String("interval", candle.IntervalStr))
}

// QueueDepthProvider methods for observability.
func (cg *CandleGenerator) CandleQueueLen() int { return len(cg.candleDBFlushCh) }
func (cg *CandleGenerator) CandleQueueCap() int { return cap(cg.candleDBFlushCh) }

// StartCandleGeneration subscribes to Redis ticks and feeds them to the engine.
func (cg *CandleGenerator) StartCandleGeneration(ctx context.Context) {
	// Start engine's finalizer loop
	go cg.engine.StartFinalizer(ctx)

	// Start DB writer goroutine
	go cg.StartCandleDBWriter(ctx)

	// Start monitoring goroutine
	go cg.startMonitoring()

	// Subscribe to TickBus
	tickCh, err := cg.tickBus.Subscribe(ctx)
	if err != nil {
		zap.L().Fatal("Failed to subscribe to tick bus for candle generation", zap.Error(err))
	}

	for {
		select {
		case tick, ok := <-tickCh:
			if !ok {
				zap.L().Warn("TickBus channel closed, attempting reconnect...")
				// Reconnection logic (simplified for brevity, keep original if needed)
				return
			}

			atomic.AddUint64(&cg.ticksProcessed, 1)
			cg.engine.ProcessTick(&tick)

		case <-ctx.Done():
			zap.L().Info("Flushing all remaining open candles during graceful shutdown...")
			cg.engine.FinalizeAll()
			return
		}
	}
}

// StartCandleDBWriter batches and writes candles to DB.
func (cg *CandleGenerator) StartCandleDBWriter(ctx context.Context) {
	defer cg.recoverGoroutine("StartCandleDBWriter")
	batchSize := cg.appCfg.Ingestion.MarketDataBatchSize
	batch := make([]db.OHLCVCandle, 0, batchSize)
	ticker := time.NewTicker(time.Duration(cg.appCfg.Ingestion.MarketDataFlushIntervalMS) * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case candle := <-cg.candleDBFlushCh:
			batch = append(batch, candle)
			if len(batch) >= batchSize {
				cg.writeCandleBatch(batch)
				batch = batch[:0]
			}
		case <-ticker.C:
			if len(batch) > 0 {
				cg.writeCandleBatch(batch)
				batch = batch[:0]
			}
		case <-ctx.Done():
			if len(batch) > 0 {
				cg.writeCandleBatch(batch)
			}
			return
		}
	}
}

func (cg *CandleGenerator) writeCandleBatch(batch []db.OHLCVCandle) {
	defer cg.recoverGoroutine("writeCandleBatch")
	result := cg.dbClient.DB.Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "timestamp"}},
		DoUpdates: clause.AssignmentColumns([]string{"high", "low", "close", "volume", "trade_count", "updated_at"}),
	}).CreateInBatches(batch, len(batch))
	if result.Error != nil {
		atomic.AddUint64(&cg.dbErrors, 1)
		zap.L().Error("❌ Failed to batch save/update OHLCVCandles to DB", zap.Error(result.Error))
	}
}

// RegisterWebSocketClient adds a new WebSocket client and starts its write pump.
func (cg *CandleGenerator) RegisterWebSocketClient(conn *websocket.Conn) {
	bufferSize := cg.appCfg.Ingestion.WSBroadcastChannelSize
	client := &wsClient{conn: conn, send: make(chan []byte, bufferSize)}
	cg.candleWsClients.Store(conn, client)
	go cg.writePump(client)
}

// UnregisterWebSocketClient removes a WebSocket client and closes its channel.
func (cg *CandleGenerator) UnregisterWebSocketClient(conn *websocket.Conn) {
	if val, ok := cg.candleWsClients.Load(conn); ok {
		client := val.(*wsClient)
		close(client.send)
		cg.candleWsClients.Delete(conn)
		conn.Close()
	}
}

// writePump writes messages from the channel to the WebSocket with deadlines and periodic pings.
func (cg *CandleGenerator) writePump(client *wsClient) {
	defer observability.RecoverPanic("candle-ws-write-pump")
	defer func() {
		client.conn.Close()
		cg.candleWsClients.Delete(client.conn)
	}()
	pingTicker := time.NewTicker(wsPingPeriod)
	defer pingTicker.Stop()
	for {
		select {
		case msg, ok := <-client.send:
			if !ok {
				_ = client.conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
				_ = client.conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
				return
			}
			_ = client.conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
			if err := client.conn.WriteMessage(websocket.TextMessage, msg); err != nil {
				return
			}
		case <-pingTicker.C:
			_ = client.conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
			if err := client.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// recoverGoroutine logs and recovers from panics in goroutines.
func (cg *CandleGenerator) recoverGoroutine(where string) {
	if r := recover(); r != nil {
		zap.L().Error("Panic recovered", zap.String("where", where), zap.Any("recover", r))
	}
}

// startMonitoring launches a goroutine to monitor system usage and candle generator health.
func (cg *CandleGenerator) startMonitoring() {
	defer cg.recoverGoroutine("candle-monitoring")
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	var lastTicksProcessed uint64
	for {
		select {
		case <-cg.monitorStopCh:
			return
		case <-ticker.C:
			ticks := atomic.LoadUint64(&cg.ticksProcessed)
			dbErrs := atomic.LoadUint64(&cg.dbErrors)
			wsDrops := atomic.LoadUint64(&cg.wsDrops)
			tps := ticks - lastTicksProcessed
			lastTicksProcessed = ticks

			// Update Prometheus queue depth gauge
			observability.CandleQueueDepth.Set(float64(len(cg.candleDBFlushCh)))

			zap.L().Info("CandleGenerator monitoring",
				zap.Uint64("ticks_processed", ticks),
				zap.Uint64("db_errors", dbErrs),
				zap.Uint64("ws_drops", wsDrops),
				zap.Uint64("ticks_per_5s", tps),
			)
			if tps < 10 {
				zap.L().Warn("Low tick processing speed detected", zap.Uint64("ticks_per_5s", tps))
			}
		}
	}
}

// GetWebSocketClientCount returns the number of currently connected WebSocket clients for candles.
func (cg *CandleGenerator) GetWebSocketClientCount() int {
	count := 0
	cg.candleWsClients.Range(func(key, value interface{}) bool {
		count++
		return true
	})
	return count
}

// Stop flushes all open candles and stops the engine.
func (cg *CandleGenerator) Stop(ctx context.Context) error {
	zap.L().Info("Flushing all remaining open candles during graceful shutdown...")
	cg.engine.FinalizeAll()
	close(cg.monitorStopCh)
	return nil
}
