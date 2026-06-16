package data

import (
	"bytes"
	"context"
	"encoding/json"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/indicators"
	"github.com/Bhavik2205/ML-Bot/internal/observability"
	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

// wsClient and WebSocket timing constants are shared with candles.go / ingest.go.
type wsClient struct {
	conn *websocket.Conn
	send chan []byte
}

// stateKey uniquely identifies one (instrument, interval) indicator state set.
type stateKey struct {
	token    uint32
	interval string
}

// IndicatorManager aggregates completed OHLCV candles, updates stateful
// incremental indicators in O(1), persists results via a batch writer, and
// broadcasts a single consolidated WS message per candle event.
type IndicatorManager struct {
	dbClient           *db.DBClient
	appCfg             *utils.AppConfig
	indicatorsCfg      *utils.IndicatorsConfig
	inputCandleCh      <-chan indicators.Candle
	indicatorWsClients *sync.Map

	// stateMu protects states map. Per-entry locking is not needed because
	// processCandle already runs on a single goroutine (the input loop).
	stateMu sync.RWMutex
	states  map[stateKey]*indicators.IndicatorStateSet

	batchWriter *IndicatorBatchWriter

	shedding atomic.Bool

	// metrics
	indicatorsProcessed uint64
	dbErrors            uint64
	wsDrops             uint64
}

// jsonBufPool avoids per-broadcast heap allocations for the WS envelope.
var jsonBufPool = sync.Pool{New: func() interface{} { return new(bytes.Buffer) }}

// NewIndicatorManager creates and wires up a new IndicatorManager.
func NewIndicatorManager(
	dbC *db.DBClient,
	appCfg *utils.AppConfig,
	indicatorsCfg *utils.IndicatorsConfig,
	inputCandleCh <-chan indicators.Candle,
	wsClients *sync.Map,
) *IndicatorManager {
	flushSize := indicatorsCfg.OutputChannelBufferSize
	if flushSize <= 0 {
		flushSize = 200
	}
	// Flush every 500 ms; size-triggered flush fires earlier under load.
	bw := NewIndicatorBatchWriter(dbC.DB, flushSize, 500*time.Millisecond)

	return &IndicatorManager{
		dbClient:           dbC,
		appCfg:             appCfg,
		indicatorsCfg:      indicatorsCfg,
		inputCandleCh:      inputCandleCh,
		indicatorWsClients: wsClients,
		states:             make(map[stateKey]*indicators.IndicatorStateSet),
		batchWriter:        bw,
	}
}

// IndicatorQueueLen / IndicatorQueueCap satisfy the observability interface.
// With stateful indicators there is no intermediate channel, so we expose the
// batch writer's pending-row count as a proxy for queue depth.
func (im *IndicatorManager) IndicatorQueueLen() int { return im.batchWriter.PendingCount() }
func (im *IndicatorManager) IndicatorQueueCap() int { return im.indicatorsCfg.OutputChannelBufferSize }

// SetShedding enables or disables load shedding.
func (im *IndicatorManager) SetShedding(v bool) { im.shedding.Store(v) }

// -------------------------------------------------------------------
// WebSocket client management
// -------------------------------------------------------------------

func (im *IndicatorManager) writePump(client *wsClient) {
	defer observability.RecoverPanic("indicator-ws-write-pump")
	defer func() {
		client.conn.Close()
		im.indicatorWsClients.Delete(client.conn)
	}()
	pingTicker := time.NewTicker(wsPingPeriod)
	defer pingTicker.Stop()
	for {
		select {
		case msg, ok := <-client.send:
			if !ok {
				_ = client.conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
				_ = client.conn.WriteMessage(websocket.CloseMessage,
					websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
				return
			}
			_ = client.conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
			if err := client.conn.WriteMessage(websocket.TextMessage, msg); err != nil {
				zap.L().Error("WebSocket write error, closing connection", zap.Error(err))
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

func (im *IndicatorManager) RegisterWebSocketClient(conn *websocket.Conn) {
	client := &wsClient{conn: conn, send: make(chan []byte, 256)}
	im.indicatorWsClients.Store(conn, client)
	go im.writePump(client)
	zap.L().Info("Indicator WebSocket client registered",
		zap.String("remote_addr", conn.RemoteAddr().String()))
}

func (im *IndicatorManager) UnregisterWebSocketClient(conn *websocket.Conn) {
	if val, ok := im.indicatorWsClients.LoadAndDelete(conn); ok {
		close(val.(*wsClient).send)
	}
	conn.Close()
}

func (im *IndicatorManager) GetWebSocketClientCount() int {
	count := 0
	im.indicatorWsClients.Range(func(_, _ interface{}) bool { count++; return true })
	return count
}

// -------------------------------------------------------------------
// Main loop
// -------------------------------------------------------------------

func (im *IndicatorManager) StartIndicatorCalculations(ctx context.Context) {
	zap.L().Info("✅ Indicator manager started (stateful/incremental mode)")

	// Warm up stateful indicators from DB history so they're ready immediately.
	im.warmUpFromHistory()

	// Batch writer flushes on timer and size threshold.
	go im.batchWriter.Run(ctx)
	go im.startMonitoring(ctx)

	for {
		select {
		case candle, ok := <-im.inputCandleCh:
			if !ok {
				zap.L().Error("Indicator input channel closed. Stopping.")
				return
			}
			if im.shedding.Load() {
				continue
			}
			im.processCandle(candle)
		case <-ctx.Done():
			zap.L().Info("Context cancelled, stopping indicator manager.")
			return
		}
	}
}

// -------------------------------------------------------------------
// Hot path – O(1) per candle
// -------------------------------------------------------------------

func (im *IndicatorManager) processCandle(c indicators.Candle) {
	key := stateKey{token: c.InstrumentToken, interval: c.Interval}

	im.stateMu.RLock()
	s, ok := im.states[key]
	im.stateMu.RUnlock()

	if !ok {
		s = im.newStateSet()
		im.stateMu.Lock()
		// double-check after write lock
		if existing, exists := im.states[key]; exists {
			s = existing
		} else {
			im.states[key] = s
		}
		im.stateMu.Unlock()
	}

	cfg := im.indicatorsCfg
	src := dataSource(im.appCfg)
	ts := c.Timestamp

	// Accumulate all results for a single consolidated WS broadcast.
	// Pre-allocate for up to 10 indicators.
	wsPayload := make(map[string]interface{}, 10)

	// --- SMA ---
	if cfg.SMA.Enabled {
		if val, ready := s.SMA.Update(c.Close); ready {
			im.batchWriter.AddSMA(db.IndicatorSMA{
				InstrumentToken: c.InstrumentToken, Interval: c.Interval,
				Period: cfg.SMA.Period, Timestamp: ts, Value: val, DataSource: src,
			})
			wsPayload["sma"] = map[string]interface{}{"period": cfg.SMA.Period, "value": val}
			atomic.AddUint64(&im.indicatorsProcessed, 1)
		}
	}

	// --- EMA ---
	if cfg.EMA.Enabled {
		if val, ready := s.EMA.Update(c.Close); ready {
			im.batchWriter.AddEMA(db.IndicatorEMA{
				InstrumentToken: c.InstrumentToken, Interval: c.Interval,
				Period: cfg.EMA.ShortPeriod, Timestamp: ts, Value: val, DataSource: src,
			})
			wsPayload["ema"] = map[string]interface{}{"period": cfg.EMA.ShortPeriod, "value": val}
			atomic.AddUint64(&im.indicatorsProcessed, 1)
		}
	}

	// --- RSI ---
	if cfg.RSI.Enabled {
		if val, ready := s.RSI.Update(c.Close); ready {
			im.batchWriter.AddRSI(db.IndicatorRSI{
				InstrumentToken: c.InstrumentToken, Interval: c.Interval,
				Period: cfg.RSI.Period, Timestamp: ts, Value: val, DataSource: src,
			})
			wsPayload["rsi"] = map[string]interface{}{"period": cfg.RSI.Period, "value": val}
			atomic.AddUint64(&im.indicatorsProcessed, 1)
		}
	}

	// --- MACD ---
	if cfg.MACD.Enabled {
		if macdLine, sigLine, hist, ready := s.MACD.Update(c.Close); ready {
			im.batchWriter.AddMACD(db.IndicatorMACD{
				InstrumentToken: c.InstrumentToken, Interval: c.Interval,
				FastPeriod: cfg.MACD.FastPeriod, SlowPeriod: cfg.MACD.SlowPeriod, SignalPeriod: cfg.MACD.SignalPeriod,
				Timestamp: ts, MACDLine: macdLine, SignalLine: sigLine, Histogram: hist, DataSource: src,
			})
			wsPayload["macd"] = map[string]interface{}{"macd_line": macdLine, "signal_line": sigLine, "histogram": hist}
			atomic.AddUint64(&im.indicatorsProcessed, 1)
		}
	}

	// --- ATR ---
	if cfg.ATR.Enabled {
		if val, ready := s.ATR.Update(c.High, c.Low, c.Close); ready {
			im.batchWriter.AddATR(db.IndicatorATR{
				InstrumentToken: c.InstrumentToken, Interval: c.Interval,
				Period: cfg.ATR.Period, Timestamp: ts, Value: val, DataSource: src,
			})
			wsPayload["atr"] = map[string]interface{}{"period": cfg.ATR.Period, "value": val}
			atomic.AddUint64(&im.indicatorsProcessed, 1)
		}
	}

	// --- Bollinger Bands ---
	if cfg.BollingerBands.Enabled {
		if upper, middle, lower, ready := s.Bollinger.Update(c.Close); ready {
			im.batchWriter.AddBollinger(db.IndicatorBollingerBands{
				InstrumentToken: c.InstrumentToken, Interval: c.Interval,
				Period: cfg.BollingerBands.Period, NumStdDev: cfg.BollingerBands.NumStdDev,
				Timestamp: ts, UpperBand: upper, MiddleBand: middle, LowerBand: lower, DataSource: src,
			})
			wsPayload["bollinger"] = map[string]interface{}{"upper": upper, "middle": middle, "lower": lower}
			atomic.AddUint64(&im.indicatorsProcessed, 1)
		}
	}

	// --- Stochastic ---
	if cfg.Stochastic.Enabled {
		if k, d, ready := s.Stochastic.Update(c.High, c.Low, c.Close); ready {
			im.batchWriter.AddStochastic(db.IndicatorStochastic{
				InstrumentToken: c.InstrumentToken, Interval: c.Interval,
				KPeriod: cfg.Stochastic.KPeriod, DPeriod: cfg.Stochastic.DPeriod,
				Timestamp: ts, KValue: k, DValue: d, DataSource: src,
			})
			wsPayload["stochastic"] = map[string]interface{}{"k": k, "d": d}
			atomic.AddUint64(&im.indicatorsProcessed, 1)
		}
	}

	// --- OBV ---
	if cfg.OBV.Enabled {
		if val, ready := s.OBV.Update(c.Close, c.Volume); ready {
			im.batchWriter.AddOBV(db.IndicatorOBV{
				InstrumentToken: c.InstrumentToken, Interval: c.Interval,
				Timestamp: ts, Value: val, DataSource: src,
			})
			wsPayload["obv"] = val
			atomic.AddUint64(&im.indicatorsProcessed, 1)
		}
	}

	// --- VWAP ---
	if cfg.VWAP.Enabled {
		if val, ready := s.VWAP.Update(c.Open, c.High, c.Low, c.Close, c.Volume); ready {
			im.batchWriter.AddVWAP(db.IndicatorVWAP{
				InstrumentToken: c.InstrumentToken, Interval: c.Interval,
				Timestamp: ts, Value: val, DataSource: src,
			})
			wsPayload["vwap"] = val
			atomic.AddUint64(&im.indicatorsProcessed, 1)
		}
	}

	// --- ADX ---
	if cfg.ADX.Enabled {
		if adxVal, plusDI, minusDI, ready := s.ADX.Update(c.High, c.Low, c.Close); ready {
			im.batchWriter.AddADX(db.IndicatorADX{
				InstrumentToken: c.InstrumentToken, Interval: c.Interval,
				Period: cfg.ADX.Period, Timestamp: ts,
				ADXValue: adxVal, PlusDI: plusDI, MinusDI: minusDI, DataSource: src,
			})
			wsPayload["adx"] = map[string]interface{}{"adx": adxVal, "plus_di": plusDI, "minus_di": minusDI}
			atomic.AddUint64(&im.indicatorsProcessed, 1)
		}
	}

	// Single consolidated WS broadcast per candle — one marshal, one send per client.
	if len(wsPayload) > 0 {
		im.broadcastIndicators(c.InstrumentToken, c.Interval, ts, wsPayload)
	}
}

// broadcastIndicators marshals one envelope and fans it out to all WS clients.
func (im *IndicatorManager) broadcastIndicators(token uint32, interval string, ts time.Time, payload map[string]interface{}) {
	buf := jsonBufPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer jsonBufPool.Put(buf)

	envelope := map[string]interface{}{
		"type":            "INDICATOR_UPDATE",
		"instrumentToken": token,
		"interval":        interval,
		"timestamp":       ts,
		"indicators":      payload,
	}
	if err := json.NewEncoder(buf).Encode(envelope); err != nil {
		zap.L().Error("indicator broadcast: marshal failed", zap.Error(err))
		return
	}
	// Copy once — all clients share the same immutable bytes.
	msg := make([]byte, buf.Len())
	copy(msg, buf.Bytes())

	im.indicatorWsClients.Range(func(_, value interface{}) bool {
		client, ok := value.(*wsClient)
		if !ok {
			return true
		}
		select {
		case client.send <- msg:
		default:
			atomic.AddUint64(&im.wsDrops, 1)
		}
		return true
	})
}

// -------------------------------------------------------------------
// State factory
// -------------------------------------------------------------------

func (im *IndicatorManager) newStateSet() *indicators.IndicatorStateSet {
	cfg := im.indicatorsCfg
	return &indicators.IndicatorStateSet{
		SMA:        indicators.NewSMAState(cfg.SMA.Period),
		EMA:        indicators.NewEMAState(cfg.EMA.ShortPeriod),
		RSI:        indicators.NewRSIState(cfg.RSI.Period),
		MACD:       indicators.NewMACDState(cfg.MACD.FastPeriod, cfg.MACD.SlowPeriod, cfg.MACD.SignalPeriod),
		ATR:        indicators.NewATRState(cfg.ATR.Period),
		Bollinger:  indicators.NewBollingerState(cfg.BollingerBands.Period, cfg.BollingerBands.NumStdDev),
		Stochastic: indicators.NewStochasticState(cfg.Stochastic.KPeriod, cfg.Stochastic.DPeriod),
		OBV:        indicators.NewOBVState(),
		VWAP:       indicators.NewVWAPState(),
		ADX:        indicators.NewADXState(cfg.ADX.Period),
	}
}

// -------------------------------------------------------------------
// Warm-up from historical candles (runs once at startup)
// -------------------------------------------------------------------

// warmUpFromHistory replays DB candles through stateful indicators so they are
// ready to emit values from the very first live candle after restart.
func (im *IndicatorManager) warmUpFromHistory() {
	type row struct {
		InstrumentToken uint32    `gorm:"column:instrument_token"`
		Interval        string    `gorm:"column:interval"`
		Timestamp       time.Time `gorm:"column:timestamp"`
		Open            float64   `gorm:"column:open"`
		High            float64   `gorm:"column:high"`
		Low             float64   `gorm:"column:low"`
		Close           float64   `gorm:"column:close"`
		Volume          float64   `gorm:"column:volume"`
	}

	// Max candles needed across all indicators; 2*ADX.Period is the worst case.
	maxNeeded := 2*im.indicatorsCfg.ADX.Period + 10

	for _, interval := range im.appCfg.Candles.Intervals {
		var rows []row
		err := im.dbClient.DB.Raw(`
			SELECT instrument_token, interval, timestamp, open, high, low, close, volume
			FROM (
				SELECT *, ROW_NUMBER() OVER (
					PARTITION BY instrument_token
					ORDER BY timestamp DESC
				) AS rn
				FROM ohlcv_candles
				WHERE interval = ?
			) sub
			WHERE rn <= ?
			ORDER BY instrument_token, timestamp ASC
		`, interval, maxNeeded).Scan(&rows).Error
		if err != nil {
			zap.L().Error("warmup: failed to load history", zap.String("interval", interval), zap.Error(err))
			continue
		}

		loaded := 0
		for _, r := range rows {
			candle := indicators.Candle{
				InstrumentToken: r.InstrumentToken,
				Interval:        r.Interval,
				Timestamp:       r.Timestamp,
				Open:            r.Open,
				High:            r.High,
				Low:             r.Low,
				Close:           r.Close,
				Volume:          r.Volume,
			}
			key := stateKey{token: r.InstrumentToken, interval: r.Interval}
			im.stateMu.Lock()
			if _, exists := im.states[key]; !exists {
				im.states[key] = im.newStateSet()
			}
			s := im.states[key]
			im.stateMu.Unlock()

			// Feed candle into states silently (no DB write, no broadcast).
			im.feedStateOnly(s, candle)
			loaded++
		}
		zap.L().Info("warmup complete", zap.String("interval", interval), zap.Int("candles_fed", loaded))
	}
}

// feedStateOnly advances all state machines without triggering any output.
func (im *IndicatorManager) feedStateOnly(s *indicators.IndicatorStateSet, c indicators.Candle) {
	cfg := im.indicatorsCfg
	if cfg.SMA.Enabled {
		s.SMA.Update(c.Close) //nolint:errcheck
	}
	if cfg.EMA.Enabled {
		s.EMA.Update(c.Close)
	}
	if cfg.RSI.Enabled {
		s.RSI.Update(c.Close)
	}
	if cfg.MACD.Enabled {
		s.MACD.Update(c.Close)
	}
	if cfg.ATR.Enabled {
		s.ATR.Update(c.High, c.Low, c.Close)
	}
	if cfg.BollingerBands.Enabled {
		s.Bollinger.Update(c.Close)
	}
	if cfg.Stochastic.Enabled {
		s.Stochastic.Update(c.High, c.Low, c.Close)
	}
	if cfg.OBV.Enabled {
		s.OBV.Update(c.Close, c.Volume)
	}
	if cfg.VWAP.Enabled {
		s.VWAP.Update(c.Open, c.High, c.Low, c.Close, c.Volume)
	}
	if cfg.ADX.Enabled {
		s.ADX.Update(c.High, c.Low, c.Close)
	}
}

// -------------------------------------------------------------------
// Monitoring
// -------------------------------------------------------------------

func (im *IndicatorManager) startMonitoring(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			processed := atomic.SwapUint64(&im.indicatorsProcessed, 0)
			dbErrs := atomic.SwapUint64(&im.dbErrors, 0)
			wsDrops := atomic.SwapUint64(&im.wsDrops, 0)
			observability.IndicatorQueueDepth.Set(float64(im.batchWriter.PendingCount()))
			zap.L().Info("📊 IndicatorManager",
				zap.Uint64("indicators_processed_5s", processed),
				zap.Uint64("db_errors", dbErrs),
				zap.Uint64("ws_drops", wsDrops),
				zap.Int("batch_pending", im.batchWriter.PendingCount()),
			)
		case <-ctx.Done():
			return
		}
	}
}

// -------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------

func dataSource(appCfg *utils.AppConfig) string {
	if appCfg.Market.Simulate {
		return "simulation"
	}
	return "live"
}
