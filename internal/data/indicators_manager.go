package data

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/indicators"
	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"github.com/gorilla/websocket"
	"go.uber.org/zap"
	"gorm.io/gorm/clause"
)

type wsClient struct {
	conn *websocket.Conn
	send chan []byte
}

// CandleHistory holds a series of candles for a specific instrument and interval.
type CandleHistory struct {
	Candles []indicators.Candle
	mu      sync.Mutex
}

// IndicatorManager aggregates completed OHLCV candles, calculates technical indicators,
// persists them to the database, and broadcasts them via WebSocket.
type IndicatorManager struct {
	dbClient           *db.DBClient
	appCfg             *utils.AppConfig
	indicatorsCfg      *utils.IndicatorsConfig
	inputCandleCh      <-chan indicators.Candle
	indicatorWsClients *sync.Map

	candleHistory map[uint32]map[string]*CandleHistory
	historyMu     sync.RWMutex

	maxHistoryPeriods    map[string]int
	processedIndicatorCh chan indicators.IndicatorResult
	outputWorkerCount    int

	// Monitoring metrics
	indicatorsProcessed uint64
	dbErrors            uint64
	wsDrops             uint64
}

// NewIndicatorManager creates and returns a new instance of IndicatorManager.
func NewIndicatorManager(
	dbC *db.DBClient,
	appCfg *utils.AppConfig,
	indicatorsCfg *utils.IndicatorsConfig,
	inputCandleCh <-chan indicators.Candle,
	wsClients *sync.Map,
) *IndicatorManager {
	// Use configurable buffer size (default 5000 if not set)
	bufferSize := indicatorsCfg.OutputChannelBufferSize
	if bufferSize <= 0 {
		bufferSize = 5000
	}

	im := &IndicatorManager{
		dbClient:             dbC,
		appCfg:               appCfg,
		indicatorsCfg:        indicatorsCfg,
		inputCandleCh:        inputCandleCh,
		indicatorWsClients:   wsClients,
		candleHistory:        make(map[uint32]map[string]*CandleHistory),
		maxHistoryPeriods:    make(map[string]int),
		processedIndicatorCh: make(chan indicators.IndicatorResult, bufferSize),
		outputWorkerCount:    indicatorsCfg.OutputWorkerCount,
	}
	// Ensure a sane default if config value is missing or zero
	if im.outputWorkerCount <= 0 {
		im.outputWorkerCount = 30
	}

	for _, interval := range appCfg.Candles.Intervals {
		maxPeriod := 0
		allIndicators := []indicators.Indicator{
			indicators.SMA{},
			indicators.EMA{},
			indicators.MACD{},
			indicators.ATR{},
			indicators.RSI{},
			indicators.Stochastic{},
			indicators.BollingerBands{},
			indicators.OBV{},
			indicators.VWAP{},
			indicators.ADX{},
		}
		for _, ind := range allIndicators {
			if ind.IsEnabled(im.indicatorsCfg) {
				required := ind.GetMinRequiredCandles(im.indicatorsCfg)
				if required > maxPeriod {
					maxPeriod = required
				}
			}
		}
		im.maxHistoryPeriods[interval] = maxPeriod + 2
		zap.L().Info("Configured max history for indicator calculations",
			zap.String("interval", interval),
			zap.Int("max_candles", im.maxHistoryPeriods[interval]))
	}
	return im
}

// writePump writes messages from the channel to the WebSocket with deadlines and periodic pings.
func (im *IndicatorManager) writePump(client *wsClient) {
	defer func() {
		if r := recover(); r != nil {
			zap.L().Error("Panic in indicator writePump", zap.Any("recover", r))
		}
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
				_ = client.conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
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

// RegisterWebSocketClient should be called when a new client connects.
func (im *IndicatorManager) RegisterWebSocketClient(conn *websocket.Conn) {
	client := &wsClient{
		conn: conn,
		send: make(chan []byte, 256),
	}
	im.indicatorWsClients.Store(conn, client)
	go im.writePump(client)
	zap.L().Info("Indicator WebSocket client registered", zap.String("remote_addr", conn.RemoteAddr().String()))
}

// UnregisterWebSocketClient closes and removes a WebSocket client.
func (im *IndicatorManager) UnregisterWebSocketClient(conn *websocket.Conn) {
	if val, ok := im.indicatorWsClients.LoadAndDelete(conn); ok {
		client := val.(*wsClient)
		close(client.send)
	}
	conn.Close()
	zap.L().Info("Indicator WebSocket client unregistered", zap.String("remote_addr", conn.RemoteAddr().String()))
}

// StartIndicatorCalculations listens for incoming candles and processes them
// for indicator calculation. This function should be run as a goroutine.
func (im *IndicatorManager) StartIndicatorCalculations(ctx context.Context) {
	zap.L().Info("✅ Indicator manager started, listening for new candles...")

	// Load historical candles from DB so indicators can calculate immediately on restart
	im.loadHistoricalCandles()

	// Start the output processing workers
	im.StartOutputProcessing(ctx)
	// Start monitoring goroutine
	go im.startMonitoring(ctx)

	for {
		select {
		case candle, ok := <-im.inputCandleCh:
			if !ok {
				zap.L().Error("Indicator input candle channel closed unexpectedly. Stopping indicator manager.")
				return
			}
			im.processCandle(candle)
		case <-ctx.Done():
			zap.L().Info("Context cancelled, stopping indicator manager.")
			close(im.processedIndicatorCh)
			return
		}
	}
}

// Monitoring goroutine for processed indicators, DB errors, and WS drops.
func (im *IndicatorManager) startMonitoring(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			processed := atomic.SwapUint64(&im.indicatorsProcessed, 0)
			dbErrs := atomic.SwapUint64(&im.dbErrors, 0)
			wsDrops := atomic.SwapUint64(&im.wsDrops, 0)
			zap.L().Info("📊 IndicatorManager monitoring",
				zap.Uint64("indicators_processed", processed),
				zap.Uint64("db_errors", dbErrs),
				zap.Uint64("ws_drops", wsDrops),
			)
		case <-ctx.Done():
			return
		}
	}
}

// StartOutputProcessing starts a pool of goroutines to handle database persistence and WebSocket broadcasting.
func (im *IndicatorManager) StartOutputProcessing(ctx context.Context) {
	for i := 0; i < im.outputWorkerCount; i++ {
		go func(workerID int) {
			defer func() {
				if r := recover(); r != nil {
					zap.L().Error("Panic in indicator output worker",
						zap.Int("worker_id", workerID), zap.Any("recover", r))
				}
			}()
			zap.L().Info("📦 Indicator output worker started", zap.Int("worker_id", workerID))
			for {
				select {
				case indicatorResult, ok := <-im.processedIndicatorCh:
					if !ok {
						zap.L().Info("📦 Indicator output channel closed, worker stopping", zap.Int("worker_id", workerID))
						return
					}
					im.handleOutput(indicatorResult)
				case <-ctx.Done():
					zap.L().Info("📦 Context cancelled, worker stopping", zap.Int("worker_id", workerID))
					return
				}
			}
		}(i)
	}
}

// handleOutput saves the calculated indicator to the database and broadcasts it via WebSocket.
func (im *IndicatorManager) handleOutput(indicator indicators.IndicatorResult) {
	var conflictColumns []clause.Column
	indicatorName := indicator.GetIndicatorName()
	conflictColumns = append(conflictColumns,
		clause.Column{Name: "instrument_token"},
		clause.Column{Name: "interval"},
		clause.Column{Name: "timestamp"},
	)
	switch indicatorName {
	case "SMA", "EMA", "ATR", "RSI", "ADX":
		conflictColumns = append(conflictColumns, clause.Column{Name: "period"})
	case "MACD":
		conflictColumns = append(conflictColumns,
			clause.Column{Name: "fast_period"},
			clause.Column{Name: "slow_period"},
			clause.Column{Name: "signal_period"},
		)
	case "Stochastic":
		conflictColumns = append(conflictColumns,
			clause.Column{Name: "k_period"},
			clause.Column{Name: "d_period"},
		)
	case "BollingerBands":
		conflictColumns = append(conflictColumns,
			clause.Column{Name: "period"},
			clause.Column{Name: "num_std_dev"},
		)
	case "OBV", "VWAP":
		// Only common columns
	default:
		zap.L().Warn("Unknown indicator name encountered for conflict resolution. Saving with common primary keys only.",
			zap.String("indicator_name", indicatorName),
			zap.Uint32("instrument_token", indicator.GetInstrumentToken()),
			zap.String("interval", indicator.GetInterval()),
		)
	}

	// 1. Save to Database
	var err error
	err = im.dbClient.DB.Clauses(clause.OnConflict{
		Columns:   conflictColumns,
		DoUpdates: clause.AssignmentColumns(db.GetUpdatableIndicatorColumns(indicatorName)),
	}).Create(indicator).Error

	if err != nil {
		atomic.AddUint64(&im.dbErrors, 1)
		zap.L().Error("Failed to save indicator to database",
			zap.Error(err),
			zap.Uint32("instrument_token", indicator.GetInstrumentToken()),
			zap.String("interval", indicator.GetInterval()),
			zap.String("indicator_name", indicator.GetIndicatorName()))
	}

	// 2. Broadcast via WebSocket
	message, marshalErr := json.Marshal(map[string]interface{}{
		"type":            "INDICATOR_UPDATE",
		"instrumentToken": indicator.GetInstrumentToken(),
		"interval":        indicator.GetInterval(),
		"timestamp":       indicator.GetTimestamp(),
		"indicator":       indicator,
	})
	if marshalErr != nil {
		zap.L().Error("Failed to marshal indicator for WebSocket broadcast", zap.Error(marshalErr),
			zap.Uint32("token", indicator.GetInstrumentToken()), zap.String("interval", indicator.GetInterval()), zap.Any("indicator", indicator))
		return
	}

	im.indicatorWsClients.Range(func(key, value interface{}) bool {
		client, ok := value.(*wsClient)
		if !ok {
			zap.L().Warn("Found non-wsClient in indicatorWsClients map, deleting.", zap.Any("key", key))
			im.indicatorWsClients.Delete(key)
			return true
		}
		select {
		case client.send <- message:
		default:
			atomic.AddUint64(&im.wsDrops, 1)
			zap.L().Warn("WebSocket send channel full, dropping indicator message")
		}
		return true
	})

	atomic.AddUint64(&im.indicatorsProcessed, 1)

	if err == nil {
		zap.L().Info("✅ Indicator processed (saved and broadcasted)",
			zap.Uint32("instrument_token", indicator.GetInstrumentToken()),
			zap.String("interval", indicator.GetInterval()),
			zap.String("indicator_name", indicator.GetIndicatorName()),
			zap.Time("timestamp", indicator.GetTimestamp()))
	} else {
		zap.L().Warn("⚠️ Indicator processed (DB save failed, but broadcasted)",
			zap.Uint32("instrument_token", indicator.GetInstrumentToken()),
			zap.String("interval", indicator.GetInterval()),
			zap.String("indicator_name", indicator.GetIndicatorName()),
			zap.Time("timestamp", indicator.GetTimestamp()),
			zap.Error(err))
	}
}

// processCandle adds the new candle to history and triggers indicator calculations.
func (im *IndicatorManager) processCandle(newCandle indicators.Candle) {
	zap.L().Debug("Candle received by IndicatorManager",
		zap.Uint32("token", newCandle.InstrumentToken),
		zap.String("interval", newCandle.Interval),
		zap.Time("timestamp", newCandle.Timestamp))

	im.historyMu.Lock()
	defer im.historyMu.Unlock()

	instrumentHistory, ok := im.candleHistory[newCandle.InstrumentToken]
	if !ok {
		instrumentHistory = make(map[string]*CandleHistory)
		im.candleHistory[newCandle.InstrumentToken] = instrumentHistory
	}

	candleSeries, ok := instrumentHistory[newCandle.Interval]
	if !ok {
		candleSeries = &CandleHistory{Candles: make([]indicators.Candle, 0)}
		instrumentHistory[newCandle.Interval] = candleSeries
	}

	candleSeries.mu.Lock()
	defer candleSeries.mu.Unlock()

	candleSeries.Candles = append(candleSeries.Candles, newCandle)
	zap.L().Debug("Candle added to history for indicator calculation",
		zap.Uint32("token", newCandle.InstrumentToken),
		zap.String("interval", newCandle.Interval),
		zap.Int("history_len", len(candleSeries.Candles)))

	maxLen := im.maxHistoryPeriods[newCandle.Interval] + 100
	if len(candleSeries.Candles) > maxLen {
		candleSeries.Candles = candleSeries.Candles[len(candleSeries.Candles)-maxLen:]
	}

	if len(candleSeries.Candles) >= im.getMinRequiredCandles(newCandle.Interval) {
		candlesCopy := make([]indicators.Candle, len(candleSeries.Candles))
		copy(candlesCopy, candleSeries.Candles)
		im.calculateAndStoreAllIndicators(newCandle.InstrumentToken, newCandle.Interval, candlesCopy)
	} else {
		zap.L().Debug("Not enough candles for indicator calculation yet",
			zap.Uint32("token", newCandle.InstrumentToken),
			zap.String("interval", newCandle.Interval),
			zap.Int("current_len", len(candleSeries.Candles)),
			zap.Int("required_len", im.getMinRequiredCandles(newCandle.Interval)))
	}
}

// getMinRequiredCandles determines the minimum number of candles required to calculate all indicators
func (im *IndicatorManager) getMinRequiredCandles(interval string) int {
	return im.maxHistoryPeriods[interval]
}

// calculateAndStoreAllIndicators calculates all configured indicators concurrently for a given
// instrument and interval using the provided historical candles, then sends results to the output channel.
func (im *IndicatorManager) calculateAndStoreAllIndicators(token uint32, interval string, candles []indicators.Candle) {
	var wg sync.WaitGroup

	allIndicators := []indicators.Indicator{
		indicators.SMA{},
		indicators.EMA{},
		indicators.MACD{},
		indicators.ATR{},
		indicators.RSI{},
		indicators.Stochastic{},
		indicators.BollingerBands{},
		indicators.OBV{},
		indicators.VWAP{},
		indicators.ADX{},
	}

	for _, ind := range allIndicators {
		if ind.IsEnabled(im.indicatorsCfg) {
			wg.Add(1)
			go func(indicatorType indicators.Indicator) {
				defer wg.Done()
				defer func() {
					if r := recover(); r != nil {
						zap.L().Error("Panic in indicator calculation",
							zap.String("indicator", indicatorType.GetName()),
							zap.Any("recover", r))
					}
				}()
				minHistory := indicatorType.GetMinRequiredCandles(im.indicatorsCfg)
				if len(candles) < minHistory {
					zap.L().Debug("Not enough candles for indicator calculation yet",
						zap.String("indicator", indicatorType.GetName()),
						zap.Uint32("token", token), zap.String("interval", interval),
						zap.Int("current_len", len(candles)), zap.Int("required_len", minHistory))
					return
				}
				results, err := indicatorType.Calculate(candles, im.appCfg, im.indicatorsCfg)
				if err != nil {
					zap.L().Error(fmt.Sprintf("%s calculation failed", indicatorType.GetName()), zap.Error(err), zap.Uint32("token", token), zap.String("interval", interval))
					return
				}
				switch v := results.(type) {
				case []indicators.SMA:
					if len(v) > 0 && !math.IsNaN(v[len(v)-1].Value) {
						select {
						case im.processedIndicatorCh <- v[len(v)-1]:
						default:
							atomic.AddUint64(&im.wsDrops, 1)
						}
					} else {
						zap.L().Warn(fmt.Sprintf("Latest %s value is NaN or no results", indicatorType.GetName()), zap.Uint32("token", token), zap.String("interval", interval))
					}
				case []indicators.EMA:
					if len(v) > 0 && !math.IsNaN(v[len(v)-1].Value) {
						select {
						case im.processedIndicatorCh <- v[len(v)-1]:
						default:
							atomic.AddUint64(&im.wsDrops, 1)
						}
					} else {
						zap.L().Warn(fmt.Sprintf("Latest %s value is NaN or no results", indicatorType.GetName()), zap.Uint32("token", token), zap.String("interval", interval))
					}
				case []indicators.MACD:
					if len(v) > 0 {
						latest := v[len(v)-1]
						if !math.IsNaN(latest.MACDLine) && !math.IsNaN(latest.SignalLine) && !math.IsNaN(latest.Histogram) {
							select {
							case im.processedIndicatorCh <- latest:
							default:
								atomic.AddUint64(&im.wsDrops, 1)
							}
						} else {
							zap.L().Warn(fmt.Sprintf("Latest %s value is NaN or no results", indicatorType.GetName()), zap.Uint32("token", token), zap.String("interval", interval))
						}
					}
				case []indicators.ATR:
					if len(v) > 0 && !math.IsNaN(v[len(v)-1].Value) {
						select {
						case im.processedIndicatorCh <- v[len(v)-1]:
						default:
							atomic.AddUint64(&im.wsDrops, 1)
						}
					} else {
						zap.L().Warn(fmt.Sprintf("Latest %s value is NaN or no results", indicatorType.GetName()), zap.Uint32("token", token), zap.String("interval", interval))
					}
				case []indicators.RSI:
					if len(v) > 0 && !math.IsNaN(v[len(v)-1].Value) {
						select {
						case im.processedIndicatorCh <- v[len(v)-1]:
						default:
							atomic.AddUint64(&im.wsDrops, 1)
						}
					} else {
						zap.L().Warn(fmt.Sprintf("Latest %s value is NaN or no results", indicatorType.GetName()), zap.Uint32("token", token), zap.String("interval", interval))
					}
				case []indicators.Stochastic:
					if len(v) > 0 {
						latest := v[len(v)-1]
						if !math.IsNaN(latest.KValue) && !math.IsNaN(latest.DValue) {
							select {
							case im.processedIndicatorCh <- latest:
							default:
								atomic.AddUint64(&im.wsDrops, 1)
							}
						} else {
							zap.L().Warn(fmt.Sprintf("Latest %s value is NaN or no results", indicatorType.GetName()), zap.Uint32("token", token), zap.String("interval", interval))
						}
					}
				case []indicators.BollingerBands:
					if len(v) > 0 {
						latest := v[len(v)-1]
						if !math.IsNaN(latest.MiddleBand) && !math.IsNaN(latest.UpperBand) && !math.IsNaN(latest.LowerBand) {
							select {
							case im.processedIndicatorCh <- latest:
							default:
								atomic.AddUint64(&im.wsDrops, 1)
							}
						} else {
							zap.L().Warn(fmt.Sprintf("Latest %s value is NaN or no results", indicatorType.GetName()), zap.Uint32("token", token), zap.String("interval", interval))
						}
					}
				case []indicators.OBV:
					if len(v) > 0 && !math.IsNaN(v[len(v)-1].Value) {
						select {
						case im.processedIndicatorCh <- v[len(v)-1]:
						default:
							atomic.AddUint64(&im.wsDrops, 1)
						}
					} else {
						zap.L().Warn(fmt.Sprintf("Latest %s value is NaN or no results", indicatorType.GetName()), zap.Uint32("token", token), zap.String("interval", interval))
					}
				case []indicators.VWAP:
					if len(v) > 0 && !math.IsNaN(v[len(v)-1].Value) {
						select {
						case im.processedIndicatorCh <- v[len(v)-1]:
						default:
							atomic.AddUint64(&im.wsDrops, 1)
						}
					} else {
						zap.L().Warn(fmt.Sprintf("Latest %s value is NaN or no results", indicatorType.GetName()), zap.Uint32("token", token), zap.String("interval", interval))
					}
				case []indicators.ADX:
					if len(v) > 0 {
						latest := v[len(v)-1]
						if !math.IsNaN(latest.ADXValue) {
							select {
							case im.processedIndicatorCh <- latest:
							default:
								atomic.AddUint64(&im.wsDrops, 1)
							}
						} else {
							zap.L().Warn(fmt.Sprintf("Latest %s value is NaN or no results", indicatorType.GetName()), zap.Uint32("token", token), zap.String("interval", interval))
						}
					}
				default:
					zap.L().Error("Unknown indicator result type", zap.Any("result_type", fmt.Sprintf("%T", results)), zap.String("indicator", indicatorType.GetName()), zap.Uint32("token", token))
				}
			}(ind)
		}
	}
	wg.Wait()
}

func (im *IndicatorManager) recoverGoroutine(where string) {
	if r := recover(); r != nil {
		zap.L().Error("Panic recovered", zap.String("where", where), zap.Any("recover", r))
	}
}

// loadHistoricalCandles loads existing candles from ohlcv_candles into memory
// so indicators can be calculated immediately on restart without waiting for
// new candles to accumulate.
func (im *IndicatorManager) loadHistoricalCandles() {
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

	// For each configured interval, load the last N candles needed for indicators
	for _, interval := range im.appCfg.Candles.Intervals {
		maxNeeded := im.maxHistoryPeriods[interval] + 100
		var rows []row
		err := im.dbClient.DB.Raw(`
			SELECT instrument_token, interval, timestamp, open, high, low, close, volume
			FROM ohlcv_candles
			WHERE interval = ?
			ORDER BY instrument_token, timestamp ASC
		`, interval).Scan(&rows).Error
		if err != nil {
			zap.L().Error("Failed to load historical candles from DB",
				zap.String("interval", interval), zap.Error(err))
			continue
		}

		// Group by instrument token
		byToken := make(map[uint32][]row)
		for _, r := range rows {
			byToken[r.InstrumentToken] = append(byToken[r.InstrumentToken], r)
		}

		loaded := 0
		for token, tokenRows := range byToken {
			// Keep only the last maxNeeded candles
			if len(tokenRows) > maxNeeded {
				tokenRows = tokenRows[len(tokenRows)-maxNeeded:]
			}

			candles := make([]indicators.Candle, len(tokenRows))
			for i, r := range tokenRows {
				candles[i] = indicators.Candle{
					InstrumentToken: r.InstrumentToken,
					Interval:        r.Interval,
					Timestamp:       r.Timestamp,
					Open:            r.Open,
					High:            r.High,
					Low:             r.Low,
					Close:           r.Close,
					Volume:          r.Volume,
				}
			}

			im.historyMu.Lock()
			if _, ok := im.candleHistory[token]; !ok {
				im.candleHistory[token] = make(map[string]*CandleHistory)
			}
			im.candleHistory[token][interval] = &CandleHistory{Candles: candles}
			im.historyMu.Unlock()
			loaded++
		}

		zap.L().Info("Loaded historical candles for interval",
			zap.String("interval", interval),
			zap.Int("instruments", loaded),
			zap.Int("total_rows", len(rows)))
	}
}

// GetWebSocketClientCount returns the number of currently connected indicator WebSocket clients.
func (im *IndicatorManager) GetWebSocketClientCount() int {
	count := 0
	im.indicatorWsClients.Range(func(key, value interface{}) bool {
		count++
		return true
	})
	return count
}
