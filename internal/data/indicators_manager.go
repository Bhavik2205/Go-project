package data

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"

	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/indicators"
	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"github.com/gorilla/websocket"
	"go.uber.org/zap"
	"gorm.io/gorm/clause"
)

// CandleHistory holds a series of candles for a specific instrument and interval.
// It includes a mutex to protect concurrent access to the Candles slice.
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
	inputCandleCh      <-chan indicators.Candle // Channel to receive completed candles
	indicatorWsClients *sync.Map                // WebSocket clients for broadcasting indicators

	// Stores historical candles needed for indicator calculations
	// map[instrumentToken][interval]*CandleHistory
	candleHistory map[uint32]map[string]*CandleHistory
	historyMu     sync.RWMutex // Protects the top-level candleHistory map

	// Pre-calculated max history needed for each interval based on indicator periods
	maxHistoryPeriods map[string]int // map[interval]max_period_needed_for_any_indicator

	// New channel for processed indicators to be sent for persistence and broadcast
	processedIndicatorCh chan indicators.IndicatorResult
	outputWorkerCount    int // Number of goroutines to handle output processing
}

// NewIndicatorManager creates and returns a new instance of IndicatorManager.
// It initializes data structures and pre-calculates the maximum candle history required.
func NewIndicatorManager(
	dbC *db.DBClient,
	appCfg *utils.AppConfig,
	indicatorsCfg *utils.IndicatorsConfig,
	inputCandleCh <-chan indicators.Candle,
	wsClients *sync.Map,
) *IndicatorManager {
	im := &IndicatorManager{
		dbClient:           dbC,
		appCfg:             appCfg,
		indicatorsCfg:      indicatorsCfg,
		inputCandleCh:      inputCandleCh,
		indicatorWsClients: wsClients,
		candleHistory:      make(map[uint32]map[string]*CandleHistory),
		maxHistoryPeriods:  make(map[string]int),
		// Use a buffered channel to prevent blocking if output processing is slower than calculation
		processedIndicatorCh: make(chan indicators.IndicatorResult, 1000), // Buffer size can be tuned
		outputWorkerCount:    5,                                           // Tune based on typical I/O latency and CPU cores
	}

	// Pre-calculate max history needed for each interval
	for _, interval := range appCfg.Candles.Intervals {
		maxPeriod := 0
		// Iterate over all enabled indicators to find the maximum required history
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

		// Add a buffer to ensure enough data
		im.maxHistoryPeriods[interval] = maxPeriod + 2 // Add a small buffer
		zap.L().Info("Configured max history for indicator calculations",
			zap.String("interval", interval),
			zap.Int("max_candles", im.maxHistoryPeriods[interval]))
	}

	return im
}

// StartIndicatorCalculations listens for incoming candles and processes them
// for indicator calculation. This function should be run as a goroutine.
func (im *IndicatorManager) StartIndicatorCalculations(ctx context.Context) {
	zap.L().Info("✅ Indicator manager started, listening for new candles...")

	// Start the output processing workers
	im.StartOutputProcessing(ctx)

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
			// Close the processedIndicatorCh to signal output workers to finish
			close(im.processedIndicatorCh)
			return
		}
	}
}

// StartOutputProcessing starts a pool of goroutines to handle database persistence and WebSocket broadcasting.
func (im *IndicatorManager) StartOutputProcessing(ctx context.Context) {
	for i := 0; i < im.outputWorkerCount; i++ {
		go func(workerID int) {
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
// This function is now called by worker goroutines.
func (im *IndicatorManager) handleOutput(indicator indicators.IndicatorResult) {
	// 1. Save to Database
	// Using the IndicatorResult interface, GORM should be able to save it if the underlying struct
	// has the necessary gorm tags (which they do, via CommonIndicatorResult).
	// OnConflict is used to update existing entries if the indicator for the same token, interval, timestamp, and name already exists.
	var err error
	err = im.dbClient.DB.Clauses(clause.OnConflict{UpdateAll: true}).Create(indicator).Error
	if err != nil {
		zap.L().Error("Failed to save indicator to database",
			zap.Error(err),
			zap.Uint32("instrument_token", indicator.GetInstrumentToken()),
			zap.String("interval", indicator.GetInterval()),
			zap.String("indicator_name", indicator.GetIndicatorName()))
		// IMPORTANT CHANGE: Removed 'return' here.
		// We will now proceed to broadcast even if DB save fails.
	}

	// 2. Broadcast via WebSocket
	message, marshalErr := json.Marshal(map[string]interface{}{
		"type":            "INDICATOR_UPDATE",
		"instrumentToken": indicator.GetInstrumentToken(),
		"interval":        indicator.GetInterval(),
		"timestamp":       indicator.GetTimestamp(),
		"indicator":       indicator, // This will now include the IndicatorName field
	})
	if marshalErr != nil {
		zap.L().Error("Failed to marshal indicator for WebSocket broadcast", zap.Error(marshalErr),
			zap.Uint32("token", indicator.GetInstrumentToken()), zap.String("interval", indicator.GetInterval()), zap.Any("indicator", indicator))
		return // Still return if marshaling for broadcast fails, as we can't send malformed data.
	}

	// Broadcast to all connected WebSocket clients
	// Note: Iterating a sync.Map can be a bottleneck with a very large number of clients.
	// For extreme scale, consider a dedicated WebSocket fan-out service or more advanced patterns.
	im.indicatorWsClients.Range(func(key, value interface{}) bool {
		conn, ok := value.(*websocket.Conn)
		if !ok {
			zap.L().Warn("Found non-websocket.Conn in indicatorWsClients map, deleting.", zap.Any("key", key))
			im.indicatorWsClients.Delete(key)
			return true
		}

		// Non-blocking write to WebSocket, but actual network I/O might still block briefly
		err := conn.WriteMessage(websocket.TextMessage, message)
		if err != nil {
			zap.L().Error("Failed to write indicator message to WebSocket client, removing client",
				zap.Error(err),
				zap.String("remote_addr", conn.RemoteAddr().String()),
				zap.Uint32("token", indicator.GetInstrumentToken()))
			im.indicatorWsClients.Delete(key)
		}
		return true
	})

	// Consolidated log for successful handling (save and broadcast)
	// Log will now indicate if save failed but broadcast proceeded.
	if err == nil { // Check 'err' from DB save
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
			zap.Error(err)) // Include the DB error in the broadcast log
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
		// Pass a copy of the slice header to prevent concurrent modification issues
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
// for a given interval.
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

				// Type assertion and sending to the processedIndicatorCh
				// We expect the Calculate method to return a slice of the concrete indicator struct,
				// from which we take the latest result.
				switch v := results.(type) {
				case []indicators.SMA:
					if len(v) > 0 && !math.IsNaN(v[len(v)-1].Value) {
						im.processedIndicatorCh <- v[len(v)-1]
					} else {
						zap.L().Warn(fmt.Sprintf("Latest %s value is NaN or no results", indicatorType.GetName()), zap.Uint32("token", token), zap.String("interval", interval))
					}
				case []indicators.EMA:
					if len(v) > 0 && !math.IsNaN(v[len(v)-1].Value) {
						im.processedIndicatorCh <- v[len(v)-1]
					} else {
						zap.L().Warn(fmt.Sprintf("Latest %s value is NaN or no results", indicatorType.GetName()), zap.Uint32("token", token), zap.String("interval", interval))
					}
				case []indicators.MACD:
					if len(v) > 0 {
						latest := v[len(v)-1]
						if !math.IsNaN(latest.MACDLine) && !math.IsNaN(latest.SignalLine) && !math.IsNaN(latest.Histogram) {
							im.processedIndicatorCh <- latest
						} else {
							zap.L().Warn(fmt.Sprintf("Latest %s value is NaN or no results", indicatorType.GetName()), zap.Uint32("token", token), zap.String("interval", interval))
						}
					}
				case []indicators.ATR:
					if len(v) > 0 && !math.IsNaN(v[len(v)-1].Value) {
						im.processedIndicatorCh <- v[len(v)-1]
					} else {
						zap.L().Warn(fmt.Sprintf("Latest %s value is NaN or no results", indicatorType.GetName()), zap.Uint32("token", token), zap.String("interval", interval))
					}
				case []indicators.RSI:
					if len(v) > 0 && !math.IsNaN(v[len(v)-1].Value) {
						im.processedIndicatorCh <- v[len(v)-1]
					} else {
						zap.L().Warn(fmt.Sprintf("Latest %s value is NaN or no results", indicatorType.GetName()), zap.Uint32("token", token), zap.String("interval", interval))
					}
				case []indicators.Stochastic:
					if len(v) > 0 {
						latest := v[len(v)-1]
						if !math.IsNaN(latest.KValue) && !math.IsNaN(latest.DValue) {
							im.processedIndicatorCh <- latest
						} else {
							zap.L().Warn(fmt.Sprintf("Latest %s value is NaN or no results", indicatorType.GetName()), zap.Uint32("token", token), zap.String("interval", interval))
						}
					}
				case []indicators.BollingerBands:
					if len(v) > 0 {
						latest := v[len(v)-1]
						if !math.IsNaN(latest.MiddleBand) && !math.IsNaN(latest.UpperBand) && !math.IsNaN(latest.LowerBand) {
							im.processedIndicatorCh <- latest
						} else {
							zap.L().Warn(fmt.Sprintf("Latest %s value is NaN or no results", indicatorType.GetName()), zap.Uint32("token", token), zap.String("interval", interval))
						}
					}
				case []indicators.OBV:
					if len(v) > 0 && !math.IsNaN(v[len(v)-1].Value) {
						im.processedIndicatorCh <- v[len(v)-1]
					} else {
						zap.L().Warn(fmt.Sprintf("Latest %s value is NaN or no results", indicatorType.GetName()), zap.Uint32("token", token), zap.String("interval", interval))
					}
				case []indicators.VWAP:
					if len(v) > 0 && !math.IsNaN(v[len(v)-1].Value) {
						im.processedIndicatorCh <- v[len(v)-1]
					} else {
						zap.L().Warn(fmt.Sprintf("Latest %s value is NaN or no results", indicatorType.GetName()), zap.Uint32("token", token), zap.String("interval", interval))
					}
				case []indicators.ADX:
					if len(v) > 0 {
						latest := v[len(v)-1]
						if !math.IsNaN(latest.ADXValue) {
							im.processedIndicatorCh <- latest
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
	wg.Wait() // Wait for all calculation goroutines to complete their work for this candle
}
