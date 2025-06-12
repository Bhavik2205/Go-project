package data

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/indicators" // Assuming this package provides the calculation functions
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
	}

	// Pre-calculate max history needed for each interval
	for _, interval := range appCfg.Candles.Intervals {
		maxPeriod := 0
		// Max period for SMA, EMA, ATR, RSI, ADX
		if im.indicatorsCfg.SMA.Period > maxPeriod {
			maxPeriod = im.indicatorsCfg.SMA.Period
		}
		if im.indicatorsCfg.EMA.LongPeriod > maxPeriod { // EMA uses long period
			maxPeriod = im.indicatorsCfg.EMA.LongPeriod
		}
		if im.indicatorsCfg.ATR.Period > maxPeriod {
			maxPeriod = im.indicatorsCfg.ATR.Period
		}
		if im.indicatorsCfg.RSI.Period > maxPeriod {
			maxPeriod = im.indicatorsCfg.RSI.Period
		}
		if im.indicatorsCfg.ADX.Period > maxPeriod {
			maxPeriod = im.indicatorsCfg.ADX.Period
		}

		// MACD involves EMAs, typically FastPeriod and SlowPeriod
		// A common rule of thumb for MACD (12, 26, 9) is to need about 26 (slow EMA) + 9 (signal EMA) -1 = 34 candles for the first complete MACD + Signal.
		macdEffectivePeriod := im.indicatorsCfg.MACD.SlowPeriod + im.indicatorsCfg.MACD.SignalPeriod - 1
		if macdEffectivePeriod > maxPeriod {
			maxPeriod = macdEffectivePeriod
		}

		// Stochastic requires K and D periods
		// Needs K period for %K, then D period for SMA of %K. So, KPeriod + DPeriod -1 is a safe minimum.
		stochEffectivePeriod := im.indicatorsCfg.Stochastic.KPeriod + im.indicatorsCfg.Stochastic.DPeriod - 1
		if stochEffectivePeriod > maxPeriod {
			maxPeriod = stochEffectivePeriod
		}

		// Bollinger Bands
		if im.indicatorsCfg.BollingerBands.Period > maxPeriod {
			maxPeriod = im.indicatorsCfg.BollingerBands.Period
		}

		// Add a buffer to ensure enough data
		// Some indicators like ADX might need a few more candles for initial smoothing.
		im.maxHistoryPeriods[interval] = maxPeriod + 5 // Add a small buffer
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

	for {
		select {
		case candle, ok := <-im.inputCandleCh:
			if !ok {
				zap.L().Error("Indicator input candle channel closed unexpectedly.")
				return
			}
			im.processCandle(candle)
		case <-ctx.Done():
			zap.L().Info("Context cancelled, stopping indicator manager.")
			// Optionally flush any pending calculations, though indicators are mostly per-candle
			return
		}
	}
}

// processCandle adds the new candle to history and triggers indicator calculations.
func (im *IndicatorManager) processCandle(newCandle indicators.Candle) {
	// ⭐ LOG: Confirms candle was received by IndicatorManager
	zap.L().Debug("Candle received by IndicatorManager",
		zap.Uint32("token", newCandle.InstrumentToken),
		zap.String("interval", newCandle.Interval),
		zap.Time("timestamp", newCandle.Timestamp))

	im.historyMu.Lock()
	defer im.historyMu.Unlock()

	// Get or create history for this instrument and interval
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

	// Append the new candle
	candleSeries.Candles = append(candleSeries.Candles, newCandle)
	// ⭐ LOG: Confirms candle was added to history
	zap.L().Debug("Candle added to history for indicator calculation",
		zap.Uint32("token", newCandle.InstrumentToken),
		zap.String("interval", newCandle.Interval),
		zap.Int("history_len", len(candleSeries.Candles)))

	// Trim history to required length
	// maxLen includes the actual calculation period + a buffer
	maxLen := im.maxHistoryPeriods[newCandle.Interval] + 100 // Add a larger buffer to ensure enough candles for all calculations
	if len(candleSeries.Candles) > maxLen {
		candleSeries.Candles = candleSeries.Candles[len(candleSeries.Candles)-maxLen:]
	}

	zap.L().Debug("Candle added to history for indicator calculation",
		zap.Uint32("token", newCandle.InstrumentToken),
		zap.String("interval", newCandle.Interval),
		zap.Time("timestamp", newCandle.Timestamp),
		zap.Int("history_len", len(candleSeries.Candles)))

	// Trigger calculations if enough historical data is available
	if len(candleSeries.Candles) >= im.getMinRequiredCandles(newCandle.Interval) {
		im.calculateAndStoreAllIndicators(newCandle.InstrumentToken, newCandle.Interval, candleSeries.Candles)
	} else {
		zap.L().Debug("Not enough candles for indicator calculation yet",
			zap.Uint32("token", newCandle.InstrumentToken),
			zap.String("interval", newCandle.Interval),
			zap.Int("current_len", len(candleSeries.Candles)),
			zap.Int("required_len", im.getMinRequiredCandles(newCandle.Interval)))
	}
}

// getMinRequiredCandles determines the minimum number of candles required to calculate all indicators
// for a given interval. This is usually related to the longest period used by any indicator.
func (im *IndicatorManager) getMinRequiredCandles(interval string) int {
	// A small buffer is often needed as some calculations require more than just the exact period length.
	// For example, an N-period EMA needs N candles to start, and often N+1 for the first "smoothed" value.
	// MACD requires 2 EMAs, then a signal line EMA.
	return im.maxHistoryPeriods[interval]
}

// calculateAndStoreAllIndicators calculates all configured indicators for a given instrument and interval
// using the provided historical candles, then stores and broadcasts them.
func (im *IndicatorManager) calculateAndStoreAllIndicators(token uint32, interval string, candles []indicators.Candle) {
	// Ensure we have at least one candle for the current timestamp
	if len(candles) == 0 {
		zap.L().Warn("No candles available for indicator calculation",
			zap.Uint32("token", token),
			zap.String("interval", interval))
		return
	}
	currentCandle := candles[len(candles)-1] // The most recent candle

	// --- SMA Calculation ---
	smaPeriod := im.indicatorsCfg.SMA.Period
	if len(candles) >= smaPeriod {
		smaValues, err := indicators.CalculateSMA(candles, smaPeriod)
		if err != nil {
			zap.L().Error("SMA calculation failed", zap.Error(err), zap.Uint32("token", token), zap.String("interval", interval))
		} else if len(smaValues) > 0 && !math.IsNaN(smaValues[len(smaValues)-1].Value) {
			sma := db.IndicatorSMA{
				InstrumentToken: token,
				Interval:        interval,
				Period:          smaPeriod,
				Timestamp:       currentCandle.Timestamp,
				Value:           smaValues[len(smaValues)-1].Value,
			}
			im.saveAndBroadcastIndicator(&sma)
		} else {
			zap.L().Warn("SMA calculation resulted in NaN or empty slice", zap.Uint32("token", token), zap.String("interval", interval))
		}
	} else {
		zap.L().Debug("Not enough candles for SMA calculation", zap.Uint32("token", token), zap.String("interval", interval), zap.Int("needed", smaPeriod), zap.Int("have", len(candles)))
	}

	// --- EMA Calculation ---
	emaLongPeriod := im.indicatorsCfg.EMA.LongPeriod // Use long period as the period for EMA struct
	if len(candles) >= emaLongPeriod {
		emaResults, err := indicators.CalculateEMA(candles, emaLongPeriod)
		if err != nil {
			zap.L().Error("EMA calculation failed", zap.Error(err), zap.Uint32("token", token), zap.String("interval", interval))
		} else if len(emaResults) > 0 && !math.IsNaN(emaResults[len(emaResults)-1].Value) {
			ema := db.IndicatorEMA{
				InstrumentToken: token,
				Interval:        interval,
				Period:          emaLongPeriod, // Store EMA for the long period
				Timestamp:       currentCandle.Timestamp,
				Value:           emaResults[len(emaResults)-1].Value,
			}
			im.saveAndBroadcastIndicator(&ema)
		} else {
			zap.L().Warn("EMA calculation resulted in NaN or empty slice", zap.Uint32("token", token), zap.String("interval", interval))
		}
	} else {
		zap.L().Debug("Not enough candles for EMA calculation", zap.Uint32("token", token), zap.String("interval", interval), zap.Int("needed", emaLongPeriod), zap.Int("have", len(candles)))
	}

	// --- MACD Calculation ---
	macdFastPeriod := im.indicatorsCfg.MACD.FastPeriod
	macdSlowPeriod := im.indicatorsCfg.MACD.SlowPeriod
	macdSignalPeriod := im.indicatorsCfg.MACD.SignalPeriod
	// MACD needs enough candles for its longest EMA + signal EMA.
	macdMinRequired := macdSlowPeriod + macdSignalPeriod - 1 // Adjusted minimum for first valid signal
	if len(candles) >= macdMinRequired {
		macdResults, err := indicators.CalculateMACD(candles, macdFastPeriod, macdSlowPeriod, macdSignalPeriod)
		if err != nil {
			zap.L().Error("MACD calculation failed", zap.Error(err), zap.Uint32("token", token), zap.String("interval", interval))
		} else if len(macdResults) > 0 &&
			!math.IsNaN(macdResults[len(macdResults)-1].MACDLine) &&
			!math.IsNaN(macdResults[len(macdResults)-1].SignalLine) &&
			!math.IsNaN(macdResults[len(macdResults)-1].Histogram) {
			lastMACD := macdResults[len(macdResults)-1]
			macd := db.IndicatorMACD{
				InstrumentToken: token,
				Interval:        interval,
				FastPeriod:      macdFastPeriod,
				SlowPeriod:      macdSlowPeriod,
				SignalPeriod:    macdSignalPeriod,
				Timestamp:       currentCandle.Timestamp,
				MACDLine:        lastMACD.MACDLine,
				SignalLine:      lastMACD.SignalLine,
				Histogram:       lastMACD.Histogram,
			}
			im.saveAndBroadcastIndicator(&macd)
		} else {
			zap.L().Warn("MACD calculation resulted in NaN or empty slice", zap.Uint32("token", token), zap.String("interval", interval))
		}
	} else {
		zap.L().Debug("Not enough candles for MACD calculation", zap.Uint32("token", token), zap.String("interval", interval), zap.Int("needed", macdMinRequired), zap.Int("have", len(candles)))
	}

	// --- ATR Calculation ---
	atrPeriod := im.indicatorsCfg.ATR.Period
	if len(candles) >= atrPeriod {
		atrResults, err := indicators.CalculateATR(candles, atrPeriod)
		if err != nil {
			zap.L().Error("ATR calculation failed", zap.Error(err), zap.Uint32("token", token), zap.String("interval", interval))
		} else if len(atrResults) > 0 && !math.IsNaN(atrResults[len(atrResults)-1].Value) {
			atr := db.IndicatorATR{
				InstrumentToken: token,
				Interval:        interval,
				Period:          atrPeriod,
				Timestamp:       currentCandle.Timestamp,
				Value:           atrResults[len(atrResults)-1].Value,
			}
			im.saveAndBroadcastIndicator(&atr)
		} else {
			zap.L().Warn("ATR calculation resulted in NaN or empty slice", zap.Uint32("token", token), zap.String("interval", interval))
		}
	} else {
		zap.L().Debug("Not enough candles for ATR calculation", zap.Uint32("token", token), zap.String("interval", interval), zap.Int("needed", atrPeriod), zap.Int("have", len(candles)))
	}

	// --- RSI Calculation ---
	rsiPeriod := im.indicatorsCfg.RSI.Period
	if len(candles) >= rsiPeriod+1 { // RSI needs at least period + 1 for initial average gain/loss
		rsiResults, err := indicators.CalculateRSI(candles, rsiPeriod)
		if err != nil {
			zap.L().Error("RSI calculation failed", zap.Error(err), zap.Uint32("token", token), zap.String("interval", interval))
		} else if len(rsiResults) > 0 && !math.IsNaN(rsiResults[len(rsiResults)-1].Value) {
			rsi := db.IndicatorRSI{
				InstrumentToken: token,
				Interval:        interval,
				Period:          rsiPeriod,
				Timestamp:       currentCandle.Timestamp,
				Value:           rsiResults[len(rsiResults)-1].Value,
			}
			im.saveAndBroadcastIndicator(&rsi)
		} else {
			zap.L().Warn("RSI calculation resulted in NaN or empty slice", zap.Uint32("token", token), zap.String("interval", interval))
		}
	} else {
		zap.L().Debug("Not enough candles for RSI calculation", zap.Uint32("token", token), zap.String("interval", interval), zap.Int("needed", rsiPeriod+1), zap.Int("have", len(candles)))
	}

	// --- Stochastic Calculation ---
	stochKPeriod := im.indicatorsCfg.Stochastic.KPeriod
	stochDPeriod := im.indicatorsCfg.Stochastic.DPeriod
	// Needs K period for %K, then D period for SMA of %K.
	stochMinRequired := stochKPeriod + stochDPeriod - 1 // Adjusted minimum for first valid D value
	if len(candles) >= stochMinRequired {
		stochResults, err := indicators.CalculateStochastic(candles, stochKPeriod, stochDPeriod)
		if err != nil {
			zap.L().Error("Stochastic calculation failed", zap.Error(err), zap.Uint32("token", token), zap.String("interval", interval))
		} else if len(stochResults) > 0 &&
			!math.IsNaN(stochResults[len(stochResults)-1].KValue) &&
			!math.IsNaN(stochResults[len(stochResults)-1].DValue) {
			lastStoch := stochResults[len(stochResults)-1]
			stoch := db.IndicatorStochastic{
				InstrumentToken: token,
				Interval:        interval,
				KPeriod:         stochKPeriod,
				DPeriod:         stochDPeriod,
				Timestamp:       currentCandle.Timestamp,
				KValue:          lastStoch.KValue,
				DValue:          lastStoch.DValue,
			}
			im.saveAndBroadcastIndicator(&stoch)
		} else {
			zap.L().Warn("Stochastic calculation resulted in NaN or empty slice", zap.Uint32("token", token), zap.String("interval", interval))
		}
	} else {
		zap.L().Debug("Not enough candles for Stochastic calculation", zap.Uint32("token", token), zap.String("interval", interval), zap.Int("needed", stochMinRequired), zap.Int("have", len(candles)))
	}

	// --- Bollinger Bands Calculation ---
	bbPeriod := im.indicatorsCfg.BollingerBands.Period
	bbNumStdDev := im.indicatorsCfg.BollingerBands.NumStdDev
	if len(candles) >= bbPeriod {
		bbResults, err := indicators.CalculateBollingerBands(candles, bbPeriod, bbNumStdDev)
		if err != nil {
			zap.L().Error("Bollinger Bands calculation failed", zap.Error(err), zap.Uint32("token", token), zap.String("interval", interval))
		} else if len(bbResults) > 0 &&
			!math.IsNaN(bbResults[len(bbResults)-1].UpperBand) &&
			!math.IsNaN(bbResults[len(bbResults)-1].MiddleBand) &&
			!math.IsNaN(bbResults[len(bbResults)-1].LowerBand) {
			lastBB := bbResults[len(bbResults)-1]
			bb := db.IndicatorBollingerBands{
				InstrumentToken: token,
				Interval:        interval,
				Period:          bbPeriod,
				NumStdDev:       bbNumStdDev,
				Timestamp:       currentCandle.Timestamp,
				UpperBand:       lastBB.UpperBand,
				MiddleBand:      lastBB.MiddleBand,
				LowerBand:       lastBB.LowerBand,
			}
			im.saveAndBroadcastIndicator(&bb)
		} else {
			zap.L().Warn("Bollinger Bands calculation resulted in NaN or empty slice", zap.Uint32("token", token), zap.String("interval", interval))
		}
	} else {
		zap.L().Debug("Not enough candles for Bollinger Bands calculation", zap.Uint32("token", token), zap.String("interval", interval), zap.Int("needed", bbPeriod), zap.Int("have", len(candles)))
	}

	// --- OBV Calculation ---
	// OBV typically only needs the previous OBV and current candle, so only 1 historical candle is sufficient
	if len(candles) >= 1 {
		obvResults, err := indicators.CalculateOBV(candles)
		if err != nil {
			zap.L().Error("OBV calculation failed", zap.Error(err), zap.Uint32("token", token), zap.String("interval", interval))
		} else if len(obvResults) > 0 && !math.IsNaN(obvResults[len(obvResults)-1].Value) {
			obv := db.IndicatorOBV{
				InstrumentToken: token,
				Interval:        interval,
				Timestamp:       currentCandle.Timestamp,
				Value:           obvResults[len(obvResults)-1].Value,
			}
			im.saveAndBroadcastIndicator(&obv)
		} else {
			zap.L().Warn("OBV calculation resulted in NaN or empty slice", zap.Uint32("token", token), zap.String("interval", interval))
		}
	} else {
		zap.L().Debug("Not enough candles for OBV calculation", zap.Uint32("token", token), zap.String("interval", interval), zap.Int("needed", 1), zap.Int("have", len(candles)))
	}

	// --- VWAP Calculation ---
	// VWAP is typically session-based (resets daily). It needs *all* candles from the start of the day.
	// For this, we'll assume the `candles` slice provided is from the start of the current trading day
	// or from a point where VWAP should be reset.
	// A robust system would fetch daily candles or ensure `CandleGenerator` provides daily-reset series.
	if len(candles) >= 1 {
		vwapResults, err := indicators.CalculateVWAP(candles)
		if err != nil {
			zap.L().Error("VWAP calculation failed", zap.Error(err), zap.Uint32("token", token), zap.String("interval", interval))
		} else if len(vwapResults) > 0 && !math.IsNaN(vwapResults[len(vwapResults)-1].Value) {
			vwap := db.IndicatorVWAP{
				InstrumentToken: token,
				Interval:        interval,
				Timestamp:       currentCandle.Timestamp,
				Value:           vwapResults[len(vwapResults)-1].Value,
			}
			im.saveAndBroadcastIndicator(&vwap)
		} else {
			zap.L().Warn("VWAP calculation resulted in NaN or empty slice", zap.Uint32("token", token), zap.String("interval", interval))
		}
	} else {
		zap.L().Debug("Not enough candles for VWAP calculation", zap.Uint32("token", token), zap.String("interval", interval), zap.Int("needed", 1), zap.Int("have", len(candles)))
	}

	// --- ADX Calculation ---
	adxPeriod := im.indicatorsCfg.ADX.Period
	// ADX needs period candles for True Range (TR) and Directional Movement (DM), then another period for smoothing.
	// Typically, it needs at least 2*period candles for a stable initial value.
	adxMinRequired := adxPeriod * 2 // A rough estimate for initial ADX, more precise implementations vary
	if len(candles) >= adxMinRequired {
		adxResults, err := indicators.CalculateADX(candles, adxPeriod)
		if err != nil {
			zap.L().Error("ADX calculation failed", zap.Error(err), zap.Uint32("token", token), zap.String("interval", interval))
		} else if len(adxResults) > 0 &&
			!math.IsNaN(adxResults[len(adxResults)-1].ADXValue) &&
			!math.IsNaN(adxResults[len(adxResults)-1].PlusDI) &&
			!math.IsNaN(adxResults[len(adxResults)-1].MinusDI) {
			lastADX := adxResults[len(adxResults)-1]
			adx := db.IndicatorADX{
				InstrumentToken: token,
				Interval:        interval,
				Period:          adxPeriod,
				Timestamp:       currentCandle.Timestamp,
				ADXValue:        lastADX.ADXValue,
				PlusDI:          lastADX.PlusDI,
				MinusDI:         lastADX.MinusDI,
			}
			im.saveAndBroadcastIndicator(&adx)
		} else {
			zap.L().Warn("ADX calculation resulted in NaN or empty slice", zap.Uint32("token", token), zap.String("interval", interval))
		}
	} else {
		zap.L().Debug("Not enough candles for ADX calculation", zap.Uint32("token", token), zap.String("interval", interval), zap.Int("needed", adxMinRequired), zap.Int("have", len(candles)))
	}
}

// saveAndBroadcastIndicator saves the indicator to the database and broadcasts it via WebSocket.
// It uses a type switch to handle different indicator structs.
func (im *IndicatorManager) saveAndBroadcastIndicator(indicator interface{}) {
	var err error
	var instrumentToken uint32
	var interval string
	var timestamp time.Time
	var coreLogFields []zap.Field
	var indicatorName string // Declare variable to hold the indicator name

	switch v := indicator.(type) {
	case *db.IndicatorSMA:
		indicatorName = "SMA" // Set the name
		v.IndicatorName = indicatorName // Assign to the struct
		err = im.dbClient.DB.Clauses(clause.OnConflict{
			Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "period"}, {Name: "timestamp"}},
			DoUpdates: clause.AssignmentColumns([]string{"value", "updated_at"}),
		}).Create(v).Error
		instrumentToken = v.InstrumentToken
		interval = v.Interval
		timestamp = v.Timestamp
		coreLogFields = []zap.Field{zap.String("type", "SMA"), zap.Float64("value", v.Value)}
	case *db.IndicatorEMA:
		indicatorName = "EMA" // Set the name
		v.IndicatorName = indicatorName // Assign to the struct
		err = im.dbClient.DB.Clauses(clause.OnConflict{
			Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "period"}, {Name: "timestamp"}},
			DoUpdates: clause.AssignmentColumns([]string{"value", "updated_at"}),
		}).Create(v).Error
		instrumentToken = v.InstrumentToken
		interval = v.Interval
		timestamp = v.Timestamp
		coreLogFields = []zap.Field{zap.String("type", "EMA"), zap.Float64("value", v.Value)}
	case *db.IndicatorMACD:
		indicatorName = "MACD" // Set the name
		v.IndicatorName = indicatorName // Assign to the struct
		err = im.dbClient.DB.Clauses(clause.OnConflict{
			Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "fast_period"}, {Name: "slow_period"}, {Name: "signal_period"}, {Name: "timestamp"}},
			DoUpdates: clause.AssignmentColumns([]string{"macd_line", "signal_line", "histogram", "updated_at"}),
		}).Create(v).Error
		instrumentToken = v.InstrumentToken
		interval = v.Interval
		timestamp = v.Timestamp
		coreLogFields = []zap.Field{zap.String("type", "MACD"), zap.Float64("macd_line", v.MACDLine), zap.Float64("signal_line", v.SignalLine), zap.Float64("histogram", v.Histogram)}
	case *db.IndicatorATR:
		indicatorName = "ATR" // Set the name
		v.IndicatorName = indicatorName // Assign to the struct
		err = im.dbClient.DB.Clauses(clause.OnConflict{
			Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "period"}, {Name: "timestamp"}},
			DoUpdates: clause.AssignmentColumns([]string{"value", "updated_at"}),
		}).Create(v).Error
		instrumentToken = v.InstrumentToken
		interval = v.Interval
		timestamp = v.Timestamp
		coreLogFields = []zap.Field{zap.String("type", "ATR"), zap.Float64("value", v.Value)}
	case *db.IndicatorRSI:
		indicatorName = "RSI" // Set the name
		v.IndicatorName = indicatorName // Assign to the struct
		err = im.dbClient.DB.Clauses(clause.OnConflict{
			Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "period"}, {Name: "timestamp"}},
			DoUpdates: clause.AssignmentColumns([]string{"value", "updated_at"}),
		}).Create(v).Error
		instrumentToken = v.InstrumentToken
		interval = v.Interval
		timestamp = v.Timestamp
		coreLogFields = []zap.Field{zap.String("type", "RSI"), zap.Float64("value", v.Value)}
	case *db.IndicatorStochastic:
		indicatorName = "Stochastic" // Set the name
		v.IndicatorName = indicatorName // Assign to the struct
		err = im.dbClient.DB.Clauses(clause.OnConflict{
			Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "k_period"}, {Name: "d_period"}, {Name: "timestamp"}},
			DoUpdates: clause.AssignmentColumns([]string{"k_value", "d_value", "updated_at"}),
		}).Create(v).Error
		instrumentToken = v.InstrumentToken
		interval = v.Interval
		timestamp = v.Timestamp
		coreLogFields = []zap.Field{zap.String("type", "Stochastic"), zap.Float64("k_value", v.KValue), zap.Float64("d_value", v.DValue)}
	case *db.IndicatorBollingerBands:
		indicatorName = "BollingerBands" // Set the name
		v.IndicatorName = indicatorName // Assign to the struct
		err = im.dbClient.DB.Clauses(clause.OnConflict{
			Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "period"}, {Name: "num_std_dev"}, {Name: "timestamp"}},
			DoUpdates: clause.AssignmentColumns([]string{"upper_band", "middle_band", "lower_band", "updated_at"}),
		}).Create(v).Error
		instrumentToken = v.InstrumentToken
		interval = v.Interval
		timestamp = v.Timestamp
		coreLogFields = []zap.Field{zap.String("type", "BollingerBands"), zap.Float64("upper", v.UpperBand), zap.Float64("middle", v.MiddleBand), zap.Float64("lower", v.LowerBand)}
	case *db.IndicatorOBV:
		indicatorName = "OBV" // Set the name
		v.IndicatorName = indicatorName // Assign to the struct
		err = im.dbClient.DB.Clauses(clause.OnConflict{
			Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "timestamp"}},
			DoUpdates: clause.AssignmentColumns([]string{"value", "updated_at"}),
		}).Create(v).Error
		instrumentToken = v.InstrumentToken
		interval = v.Interval
		timestamp = v.Timestamp
		coreLogFields = []zap.Field{zap.String("type", "OBV"), zap.Float64("value", v.Value)}
	case *db.IndicatorVWAP:
		indicatorName = "VWAP" // Set the name
		v.IndicatorName = indicatorName // Assign to the struct
		err = im.dbClient.DB.Clauses(clause.OnConflict{
			Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "timestamp"}},
			DoUpdates: clause.AssignmentColumns([]string{"value", "updated_at"}),
		}).Create(v).Error
		instrumentToken = v.InstrumentToken
		interval = v.Interval
		timestamp = v.Timestamp
		coreLogFields = []zap.Field{zap.String("type", "VWAP"), zap.Float64("value", v.Value)}
	case *db.IndicatorADX:
		indicatorName = "ADX" // Set the name
		v.IndicatorName = indicatorName // Assign to the struct
		err = im.dbClient.DB.Clauses(clause.OnConflict{
			Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "period"}, {Name: "timestamp"}},
			DoUpdates: clause.AssignmentColumns([]string{"adx_value", "plus_di", "minus_di", "updated_at"}),
		}).Create(v).Error
		instrumentToken = v.InstrumentToken
		interval = v.Interval
		timestamp = v.Timestamp
		coreLogFields = []zap.Field{zap.String("type", "ADX"), zap.Float64("adx_value", v.ADXValue), zap.Float64("plus_di", v.PlusDI), zap.Float64("minus_di", v.MinusDI)}
	default:
		zap.L().Error("Unknown indicator type received for saving and broadcasting", zap.Any("indicator", indicator))
		return
	}

	if err != nil {
		fields := []zap.Field{
			zap.Error(err),
			zap.Uint32("instrument_token", instrumentToken),
			zap.String("interval", interval),
			zap.Time("timestamp", timestamp),
		}
		fields = append(fields, coreLogFields...)
		zap.L().Error("❌ Failed to save/update indicator to DB", fields...)
		return
	}

	// Broadcast the indicator data via WebSocket
	message, marshalErr := json.Marshal(map[string]interface{}{
		"type":            "INDICATOR_UPDATE",
		"instrumentToken": instrumentToken,
		"interval":        interval,
		"timestamp":       timestamp,
		"indicator":       indicator, // This will now include the IndicatorName field
	})
	if marshalErr != nil {
		zap.L().Error("Failed to marshal indicator for WebSocket broadcast", zap.Error(marshalErr),
			zap.Uint32("token", instrumentToken), zap.String("interval", interval), zap.Any("indicator", indicator))
		return
	}

	im.indicatorWsClients.Range(func(key, value interface{}) bool {
		conn, ok := value.(*websocket.Conn)
		if !ok {
			zap.L().Warn("Found non-websocket.Conn in indicatorWsClients map, deleting.", zap.Any("key", key))
			im.indicatorWsClients.Delete(key)
			return true
		}

		err := conn.WriteMessage(websocket.TextMessage, message)
		if err != nil {
			zap.L().Error("Failed to write indicator message to WebSocket client", zap.Error(err), zap.String("client_key", fmt.Sprintf("%v", key)))
			// If there's a write error, it's often a closed connection, so we can remove it.
			im.indicatorWsClients.Delete(key)
		}
		return true
	})

	// Consolidated log for successful save and broadcast
	finalLogFields := []zap.Field{
		zap.Uint32("instrument_token", instrumentToken),
		zap.String("interval", interval),
		zap.Time("timestamp", timestamp),
	}
	finalLogFields = append(finalLogFields, coreLogFields...)
	zap.L().Info("✅ Saved/Updated indicator to DB and Broadcasted to WebSocket", finalLogFields...)
}