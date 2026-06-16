// internal/indicators/indicators.go
package indicators

import (
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/utils" // Import utils for config access
	"go.uber.org/zap"                             // Import zap for logging
)

func dataSourceFromConfig(appCfg *utils.AppConfig) string {
	if appCfg.Market.Simulate {
		return "simulation"
	}
	return "live" // or "kite" if you want to be more specific
}

// Helper function to get closing prices from a slice of candles
func getClosingPrices(candles []Candle) []float64 {
	prices := make([]float64, len(candles))
	for i, candle := range candles {
		prices[i] = candle.Close
	}
	return prices
}

// Helper function to get High, Low, Close prices
func getOHLC(candles []Candle) ([]float64, []float64, []float64) {
	highs := make([]float64, len(candles))
	lows := make([]float64, len(candles))
	closes := make([]float64, len(candles))
	for i, candle := range candles {
		highs[i] = candle.High
		lows[i] = candle.Low
		closes[i] = candle.Close
	}
	return highs, lows, closes
}

// Common methods for all indicators to satisfy the Indicator interface
func (s SMA) GetName() string { return "SMA" }

func (s SMA) GetMinRequiredCandles(indicatorsCfg *utils.IndicatorsConfig) int {
	return indicatorsCfg.SMA.Period
}

func (s SMA) IsEnabled(indicatorsCfg *utils.IndicatorsConfig) bool {
	return indicatorsCfg.SMA.Enabled
}

func (s SMA) Calculate(candles []Candle, appCfg *utils.AppConfig, indicatorsCfg *utils.IndicatorsConfig) (interface{}, error) {
	period := indicatorsCfg.SMA.Period
	if period <= 0 {
		err := errors.New("SMA period must be greater than 0")
		zap.L().Error("Invalid SMA period", zap.Int("period", period), zap.Error(err))
		return nil, err
	}
	if len(candles) < period {
		err := errors.New("not enough candles to calculate SMA for the given period")
		zap.L().Warn("Insufficient data for SMA calculation",
			zap.Int("required", period), zap.Int("provided", len(candles)), zap.Error(err))
		return nil, err
	}

	prices := getClosingPrices(candles)
	results := make([]SMA, 0, len(candles)-period+1)

	var sum float64
	for i := 0; i < period; i++ {
		sum += prices[i]
	}

	for i := period - 1; i < len(prices); i++ {
		currentSMA := sum / float64(period)

		results = append(results, SMA{
			IndicatorName:   s.GetName(), // Set indicator name
			InstrumentToken: candles[i].InstrumentToken,
			Interval:        candles[i].Interval,
			Period:          period,
			Timestamp:       candles[i].Timestamp,
			Value:           currentSMA,
			DataSource:      dataSourceFromConfig(appCfg),
		})

		if i < len(prices)-1 {
			sum = sum - prices[i-period+1] + prices[i+1]
		}
	}
	return results, nil
}

func (e EMA) GetName() string { return "EMA" }

func (e EMA) GetMinRequiredCandles(indicatorsCfg *utils.IndicatorsConfig) int {
	// For EMA, usually the longer period dictates the min candles if used standalone,
	// but often short_period is the one needing more data for initial SMA.
	// For simplicity, let's use the longer period if both are considered for min.
	// If only short_period is used for min candles, the below might be just indicatorsCfg.EMA.ShortPeriod.
	// Considering the `CalculateEMA` function, it needs `period` candles for the initial SMA.
	return indicatorsCfg.EMA.LongPeriod // Or max(indicatorsCfg.EMA.ShortPeriod, indicatorsCfg.EMA.LongPeriod)
}

func (e EMA) IsEnabled(indicatorsCfg *utils.IndicatorsConfig) bool {
	return indicatorsCfg.EMA.Enabled
}

func (e EMA) Calculate(candles []Candle, appCfg *utils.AppConfig, indicatorsCfg *utils.IndicatorsConfig) (interface{}, error) {
	period := indicatorsCfg.EMA.ShortPeriod // Assuming `Period` field is added for generic EMA in config. Otherwise, need to choose short/long based on context.
	// For now, let's use the ShortPeriod as the main period for the generic EMA calculation.
	// If both short and long EMAs are calculated separately (like in MACD), this function might need to be adjusted.
	// For this generic CalculateEMA, we'll assume it's for one period.
	if period <= 0 {
		err := errors.New("EMA period must be greater than 0")
		zap.L().Error("Invalid EMA period", zap.Int("period", period), zap.Error(err))
		return nil, err
	}
	if len(candles) < period {
		err := errors.New("not enough candles to calculate EMA for the given period")
		zap.L().Warn("Insufficient data for EMA calculation",
			zap.Int("required", period), zap.Int("provided", len(candles)), zap.Error(err))
		return nil, err
	}

	prices := getClosingPrices(candles)
	results := make([]EMA, 0, len(candles)-period+1)

	multiplier := 2.0 / (float64(period) + 1.0)

	var initialSMA float64
	for i := 0; i < period; i++ {
		initialSMA += prices[i]
	}
	currentEMA := initialSMA / float64(period)

	results = append(results, EMA{
		IndicatorName:   e.GetName(), // Set indicator name
		InstrumentToken: candles[period-1].InstrumentToken,
		Interval:        candles[period-1].Interval,
		Period:          period,
		Timestamp:       candles[period-1].Timestamp,
		Value:           currentEMA,
		DataSource:      dataSourceFromConfig(appCfg),
	})

	for i := period; i < len(prices); i++ {
		currentEMA = (prices[i]-currentEMA)*multiplier + currentEMA
		results = append(results, EMA{
			IndicatorName:   e.GetName(), // Set indicator name
			InstrumentToken: candles[i].InstrumentToken,
			Interval:        candles[i].Interval,
			Period:          period,
			Timestamp:       candles[i].Timestamp,
			Value:           currentEMA,
			DataSource:      dataSourceFromConfig(appCfg),
		})
	}
	return results, nil
}

func (r RSI) GetName() string { return "RSI" }

func (r RSI) GetMinRequiredCandles(indicatorsCfg *utils.IndicatorsConfig) int {
	return indicatorsCfg.RSI.Period + 1
}

func (r RSI) IsEnabled(indicatorsCfg *utils.IndicatorsConfig) bool {
	return indicatorsCfg.RSI.Enabled
}

func (r RSI) Calculate(candles []Candle, appCfg *utils.AppConfig, indicatorsCfg *utils.IndicatorsConfig) (interface{}, error) {
	period := indicatorsCfg.RSI.Period
	if period <= 0 {
		err := errors.New("RSI period must be greater than 0")
		zap.L().Error("Invalid RSI period", zap.Int("period", period), zap.Error(err))
		return nil, err
	}
	if len(candles) < period+1 {
		err := errors.New("not enough candles to calculate RSI for the given period")
		zap.L().Warn("Insufficient data for RSI calculation",
			zap.Int("required", period+1), zap.Int("provided", len(candles)), zap.Error(err))
		return nil, err
	}

	prices := getClosingPrices(candles)
	results := make([]RSI, 0, len(candles)-period)

	var avgGain float64
	var avgLoss float64

	for i := 1; i <= period; i++ {
		change := prices[i] - prices[i-1]
		if change > 0 {
			avgGain += change
		} else {
			avgLoss += -change
		}
	}

	avgGain /= float64(period)
	avgLoss /= float64(period)

	for i := period; i < len(prices); i++ {
		var currentGain float64
		var currentLoss float64

		if i > period {
			change := prices[i] - prices[i-1]
			if change > 0 {
				currentGain = change
			} else {
				currentLoss = -change
			}

			avgGain = ((avgGain * float64(period-1)) + currentGain) / float64(period)
			avgLoss = ((avgLoss * float64(period-1)) + currentLoss) / float64(period)
		}

		rs := 0.0
		if avgLoss != 0 {
			rs = avgGain / avgLoss
		}

		rsi := 0.0
		if avgLoss == 0 {
			rsi = 100.0
		} else {
			rsi = 100.0 - (100.0 / (1.0 + rs))
		}

		results = append(results, RSI{
			IndicatorName:   r.GetName(), // Set indicator name
			InstrumentToken: candles[i].InstrumentToken,
			Interval:        candles[i].Interval,
			Period:          period,
			Timestamp:       candles[i].Timestamp,
			Value:           rsi,
			DataSource:      dataSourceFromConfig(appCfg),
		})
	}
	return results, nil
}

func (m MACD) GetName() string { return "MACD" }

func (m MACD) GetMinRequiredCandles(indicatorsCfg *utils.IndicatorsConfig) int {
	// MACD requires sufficient data for the longest EMA period (slowPeriod)
	// and then additional data for the signal EMA.
	return indicatorsCfg.MACD.SlowPeriod + indicatorsCfg.MACD.SignalPeriod - 1
}

func (m MACD) IsEnabled(indicatorsCfg *utils.IndicatorsConfig) bool {
	return indicatorsCfg.MACD.Enabled
}

func (m MACD) Calculate(candles []Candle, appCfg *utils.AppConfig, indicatorsCfg *utils.IndicatorsConfig) (interface{}, error) {
	fastPeriod := indicatorsCfg.MACD.FastPeriod
	slowPeriod := indicatorsCfg.MACD.SlowPeriod
	signalPeriod := indicatorsCfg.MACD.SignalPeriod

	if fastPeriod <= 0 || slowPeriod <= 0 || signalPeriod <= 0 {
		err := errors.New("MACD periods must be greater than 0")
		zap.L().Error("Invalid MACD period(s)", zap.Int("fast", fastPeriod), zap.Int("slow", slowPeriod), zap.Int("signal", signalPeriod), zap.Error(err))
		return nil, err
	}
	if fastPeriod >= slowPeriod {
		err := errors.New("MACD fast period must be less than slow period")
		zap.L().Error("Invalid MACD period configuration", zap.Int("fast", fastPeriod), zap.Int("slow", slowPeriod), zap.Error(err))
		return nil, err
	}

	minRequiredCandles := m.GetMinRequiredCandles(indicatorsCfg)
	if len(candles) < minRequiredCandles {
		err := errors.New("not enough candles to calculate MACD for the given periods")
		zap.L().Warn("Insufficient data for MACD calculation",
			zap.Int("required", minRequiredCandles), zap.Int("provided", len(candles)), zap.Error(err))
		return nil, err
	}

	results := make([]MACD, 0)

	// Create temporary EMA structs for calculation
	emaCalculator := EMA{}
	emaConfigWithFastPeriod := *indicatorsCfg // Create a mutable copy
	emaConfigWithFastPeriod.EMA.ShortPeriod = fastPeriod

	fastEMAsRaw, err := emaCalculator.Calculate(candles, appCfg, &emaConfigWithFastPeriod)
	if err != nil {
		zap.L().Error("Error calculating fast EMA for MACD", zap.Error(err))
		return nil, errors.New("error calculating fast EMA for MACD: " + err.Error())
	}
	fastEMAs := fastEMAsRaw.([]EMA)
	fastEMAMap := make(map[time.Time]float64)
	for _, ema := range fastEMAs {
		fastEMAMap[ema.Timestamp] = ema.Value
	}

	emaConfigWithSlowPeriod := *indicatorsCfg
	emaConfigWithSlowPeriod.EMA.ShortPeriod = slowPeriod

	slowEMAsRaw, err := emaCalculator.Calculate(candles, appCfg, &emaConfigWithSlowPeriod)
	if err != nil {
		zap.L().Error("Error calculating slow EMA for MACD", zap.Error(err))
		return nil, errors.New("error calculating slow EMA for MACD: " + err.Error())
	}
	slowEMAs := slowEMAsRaw.([]EMA)
	slowEMAMap := make(map[time.Time]float64)
	for _, ema := range slowEMAs {
		slowEMAMap[ema.Timestamp] = ema.Value
	}

	macdLines := make([]struct {
		Timestamp time.Time
		Value     float64
	}, 0)

	for i := max(fastPeriod, slowPeriod) - 1; i < len(candles); i++ {
		ts := candles[i].Timestamp
		fastEMA, okFast := fastEMAMap[ts]
		slowEMA, okSlow := slowEMAMap[ts]

		if okFast && okSlow {
			macdLines = append(macdLines, struct {
				Timestamp time.Time
				Value     float64
			}{
				Timestamp: ts,
				Value:     fastEMA - slowEMA,
			})
		}
	}

	if len(macdLines) < signalPeriod {
		err := errors.New("not enough MACD line data to calculate signal line")
		zap.L().Warn("Insufficient MACD line data for signal line calculation",
			zap.Int("required", signalPeriod), zap.Int("provided", len(macdLines)), zap.Error(err))
		return nil, err
	}

	macdLineCandles := make([]Candle, len(macdLines))
	for i, line := range macdLines {
		// Find the original candle index to preserve instrument token and interval
		originalCandleIndex := -1
		for j := 0; j < len(candles); j++ {
			if candles[j].Timestamp == line.Timestamp {
				originalCandleIndex = j
				break
			}
		}

		if originalCandleIndex == -1 {
			zap.L().Error("MACD calculation error: could not find original candle for MACD line timestamp", zap.Time("timestamp", line.Timestamp))
			return nil, fmt.Errorf("MACD line timestamp %v not found in candles", line.Timestamp)
			// return nil, errors.New("internal error: MACD line to original candle index out of bounds during signal line preparation")
		}

		macdLineCandles[i] = Candle{
			InstrumentToken: candles[originalCandleIndex].InstrumentToken,
			Interval:        candles[originalCandleIndex].Interval,
			Close:           line.Value,
			Timestamp:       line.Timestamp,
		}
	}

	emaConfigWithSignalPeriod := *indicatorsCfg
	emaConfigWithSignalPeriod.EMA.ShortPeriod = signalPeriod

	signalEMAsRaw, err := emaCalculator.Calculate(macdLineCandles, appCfg, &emaConfigWithSignalPeriod)
	if err != nil {
		zap.L().Error("Error calculating signal EMA for MACD", zap.Error(err))
		return nil, errors.New("error calculating signal EMA for MACD: " + err.Error())
	}
	signalEMAs := signalEMAsRaw.([]EMA)
	signalEMAMap := make(map[time.Time]float64)
	for _, ema := range signalEMAs {
		signalEMAMap[ema.Timestamp] = ema.Value
	}

	for i := 0; i < len(macdLines); i++ {
		ts := macdLines[i].Timestamp
		macdLineVal := macdLines[i].Value
		signalLineVal, okSignal := signalEMAMap[ts]

		if okSignal {
			originalCandleIndex := -1
			for j := 0; j < len(candles); j++ {
				if candles[j].Timestamp == ts {
					originalCandleIndex = j
					break
				}
			}
			if originalCandleIndex == -1 {
				zap.L().Error("MACD calculation error: could not find original candle for final result timestamp", zap.Time("timestamp", ts))
				return nil, fmt.Errorf("MACD final result timestamp %v not found in candles", ts)
				// return nil, errors.New("internal error: MACD final result to original candle index out of bounds")
			}

			results = append(results, MACD{
				IndicatorName:   m.GetName(), // Set indicator name
				InstrumentToken: candles[originalCandleIndex].InstrumentToken,
				Interval:        candles[originalCandleIndex].Interval,
				FastPeriod:      fastPeriod,
				SlowPeriod:      slowPeriod,
				SignalPeriod:    signalPeriod,
				Timestamp:       ts,
				MACDLine:        macdLineVal,
				SignalLine:      signalLineVal,
				Histogram:       macdLineVal - signalLineVal,
				DataSource:      dataSourceFromConfig(appCfg),
			})
		}
	}
	return results, nil
}

func (a ATR) GetName() string { return "ATR" }

func (a ATR) GetMinRequiredCandles(indicatorsCfg *utils.IndicatorsConfig) int {
	return indicatorsCfg.ATR.Period
}

func (a ATR) IsEnabled(indicatorsCfg *utils.IndicatorsConfig) bool {
	return indicatorsCfg.ATR.Enabled
}

func (a ATR) Calculate(candles []Candle, appCfg *utils.AppConfig, indicatorsCfg *utils.IndicatorsConfig) (interface{}, error) {
	period := indicatorsCfg.ATR.Period
	if period <= 0 {
		err := errors.New("ATR period must be greater than 0")
		zap.L().Error("Invalid ATR period", zap.Int("period", period), zap.Error(err))
		return nil, err
	}
	if len(candles) < period {
		err := errors.New("not enough candles to calculate ATR for the given period")
		zap.L().Warn("Insufficient data for ATR calculation",
			zap.Int("required", period), zap.Int("provided", len(candles)), zap.Error(err))
		return nil, err
	}

	highs, lows, closes := getOHLC(candles)
	results := make([]ATR, 0, len(candles)-period+1)

	trueRanges := make([]float64, len(candles))
	for i := 0; i < len(candles); i++ {
		var highLow float64 = highs[i] - lows[i]
		var highPrevClose float64 = 0
		if i > 0 {
			highPrevClose = math.Abs(highs[i] - closes[i-1])
		}
		var lowPrevClose float64 = 0
		if i > 0 {
			lowPrevClose = math.Abs(lows[i] - closes[i-1])
		}

		trueRanges[i] = math.Max(highLow, math.Max(highPrevClose, lowPrevClose))
	}

	var initialATRSum float64
	for i := 0; i < period; i++ {
		initialATRSum += trueRanges[i]
	}
	currentATR := initialATRSum / float64(period)

	results = append(results, ATR{
		IndicatorName:   a.GetName(), // Set indicator name
		InstrumentToken: candles[period-1].InstrumentToken,
		Interval:        candles[period-1].Interval,
		Period:          period,
		Timestamp:       candles[period-1].Timestamp,
		Value:           currentATR,
		DataSource:      dataSourceFromConfig(appCfg),
	})

	for i := period; i < len(candles); i++ {
		currentATR = ((currentATR * float64(period-1)) + trueRanges[i]) / float64(period)
		results = append(results, ATR{
			IndicatorName:   a.GetName(), // Set indicator name
			InstrumentToken: candles[i].InstrumentToken,
			Interval:        candles[i].Interval,
			Period:          period,
			Timestamp:       candles[i].Timestamp,
			Value:           currentATR,
			DataSource:      dataSourceFromConfig(appCfg),
		})
	}
	return results, nil
}

func (s Stochastic) GetName() string { return "Stochastic" }

func (s Stochastic) GetMinRequiredCandles(indicatorsCfg *utils.IndicatorsConfig) int {
	return indicatorsCfg.Stochastic.KPeriod + indicatorsCfg.Stochastic.DPeriod - 1
}

func (s Stochastic) IsEnabled(indicatorsCfg *utils.IndicatorsConfig) bool {
	return indicatorsCfg.Stochastic.Enabled
}

func (s Stochastic) Calculate(candles []Candle, appCfg *utils.AppConfig, indicatorsCfg *utils.IndicatorsConfig) (interface{}, error) {
	kPeriod := indicatorsCfg.Stochastic.KPeriod
	dPeriod := indicatorsCfg.Stochastic.DPeriod

	if kPeriod <= 0 || dPeriod <= 0 {
		err := errors.New("Stochastic periods must be greater than 0")
		zap.L().Error("Invalid Stochastic period(s)", zap.Int("kPeriod", kPeriod), zap.Int("dPeriod", dPeriod), zap.Error(err))
		return nil, err
	}
	if len(candles) < kPeriod {
		err := errors.New("not enough candles for %K calculation")
		zap.L().Warn("Insufficient data for Stochastic %K calculation",
			zap.Int("required", kPeriod), zap.Int("provided", len(candles)), zap.Error(err))
		return nil, err
	}

	results := make([]Stochastic, 0)
	kValues := make([]float64, 0, len(candles)-kPeriod+1)

	for i := kPeriod - 1; i < len(candles); i++ {
		lookbackCandles := candles[i-kPeriod+1 : i+1]

		var highestHigh float64 = 0
		var lowestLow float64 = math.MaxFloat64

		for _, c := range lookbackCandles {
			if c.High > highestHigh {
				highestHigh = c.High
			}
			if c.Low < lowestLow {
				lowestLow = c.Low
			}
		}

		k := 0.0
		if highestHigh != lowestLow {
			k = ((lookbackCandles[len(lookbackCandles)-1].Close - lowestLow) / (highestHigh - lowestLow)) * 100.0
		} else {
			k = 50.0
		}
		kValues = append(kValues, k)

		if len(kValues) >= dPeriod {
			var dSum float64
			for j := len(kValues) - dPeriod; j < len(kValues); j++ {
				dSum += kValues[j]
			}
			d := dSum / float64(dPeriod)

			results = append(results, Stochastic{
				IndicatorName:   s.GetName(), // Set indicator name
				InstrumentToken: candles[i].InstrumentToken,
				Interval:        candles[i].Interval,
				KPeriod:         kPeriod,
				DPeriod:         dPeriod,
				Timestamp:       candles[i].Timestamp,
				KValue:          k,
				DValue:          d,
				DataSource:      dataSourceFromConfig(appCfg),
			})
		}
	}

	if len(results) == 0 {
		err := errors.New("not enough candles to calculate Stochastic for the given periods after filtering")
		zap.L().Warn("No Stochastic results generated",
			zap.Int("kPeriod", kPeriod), zap.Int("dPeriod", dPeriod), zap.Int("provided", len(candles)), zap.Error(err))
		return nil, err
	}
	return results, nil
}

func (b BollingerBands) GetName() string { return "BollingerBands" }

func (b BollingerBands) GetMinRequiredCandles(indicatorsCfg *utils.IndicatorsConfig) int {
	return indicatorsCfg.BollingerBands.Period
}

func (b BollingerBands) IsEnabled(indicatorsCfg *utils.IndicatorsConfig) bool {
	return indicatorsCfg.BollingerBands.Enabled
}

func (b BollingerBands) Calculate(candles []Candle, appCfg *utils.AppConfig, indicatorsCfg *utils.IndicatorsConfig) (interface{}, error) {
	period := indicatorsCfg.BollingerBands.Period
	numStdDev := indicatorsCfg.BollingerBands.NumStdDev

	if period <= 0 {
		err := errors.New("bollinger bands period must be greater than 0")
		zap.L().Error("Invalid Bollinger Bands period", zap.Int("period", period), zap.Error(err))
		return nil, err
	}
	if numStdDev <= 0 {
		err := errors.New("number of standard deviations must be greater than 0")
		zap.L().Error("Invalid Bollinger Bands StdDev", zap.Float64("numStdDev", numStdDev), zap.Error(err))
		return nil, err
	}
	if len(candles) < period {
		err := errors.New("not enough candles to calculate Bollinger Bands")
		zap.L().Warn("Insufficient data for Bollinger Bands calculation",
			zap.Int("required", period), zap.Int("provided", len(candles)), zap.Error(err))
		return nil, err
	}

	prices := getClosingPrices(candles)
	results := make([]BollingerBands, 0, len(candles)-period+1)

	for i := period - 1; i < len(prices); i++ {
		window := prices[i-period+1 : i+1]

		var sum float64
		for _, p := range window {
			sum += p
		}
		middleBand := sum / float64(period)

		var sumSqDiff float64
		for _, p := range window {
			sumSqDiff += math.Pow(p-middleBand, 2)
		}
		stdDev := math.Sqrt(sumSqDiff / float64(period))

		upperBand := middleBand + (stdDev * numStdDev)
		lowerBand := middleBand - (stdDev * numStdDev)

		results = append(results, BollingerBands{
			IndicatorName:   b.GetName(), // Set indicator name
			InstrumentToken: candles[i].InstrumentToken,
			Interval:        candles[i].Interval,
			Period:          period,
			NumStdDev:       numStdDev,
			Timestamp:       candles[i].Timestamp,
			UpperBand:       upperBand,
			MiddleBand:      middleBand,
			LowerBand:       lowerBand,
			DataSource:      dataSourceFromConfig(appCfg),
		})
	}
	return results, nil
}

func (o OBV) GetName() string { return "OBV" }

func (o OBV) GetMinRequiredCandles(indicatorsCfg *utils.IndicatorsConfig) int {
	return 1 // OBV needs at least one candle to start
}

func (o OBV) IsEnabled(indicatorsCfg *utils.IndicatorsConfig) bool {
	return indicatorsCfg.OBV.Enabled
}

func (o OBV) Calculate(candles []Candle, appCfg *utils.AppConfig, indicatorsCfg *utils.IndicatorsConfig) (interface{}, error) {
	if len(candles) == 0 {
		err := errors.New("no candles provided to calculate OBV")
		zap.L().Warn("No candles for OBV calculation", zap.Error(err))
		return nil, err
	}

	results := make([]OBV, len(candles))
	currentOBV := 0.0

	results[0] = OBV{
		IndicatorName:   o.GetName(), // Set indicator name
		InstrumentToken: candles[0].InstrumentToken,
		Interval:        candles[0].Interval,
		Timestamp:       candles[0].Timestamp,
		Value:           candles[0].Volume,
		DataSource:      dataSourceFromConfig(appCfg),
	}
	currentOBV = candles[0].Volume

	for i := 1; i < len(candles); i++ {
		if candles[i].Close > candles[i-1].Close {
			currentOBV += candles[i].Volume
		} else if candles[i].Close < candles[i-1].Close {
			currentOBV -= candles[i].Volume
		}

		results[i] = OBV{
			IndicatorName:   o.GetName(), // Set indicator name
			InstrumentToken: candles[i].InstrumentToken,
			Interval:        candles[i].Interval,
			Timestamp:       candles[i].Timestamp,
			Value:           currentOBV,
			DataSource:      dataSourceFromConfig(appCfg),
		}
	}
	return results, nil
}

func (v VWAP) GetName() string { return "VWAP" }

func (v VWAP) GetMinRequiredCandles(indicatorsCfg *utils.IndicatorsConfig) int {
	return 1 // VWAP needs at least one candle
}

func (v VWAP) IsEnabled(indicatorsCfg *utils.IndicatorsConfig) bool {
	return indicatorsCfg.VWAP.Enabled
}

func (v VWAP) Calculate(candles []Candle, appCfg *utils.AppConfig, indicatorsCfg *utils.IndicatorsConfig) (interface{}, error) {
	if len(candles) == 0 {
		err := errors.New("no candles provided to calculate VWAP")
		zap.L().Warn("No candles for VWAP calculation", zap.Error(err))
		return nil, err
	}

	results := make([]VWAP, len(candles))
	var cumulativePriceVolume float64
	var cumulativeVolume float64

	for i, candle := range candles {
		typicalPrice := (candle.Open + candle.High + candle.Low + candle.Close) / 4.0
		priceVolume := typicalPrice * candle.Volume
		cumulativePriceVolume += priceVolume
		cumulativeVolume += candle.Volume

		vwap := 0.0
		if cumulativeVolume > 0 {
			vwap = cumulativePriceVolume / cumulativeVolume
		}

		results[i] = VWAP{
			IndicatorName:   v.GetName(), // Set indicator name
			InstrumentToken: candle.InstrumentToken,
			Interval:        candle.Interval,
			Timestamp:       candle.Timestamp,
			Value:           vwap,
			DataSource:      dataSourceFromConfig(appCfg),
		}
	}
	return results, nil
}

func (a ADX) GetName() string { return "ADX" }

func (a ADX) GetMinRequiredCandles(indicatorsCfg *utils.IndicatorsConfig) int {
	return 2 * indicatorsCfg.ADX.Period // Needs at least 2*period for stable ADX value
}

func (a ADX) IsEnabled(indicatorsCfg *utils.IndicatorsConfig) bool {
	return indicatorsCfg.ADX.Enabled
}

func (a ADX) Calculate(candles []Candle, appCfg *utils.AppConfig, indicatorsCfg *utils.IndicatorsConfig) (interface{}, error) {
	period := indicatorsCfg.ADX.Period
	if period <= 0 {
		err := errors.New("ADX period must be greater than 0")
		zap.L().Error("Invalid ADX period", zap.Int("period", period), zap.Error(err))
		return nil, err
	}
	minRequiredCandles := a.GetMinRequiredCandles(indicatorsCfg)
	if len(candles) < minRequiredCandles {
		err := errors.New("not enough candles to calculate ADX for the given period")
		zap.L().Warn("Insufficient data for ADX calculation",
			zap.Int("required", minRequiredCandles), zap.Int("provided", len(candles)), zap.Error(err))
		return nil, err
	}

	highs, lows, closes := getOHLC(candles)
	results := make([]ADX, 0)

	trueRanges := make([]float64, len(candles))
	for i := 0; i < len(candles); i++ {
		if i == 0 {
			trueRanges[i] = highs[i] - lows[i]
		} else {
			trueRanges[i] = math.Max(highs[i]-lows[i],
				math.Max(math.Abs(highs[i]-closes[i-1]),
					math.Abs(lows[i]-closes[i-1])))
		}
	}

	plusDM := make([]float64, len(candles))
	minusDM := make([]float64, len(candles))

	for i := 1; i < len(candles); i++ {
		upMove := highs[i] - highs[i-1]
		downMove := lows[i-1] - lows[i]

		if upMove > downMove && upMove > 0 {
			plusDM[i] = upMove
		} else {
			plusDM[i] = 0
		}

		if downMove > upMove && downMove > 0 {
			minusDM[i] = downMove
		} else {
			minusDM[i] = 0
		}
	}

	smoothedTR := make([]float64, len(candles))
	smoothedPlusDM := make([]float64, len(candles))
	smoothedMinusDM := make([]float64, len(candles))

	initialTRSum := 0.0
	initialPlusDMSum := 0.0
	initialMinusDMSum := 0.0

	for i := 0; i < period; i++ {
		initialTRSum += trueRanges[i]
		initialPlusDMSum += plusDM[i]
		initialMinusDMSum += minusDM[i]
	}
	smoothedTR[period-1] = initialTRSum
	smoothedPlusDM[period-1] = initialPlusDMSum
	smoothedMinusDM[period-1] = initialMinusDMSum

	for i := period; i < len(candles); i++ {
		smoothedTR[i] = smoothedTR[i-1] - (smoothedTR[i-1] / float64(period)) + trueRanges[i]
		smoothedPlusDM[i] = smoothedPlusDM[i-1] - (smoothedPlusDM[i-1] / float64(period)) + plusDM[i]
		smoothedMinusDM[i] = smoothedMinusDM[i-1] - (smoothedMinusDM[i-1] / float64(period)) + minusDM[i]
	}

	plusDI := make([]float64, len(candles))
	minusDI := make([]float64, len(candles))

	for i := period - 1; i < len(candles); i++ {
		if smoothedTR[i] != 0 {
			plusDI[i] = (smoothedPlusDM[i] / smoothedTR[i]) * 100
			minusDI[i] = (smoothedMinusDM[i] / smoothedTR[i]) * 100
		} else {
			plusDI[i] = 0
			minusDI[i] = 0
		}
	}

	dx := make([]float64, len(candles))
	for i := period - 1; i < len(candles); i++ {
		sumDI := plusDI[i] + minusDI[i]
		if sumDI != 0 {
			dx[i] = (math.Abs(plusDI[i]-minusDI[i]) / sumDI) * 100
		} else {
			dx[i] = 0
		}
	}

	adxValues := make([]float64, len(candles))

	firstDXIndex := period - 1
	if len(dx) < firstDXIndex+period {
		err := errors.New("not enough DX values to calculate initial ADX")
		zap.L().Warn("Insufficient DX values for initial ADX calculation",
			zap.Int("required", firstDXIndex+period), zap.Int("provided", len(dx)), zap.Error(err))
		return nil, err
	}

	initialADXSum := 0.0
	for i := firstDXIndex; i < firstDXIndex+period; i++ {
		initialADXSum += dx[i]
	}
	adxValues[firstDXIndex+period-1] = initialADXSum / float64(period)

	for i := firstDXIndex + period; i < len(candles); i++ {
		adxValues[i] = (adxValues[i-1]*(float64(period)-1) + dx[i]) / float64(period)
	}

	for i := (2*period - 1); i < len(candles); i++ {
		results = append(results, ADX{
			IndicatorName:   a.GetName(), // Set indicator name
			InstrumentToken: candles[i].InstrumentToken,
			Interval:        candles[i].Interval,
			Period:          period,
			Timestamp:       candles[i].Timestamp,
			ADXValue:        adxValues[i],
			PlusDI:          plusDI[i],
			MinusDI:         minusDI[i],
			DataSource:      dataSourceFromConfig(appCfg),
		})
	}

	if len(results) == 0 {
		err := errors.New("not enough candles to calculate ADX for the given periods after filtering")
		zap.L().Warn("No ADX results generated",
			zap.Int("period", period), zap.Int("provided", len(candles)), zap.Error(err))
		return nil, err
	}
	return results, nil
}

// max returns the larger of two integers.
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
