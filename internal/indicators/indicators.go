// internal/indicators/indicators.go
package indicators

import (
	"errors"
	"math"
	"time"

	"go.uber.org/zap" // Import zap for logging
)

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

// CalculateSMA calculates the Simple Moving Average for a given set of candles.
// It returns a slice of SMA results, corresponding to the candles provided.
// The slice will be shorter than the input candles if not enough data is available
// for the initial periods.
func CalculateSMA(candles []Candle, period int) ([]SMA, error) {
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

	// Calculate sum for the first window
	var sum float64
	for i := 0; i < period; i++ {
		sum += prices[i]
	}

	// Calculate SMA for subsequent windows
	for i := period - 1; i < len(prices); i++ {
		// Calculate current SMA
		currentSMA := sum / float64(period)

		// Create SMA struct
		results = append(results, SMA{
			InstrumentToken: candles[i].InstrumentToken,
			Interval:        candles[i].Interval,
			Period:          period,
			Timestamp:       candles[i].Timestamp,
			Value:           currentSMA,
		})

		// Subtract the oldest price and add the new one for the next window, if not at the end
		if i < len(prices)-1 {
			sum = sum - prices[i-period+1] + prices[i+1]
		}
	}

	return results, nil
}

// CalculateEMA calculates the Exponential Moving Average for a given set of candles.
// It returns a slice of EMA results. EMA requires an initial SMA calculation for the first value.
func CalculateEMA(candles []Candle, period int) ([]EMA, error) {
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

	// Calculate the smoothing factor
	multiplier := 2.0 / (float64(period) + 1.0)

	// The first EMA is typically the SMA of the first 'period' closing prices
	var initialSMA float64
	for i := 0; i < period; i++ {
		initialSMA += prices[i]
	}
	currentEMA := initialSMA / float64(period)

	// Add the first EMA value (corresponding to the 'period'-th candle in the input slice)
	results = append(results, EMA{
		InstrumentToken: candles[period-1].InstrumentToken,
		Interval:        candles[period-1].Interval,
		Period:          period,
		Timestamp:       candles[period-1].Timestamp,
		Value:           currentEMA,
	})

	// Calculate subsequent EMAs
	for i := period; i < len(prices); i++ {
		currentEMA = (prices[i]-currentEMA)*multiplier + currentEMA
		results = append(results, EMA{
			InstrumentToken: candles[i].InstrumentToken,
			Interval:        candles[i].Interval,
			Period:          period,
			Timestamp:       candles[i].Timestamp,
			Value:           currentEMA,
		})
	}

	return results, nil
}

// CalculateRSI calculates the Relative Strength Index for a given set of candles.
// It returns a slice of RSI results. RSI requires at least (period + 1) candles.
func CalculateRSI(candles []Candle, period int) ([]RSI, error) {
	if period <= 0 {
		err := errors.New("RSI period must be greater than 0")
		zap.L().Error("Invalid RSI period", zap.Int("period", period), zap.Error(err))
		return nil, err
	}
	if len(candles) < period+1 { // Need period + 1 for the first gain/loss comparison
		err := errors.New("not enough candles to calculate RSI for the given period")
		zap.L().Warn("Insufficient data for RSI calculation",
			zap.Int("required", period+1), zap.Int("provided", len(candles)), zap.Error(err))
		return nil, err
	}

	prices := getClosingPrices(candles)
	results := make([]RSI, 0, len(candles)-period) // The first RSI value is at index 'period'

	// Calculate initial gains and losses for the first 'period' candles
	var avgGain float64
	var avgLoss float64

	for i := 1; i <= period; i++ {
		change := prices[i] - prices[i-1]
		if change > 0 {
			avgGain += change
		} else {
			avgLoss += -change // Absolute value of loss
		}
	}

	avgGain /= float64(period)
	avgLoss /= float64(period)

	// Calculate subsequent average gains and losses
	for i := period; i < len(prices); i++ {
		var currentGain float64
		var currentLoss float64

		if i > period { // For subsequent calculations after the initial average
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
		if avgLoss == 0 { // If no losses, RSI is 100
			rsi = 100.0
		} else {
			rsi = 100.0 - (100.0 / (1.0 + rs))
		}

		results = append(results, RSI{
			InstrumentToken: candles[i].InstrumentToken,
			Interval:        candles[i].Interval,
			Period:          period,
			Timestamp:       candles[i].Timestamp,
			Value:           rsi,
		})
	}

	return results, nil
}

// CalculateMACD calculates the Moving Average Convergence Divergence.
// It returns a slice of MACD results.
// MACD needs a sufficient number of candles for both fast and slow EMAs to be calculated.
// (e.g., max(fastPeriod, slowPeriod) + signalPeriod)
func CalculateMACD(candles []Candle, fastPeriod, slowPeriod, signalPeriod int) ([]MACD, error) {
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

	// MACD requires sufficient data for the longest EMA period (slowPeriod)
	// and then additional data for the signal EMA.
	minRequiredCandles := slowPeriod + signalPeriod - 1 // Approximately: slowPeriod for slowEMA + signalPeriod for signalEMA
	if len(candles) < minRequiredCandles {
		err := errors.New("not enough candles to calculate MACD for the given periods")
		zap.L().Warn("Insufficient data for MACD calculation",
			zap.Int("required", minRequiredCandles), zap.Int("provided", len(candles)), zap.Error(err))
		return nil, err
	}

	results := make([]MACD, 0)

	// Calculate Fast EMA
	fastEMAs, err := CalculateEMA(candles, fastPeriod)
	if err != nil {
		zap.L().Error("Error calculating fast EMA for MACD", zap.Error(err))
		return nil, errors.New("error calculating fast EMA for MACD: " + err.Error())
	}
	fastEMAMap := make(map[time.Time]float64)
	for _, ema := range fastEMAs {
		fastEMAMap[ema.Timestamp] = ema.Value
	}

	// Calculate Slow EMA
	slowEMAs, err := CalculateEMA(candles, slowPeriod)
	if err != nil {
		zap.L().Error("Error calculating slow EMA for MACD", zap.Error(err))
		return nil, errors.New("error calculating slow EMA for MACD: " + err.Error())
	}
	slowEMAMap := make(map[time.Time]float64)
	for _, ema := range slowEMAs {
		slowEMAMap[ema.Timestamp] = ema.Value
	}

	// Calculate MACD Line
	macdLines := make([]struct {
		Timestamp time.Time
		Value     float64
	}, 0)

	// MACD line starts from the point where both EMAs are available
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

	// Calculate Signal Line (EMA of MACD Line)
	macdLineCandles := make([]Candle, len(macdLines))
	for i, line := range macdLines {
		originalCandleIndex := len(candles) - len(macdLines) + i
		if originalCandleIndex < 0 || originalCandleIndex >= len(candles) {
			err := errors.New("internal error: MACD line to original candle index out of bounds during signal line preparation")
			zap.L().Fatal("MACD calculation fatal error", zap.Error(err), zap.Int("originalCandleIndex", originalCandleIndex), zap.Int("lenCandles", len(candles)), zap.Int("lenMacdLines", len(macdLines)), zap.Int("i", i))
			return nil, err // Should not happen in a correctly implemented flow
		}
		macdLineCandles[i] = Candle{
			InstrumentToken: candles[originalCandleIndex].InstrumentToken,
			Interval:        candles[originalCandleIndex].Interval,
			Close:           line.Value,
			Timestamp:       line.Timestamp,
		}
	}

	signalEMAs, err := CalculateEMA(macdLineCandles, signalPeriod)
	if err != nil {
		zap.L().Error("Error calculating signal EMA for MACD", zap.Error(err))
		return nil, errors.New("error calculating signal EMA for MACD: " + err.Error())
	}
	signalEMAMap := make(map[time.Time]float64)
	for _, ema := range signalEMAs {
		signalEMAMap[ema.Timestamp] = ema.Value
	}

	// Combine MACD Line, Signal Line, and calculate Histogram
	for i := 0; i < len(macdLines); i++ {
		ts := macdLines[i].Timestamp
		macdLineVal := macdLines[i].Value
		signalLineVal, okSignal := signalEMAMap[ts]

		if okSignal {
			originalCandleIndex := len(candles) - len(macdLines) + i
			if originalCandleIndex < 0 || originalCandleIndex >= len(candles) {
				err := errors.New("internal error: MACD final result to original candle index out of bounds")
				zap.L().Fatal("MACD calculation fatal error", zap.Error(err), zap.Int("originalCandleIndex", originalCandleIndex), zap.Int("lenCandles", len(candles)), zap.Int("lenMacdLines", len(macdLines)), zap.Int("i", i))
				return nil, err // Should not happen
			}

			results = append(results, MACD{
				InstrumentToken: candles[originalCandleIndex].InstrumentToken,
				Interval:        candles[originalCandleIndex].Interval,
				FastPeriod:      fastPeriod,
				SlowPeriod:      slowPeriod,
				SignalPeriod:    signalPeriod,
				Timestamp:       ts,
				MACDLine:        macdLineVal,
				SignalLine:      signalLineVal,
				Histogram:       macdLineVal - signalLineVal,
			})
		}
	}

	return results, nil
}

// CalculateATR calculates the Average True Range.
// It returns a slice of ATR results. Requires at least `period` candles.
func CalculateATR(candles []Candle, period int) ([]ATR, error) {
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

	// Calculate initial ATR (SMA of first 'period' true ranges)
	var initialATRSum float64
	for i := 0; i < period; i++ {
		initialATRSum += trueRanges[i]
	}
	currentATR := initialATRSum / float64(period)

	results = append(results, ATR{
		InstrumentToken: candles[period-1].InstrumentToken, // ATR corresponds to the period-th candle
		Interval:        candles[period-1].Interval,
		Period:          period,
		Timestamp:       candles[period-1].Timestamp,
		Value:           currentATR,
	})

	// Calculate subsequent ATRs (Smoothed Moving Average type calculation)
	for i := period; i < len(candles); i++ {
		currentATR = ((currentATR * float64(period-1)) + trueRanges[i]) / float64(period)
		results = append(results, ATR{
			InstrumentToken: candles[i].InstrumentToken,
			Interval:        candles[i].Interval,
			Period:          period,
			Timestamp:       candles[i].Timestamp,
			Value:           currentATR,
		})
	}

	return results, nil
}

// CalculateStochastic calculates the Stochastic Oscillator (%K and %D).
// It returns a slice of Stochastic results. Requires at least (kPeriod + dPeriod - 1) candles.
func CalculateStochastic(candles []Candle, kPeriod, dPeriod int) ([]Stochastic, error) {
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
	kValues := make([]float64, 0, len(candles)-kPeriod+1) // Store %K values for %D calculation

	for i := kPeriod - 1; i < len(candles); i++ {
		lookbackCandles := candles[i-kPeriod+1 : i+1]

		var highestHigh float64 = 0
		var lowestLow float64 = math.MaxFloat64 // Initialize with max possible float64

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
			// Handle case where High == Low (flat candle)
			k = 50.0 // Or handle as per preference, 50 is a common neutral value
		}
		kValues = append(kValues, k)

		// We only calculate %D once we have enough %K values for its SMA
		if len(kValues) >= dPeriod {
			var dSum float64
			for j := len(kValues) - dPeriod; j < len(kValues); j++ {
				dSum += kValues[j]
			}
			d := dSum / float64(dPeriod)

			results = append(results, Stochastic{
				InstrumentToken: candles[i].InstrumentToken,
				Interval:        candles[i].Interval,
				KPeriod:         kPeriod,
				DPeriod:         dPeriod,
				Timestamp:       candles[i].Timestamp,
				KValue:          k,
				DValue:          d,
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

// CalculateBollingerBands calculates Bollinger Bands.
// It returns a slice of BollingerBands results. Needs at least `period` candles.
func CalculateBollingerBands(candles []Candle, period int, numStdDev float64) ([]BollingerBands, error) {
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

		// Calculate Middle Band (SMA)
		var sum float64
		for _, p := range window {
			sum += p
		}
		middleBand := sum / float64(period)

		// Calculate Standard Deviation
		var sumSqDiff float64
		for _, p := range window {
			sumSqDiff += math.Pow(p-middleBand, 2)
		}
		stdDev := math.Sqrt(sumSqDiff / float64(period))

		upperBand := middleBand + (stdDev * numStdDev)
		lowerBand := middleBand - (stdDev * numStdDev)

		results = append(results, BollingerBands{
			InstrumentToken: candles[i].InstrumentToken,
			Interval:        candles[i].Interval,
			Period:          period,
			NumStdDev:       numStdDev,
			Timestamp:       candles[i].Timestamp,
			UpperBand:       upperBand,
			MiddleBand:      middleBand,
			LowerBand:       lowerBand,
		})
	}

	return results, nil
}

// CalculateOBV calculates On-Balance Volume.
// It returns a slice of OBV results. Needs at least 1 candle.
func CalculateOBV(candles []Candle) ([]OBV, error) {
	if len(candles) == 0 {
		err := errors.New("no candles provided to calculate OBV")
		zap.L().Warn("No candles for OBV calculation", zap.Error(err))
		return nil, err
	}

	results := make([]OBV, len(candles))
	currentOBV := 0.0

	// The first OBV is the volume of the first candle
	results[0] = OBV{
		InstrumentToken: candles[0].InstrumentToken,
		Interval:        candles[0].Interval,
		Timestamp:       candles[0].Timestamp,
		Value:           candles[0].Volume,
	}
	currentOBV = candles[0].Volume

	for i := 1; i < len(candles); i++ {
		if candles[i].Close > candles[i-1].Close {
			currentOBV += candles[i].Volume
		} else if candles[i].Close < candles[i-1].Close {
			currentOBV -= candles[i].Volume
		}
		// If close == prev close, OBV remains unchanged

		results[i] = OBV{
			InstrumentToken: candles[i].InstrumentToken,
			Interval:        candles[i].Interval,
			Timestamp:       candles[i].Timestamp,
			Value:           currentOBV,
		}
	}
	return results, nil
}

// CalculateVWAP calculates Volume Weighted Average Price.
// VWAP is typically reset at the beginning of each trading day.
// This function calculates VWAP cumulatively over the provided candles,
// assuming they belong to the same trading day.
// If you need daily reset, you'll need to call this function separately for each day's candles.
func CalculateVWAP(candles []Candle) ([]VWAP, error) {
	if len(candles) == 0 {
		err := errors.New("no candles provided to calculate VWAP")
		zap.L().Warn("No candles for VWAP calculation", zap.Error(err))
		return nil, err
	}

	results := make([]VWAP, len(candles))
	var cumulativePriceVolume float64
	var cumulativeVolume float64

	for i, candle := range candles {
		// Use a typical price calculation: (Open + High + Low + Close) / 4
		typicalPrice := (candle.Open + candle.High + candle.Low + candle.Close) / 4.0
		priceVolume := typicalPrice * candle.Volume
		cumulativePriceVolume += priceVolume
		cumulativeVolume += candle.Volume

		vwap := 0.0
		if cumulativeVolume > 0 { // Avoid division by zero
			vwap = cumulativePriceVolume / cumulativeVolume
		}

		results[i] = VWAP{
			InstrumentToken: candle.InstrumentToken,
			Interval:        candle.Interval,
			Timestamp:       candle.Timestamp,
			Value:           vwap,
		}
	}
	return results, nil
}

// CalculateADX calculates the Average Directional Index (ADX) along with +DI and -DI.
// It returns a slice of ADX results. Needs at least `period * 2` candles for a stable initial ADX value.
// Typically period = 14.
func CalculateADX(candles []Candle, period int) ([]ADX, error) {
	if period <= 0 {
		err := errors.New("ADX period must be greater than 0")
		zap.L().Error("Invalid ADX period", zap.Int("period", period), zap.Error(err))
		return nil, err
	}
	// A robust ADX calculation needs sufficient data for initial smoothing of TR/DM,
	// and then additional data for smoothing DX. At least 2*period candles are generally recommended
	// to get a stable first ADX value.
	minRequiredCandles := 2 * period
	if len(candles) < minRequiredCandles {
		err := errors.New("not enough candles to calculate ADX for the given period")
		zap.L().Warn("Insufficient data for ADX calculation",
			zap.Int("required", minRequiredCandles), zap.Int("provided", len(candles)), zap.Error(err))
		return nil, err
	}

	highs, lows, closes := getOHLC(candles)
	results := make([]ADX, 0)

	// Step 1: Calculate True Range (TR)
	trueRanges := make([]float64, len(candles))
	for i := 0; i < len(candles); i++ {
		if i == 0 {
			trueRanges[i] = highs[i] - lows[i] // No previous close for the first candle
		} else {
			trueRanges[i] = math.Max(highs[i]-lows[i],
				math.Max(math.Abs(highs[i]-closes[i-1]),
					math.Abs(lows[i]-closes[i-1])))
		}
	}

	// Step 2: Calculate Directional Movement (+DM and -DM)
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

	// Step 3: Smooth TR, +DM, -DM with Wilder's Smoothing (similar to EMA)
	// The first 'period' values are summed, then subsequent values are smoothed.
	smoothedTR := make([]float64, len(candles))
	smoothedPlusDM := make([]float64, len(candles))
	smoothedMinusDM := make([]float64, len(candles))

	// Initial sum for the first 'period' valid TR/DM values (starting from index 1 for DM)
	// TR values are valid from index 0. DM values are valid from index 1.
	// We need to sum up to 'period' values.
	// The initial smoothed value will correspond to the (period-1)-th candle.
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

	// Subsequent smoothing
	for i := period; i < len(candles); i++ {
		smoothedTR[i] = smoothedTR[i-1] - (smoothedTR[i-1] / float64(period)) + trueRanges[i]
		smoothedPlusDM[i] = smoothedPlusDM[i-1] - (smoothedPlusDM[i-1] / float64(period)) + plusDM[i]
		smoothedMinusDM[i] = smoothedMinusDM[i-1] - (smoothedMinusDM[i-1] / float64(period)) + minusDM[i]
	}

	// Step 4: Calculate +DI and -DI
	plusDI := make([]float64, len(candles))
	minusDI := make([]float64, len(candles))

	// +DI and -DI are calculated starting from the (period-1)-th candle
	for i := period - 1; i < len(candles); i++ {
		if smoothedTR[i] != 0 {
			plusDI[i] = (smoothedPlusDM[i] / smoothedTR[i]) * 100
			minusDI[i] = (smoothedMinusDM[i] / smoothedTR[i]) * 100
		} else {
			plusDI[i] = 0
			minusDI[i] = 0
		}
	}

	// Step 5: Calculate DX (Directional Index)
	dx := make([]float64, len(candles))
	// DX is calculated starting from the (period-1)-th candle
	for i := period - 1; i < len(candles); i++ {
		sumDI := plusDI[i] + minusDI[i]
		if sumDI != 0 {
			dx[i] = (math.Abs(plusDI[i]-minusDI[i]) / sumDI) * 100
		} else {
			dx[i] = 0
		}
	}

	// Step 6: Calculate ADX (Smoothed DX)
	// The first ADX value is the SMA of the first 'period' DX values.
	// This means the first ADX result corresponds to the (period-1) + period = (2*period-1)-th candle.

	adxValues := make([]float64, len(candles))

	// Start index for DX values that contribute to the first ADX
	firstDXIndex := period - 1
	if len(dx) < firstDXIndex+period { // Ensure there are enough DX values for the initial ADX sum
		err := errors.New("not enough DX values to calculate initial ADX")
		zap.L().Warn("Insufficient DX values for initial ADX calculation",
			zap.Int("required", firstDXIndex+period), zap.Int("provided", len(dx)), zap.Error(err))
		return nil, err
	}

	initialADXSum := 0.0
	for i := firstDXIndex; i < firstDXIndex+period; i++ {
		initialADXSum += dx[i]
	}
	adxValues[firstDXIndex+period-1] = initialADXSum / float64(period) // Assign first ADX to its correct candle index

	// Subsequent ADX smoothing using Wilder's smoothing
	for i := firstDXIndex + period; i < len(candles); i++ {
		adxValues[i] = (adxValues[i-1]*(float64(period)-1) + dx[i]) / float64(period)
	}

	// Combine results
	// ADX results are valid from index (2*period - 1) onwards in the original candles slice.
	for i := (2*period - 1); i < len(candles); i++ {
		results = append(results, ADX{
			InstrumentToken: candles[i].InstrumentToken,
			Interval:        candles[i].Interval,
			Period:          period,
			Timestamp:       candles[i].Timestamp,
			ADXValue:        adxValues[i],
			PlusDI:          plusDI[i],
			MinusDI:         minusDI[i],
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
