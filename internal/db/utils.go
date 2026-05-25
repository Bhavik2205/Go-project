package db

import (
	"fmt"

	"github.com/Bhavik2205/ML-Bot/internal/indicators" // Assuming `indicators` package is where types are defined
	"go.uber.org/zap"                                  // Added zap import
)

// GetUpdatableIndicatorColumns returns the columns that should be updated on conflict
// for a given indicator type. This is used by GORM's OnConflict clause.
func GetUpdatableIndicatorColumns(indicatorName string) []string {
	// Common columns that are always updated on conflict (e.g., the value itself and UpdatedAt)
	updatable := []string{"updated_at"}

	switch indicatorName {
	case "SMA", "EMA", "ATR", "RSI", "OBV", "VWAP":
		updatable = append(updatable, "value")
	case "MACD":
		updatable = append(updatable, "macd_line", "signal_line", "histogram")
	case "Stochastic":
		updatable = append(updatable, "k_value", "d_value")
	case "BollingerBands":
		updatable = append(updatable, "upper_band", "middle_band", "lower_band")
	case "ADX":
		updatable = append(updatable, "adx_value", "plus_di", "minus_di")
	default:
		// If an unknown indicator name is passed, this will return only "updated_at".
		// The error will likely still occur for such indicators if they have other
		// fields that aren't being updated, but it helps for known ones.
	}
	return updatable
}

// GetDBModelForIndicator converts an indicators.IndicatorResult interface to its corresponding
// concrete GORM model type. This is necessary for GORM's CreateInBatches.
func GetDBModelForIndicator(indicator indicators.IndicatorResult) interface{} {
	switch v := indicator.(type) {
	case indicators.SMA:
		return IndicatorSMA{
			InstrumentToken: v.GetInstrumentToken(),
			Interval:        v.GetInterval(),
			Period:          v.Period, // Access concrete field
			Timestamp:       v.GetTimestamp(),
			Value:           v.Value,
		}
	case indicators.EMA:
		return IndicatorEMA{
			InstrumentToken: v.GetInstrumentToken(),
			Interval:        v.GetInterval(),
			Period:          v.Period,
			Timestamp:       v.GetTimestamp(),
			Value:           v.Value,
		}
	case indicators.MACD:
		return IndicatorMACD{
			InstrumentToken: v.GetInstrumentToken(),
			Interval:        v.GetInterval(),
			FastPeriod:      v.FastPeriod,
			SlowPeriod:      v.SlowPeriod,
			SignalPeriod:    v.SignalPeriod,
			Timestamp:       v.GetTimestamp(),
			MACDLine:        v.MACDLine,
			SignalLine:      v.SignalLine,
			Histogram:       v.Histogram,
		}
	case indicators.ATR:
		return IndicatorATR{
			InstrumentToken: v.GetInstrumentToken(),
			Interval:        v.GetInterval(),
			Period:          v.Period,
			Timestamp:       v.GetTimestamp(),
			Value:           v.Value,
		}
	case indicators.RSI:
		return IndicatorRSI{
			InstrumentToken: v.GetInstrumentToken(),
			Interval:        v.GetInterval(),
			Period:          v.Period,
			Timestamp:       v.GetTimestamp(),
			Value:           v.Value,
		}
	case indicators.Stochastic:
		return IndicatorStochastic{
			InstrumentToken: v.GetInstrumentToken(),
			Interval:        v.GetInterval(),
			KPeriod:         v.KPeriod,
			DPeriod:         v.DPeriod,
			Timestamp:       v.GetTimestamp(),
			KValue:          v.KValue,
			DValue:          v.DValue,
		}
	case indicators.BollingerBands:
		return IndicatorBollingerBands{
			InstrumentToken: v.GetInstrumentToken(),
			Interval:        v.GetInterval(),
			Period:          v.Period,
			NumStdDev:       v.NumStdDev,
			Timestamp:       v.GetTimestamp(),
			UpperBand:       v.UpperBand,
			MiddleBand:      v.MiddleBand,
			LowerBand:       v.LowerBand,
		}
	case indicators.OBV:
		return IndicatorOBV{
			InstrumentToken: v.GetInstrumentToken(),
			Interval:        v.GetInterval(),
			Timestamp:       v.GetTimestamp(),
			Value:           v.Value,
		}
	case indicators.VWAP:
		return IndicatorVWAP{
			InstrumentToken: v.GetInstrumentToken(),
			Interval:        v.GetInterval(),
			Timestamp:       v.GetTimestamp(),
			Value:           v.Value,
		}
	case indicators.ADX:
		return IndicatorADX{
			InstrumentToken: v.GetInstrumentToken(),
			Interval:        v.GetInterval(),
			Period:          v.Period,
			Timestamp:       v.GetTimestamp(),
			ADXValue:        v.ADXValue,
			PlusDI:          v.PlusDI,
			MinusDI:         v.MinusDI,
		}
	default:
		// Fallback for unexpected types - should log an error or panic in production
		zap.L().Error("Unknown indicator type for DB model conversion", zap.Any("indicator_type", fmt.Sprintf("%T", indicator)))
		return nil
	}
}
