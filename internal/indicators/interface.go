package indicators

import (
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/utils" // Import for AppConfig and IndicatorsConfig
)

// CommonIndicatorResult embeds common fields for all indicator results.
// This helps in standardizing the database storage and WebSocket broadcasting.
type CommonIndicatorResult struct {
	IndicatorName   string    `json:"indicator_name" gorm:"index:idx_indicator_unique,unique"` // e.g., "SMA", "RSI", "MACD"
	InstrumentToken uint32    `json:"instrument_token" gorm:"index:idx_indicator_unique,unique"`
	Interval        string    `json:"interval" gorm:"index:idx_indicator_unique,unique"`
	Timestamp       time.Time `json:"timestamp" gorm:"index:idx_indicator_unique,unique"` // Timestamp of the candle for which this indicator is calculated
	// gorm:"index:idx_indicator_unique,unique" ensures that for a given instrument, interval,
	// timestamp, and indicator name, there's only one entry.
}

// IndicatorCalculationResult holds the result of a single indicator calculation
// and any error encountered during its computation.
type IndicatorCalculationResult struct {
	Indicator CommonIndicatorResult
	Value     interface{} // The actual indicator struct (e.g., SMA, MACD)
	Err       error
}

// Indicator interface defines the contract for all technical indicators.
// Each indicator calculation function should implement this interface.
type Indicator interface {
	// Calculate takes a slice of candles and returns the latest indicator result
	// and an error if calculation fails. The result should embed CommonIndicatorResult.
	Calculate(candles []Candle, appCfg *utils.AppConfig, indicatorsCfg *utils.IndicatorsConfig) (interface{}, error)
	// GetName returns the unique name of the indicator (e.g., "SMA", "RSI").
	GetName() string
	// GetMinRequiredCandles returns the minimum number of candles required to calculate
	// this indicator with its configured periods.
	GetMinRequiredCandles(indicatorsCfg *utils.IndicatorsConfig) int
	// IsEnabled checks if the indicator is enabled in the configuration.
	IsEnabled(indicatorsCfg *utils.IndicatorsConfig) bool
}

// IndicatorResult interface defines the contract for all indicator result types.
// All indicator result structs (SMA, EMA, etc.) should implement this.
type IndicatorResult interface {
	GetInstrumentToken() uint32
	GetInterval() string
	GetTimestamp() time.Time
	GetIndicatorName() string
}
