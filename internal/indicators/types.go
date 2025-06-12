package indicators

import "time"

// Candle represents an OHLCV (Open-High-Low-Close-Volume) candle.
// This struct is used for calculations and as the base data for indicators.
type Candle struct {
	InstrumentToken uint32    `json:"instrument_token"`
	Interval        string    `json:"interval"`  // e.g., "1m", "5m", "1h", "1d"
	Timestamp       time.Time `json:"timestamp"` // Start time of the candle
	Open            float64   `json:"open"`
	High            float64   `json:"high"`
	Low             float64   `json:"low"`
	Close           float64   `json:"close"`
	Volume          float64   `json:"volume"`
	TradeCount      uint32    `json:"trade_count,omitempty"` // Optional
}

// SMA represents a Simple Moving Average value.
type SMA struct {
	IndicatorName   string    `json:"indicator_name"` // Added: For identification
	InstrumentToken uint32    `json:"instrument_token"`
	Interval        string    `json:"interval"`
	Period          int       `json:"period"`    // e.g., 20
	Timestamp       time.Time `json:"timestamp"` // Timestamp of the candle for which this SMA is calculated
	Value           float64   `json:"value"`
}

// EMA represents an Exponential Moving Average value.
type EMA struct {
	IndicatorName   string    `json:"indicator_name"` // Added
	InstrumentToken uint32    `json:"instrument_token"`
	Interval        string    `json:"interval"`
	Period          int       `json:"period"`
	Timestamp       time.Time `json:"timestamp"`
	Value           float64   `json:"value"`
}

// MACD represents Moving Average Convergence Divergence values.
type MACD struct {
	IndicatorName   string    `json:"indicator_name"` // Added
	InstrumentToken uint32    `json:"instrument_token"`
	Interval        string    `json:"interval"`
	FastPeriod      int       `json:"fast_period"`   // e.g., 12
	SlowPeriod      int       `json:"slow_period"`   // e.g., 26
	SignalPeriod    int       `json:"signal_period"` // e.g., 9
	Timestamp       time.Time `json:"timestamp"`
	MACDLine        float64   `json:"macd_line"`
	SignalLine      float64   `json:"signal_line"`
	Histogram       float64   `json:"histogram"` // MACDLine - SignalLine
}

// ATR represents Average True Range value.
type ATR struct {
	IndicatorName   string    `json:"indicator_name"` // Added
	InstrumentToken uint32    `json:"instrument_token"`
	Interval        string    `json:"interval"`
	Period          int       `json:"period"`
	Timestamp       time.Time `json:"timestamp"`
	Value           float64   `json:"value"`
}

// RSI represents Relative Strength Index value.
type RSI struct {
	IndicatorName   string    `json:"indicator_name"` // Added
	InstrumentToken uint32    `json:"instrument_token"`
	Interval        string    `json:"interval"`
	Period          int       `json:"period"`
	Timestamp       time.Time `json:"timestamp"`
	Value           float64   `json:"value"`
}

// Stochastic represents Stochastic Oscillator values.
type Stochastic struct {
	IndicatorName   string    `json:"indicator_name"` // Added
	InstrumentToken uint32    `json:"instrument_token"`
	Interval        string    `json:"interval"`
	KPeriod         int       `json:"k_period"` // %K period
	DPeriod         int       `json:"d_period"` // %D period
	Timestamp       time.Time `json:"timestamp"`
	KValue          float64   `json:"k_value"` // %K line
	DValue          float64   `json:"d_value"` // %D line (SMA of %K)
}

// BollingerBands represents Bollinger Bands values.
type BollingerBands struct {
	IndicatorName   string    `json:"indicator_name"` // Added
	InstrumentToken uint32    `json:"instrument_token"`
	Interval        string    `json:"interval"`
	Period          int       `json:"period"`
	NumStdDev       float64   `json:"num_std_dev"` // Number of standard deviations (e.g., 2.0)
	Timestamp       time.Time `json:"timestamp"`
	UpperBand       float64   `json:"upper_band"`
	MiddleBand      float64   `json:"middle_band"` // Typically a SMA
	LowerBand       float64   `json:"lower_band"`
}

// OBV represents On-Balance Volume value.
type OBV struct {
	IndicatorName   string    `json:"indicator_name"` // Added
	InstrumentToken uint32    `json:"instrument_token"`
	Interval        string    `json:"interval"`
	Timestamp       time.Time `json:"timestamp"`
	Value           float64   `json:"value"`
}

// VWAP represents Volume Weighted Average Price value.
type VWAP struct {
	IndicatorName   string    `json:"indicator_name"` // Added
	InstrumentToken uint32    `json:"instrument_token"`
	Interval        string    `json:"interval"` // Even though VWAP is daily reset, it's calculated over intervals.
	Timestamp       time.Time `json:"timestamp"`
	Value           float64   `json:"value"`
}

// ADX represents Average Directional Index values.
type ADX struct {
	IndicatorName   string    `json:"indicator_name"` // Added
	InstrumentToken uint32    `json:"instrument_token"`
	Interval        string    `json:"interval"`
	Period          int       `json:"period"`
	Timestamp       time.Time `json:"timestamp"`
	ADXValue        float64   `json:"adx_value"` // The ADX line
	PlusDI          float64   `json:"plus_di"`   // Positive Directional Indicator (+DI)
	MinusDI         float64   `json:"minus_di"`  // Negative Directional Indicator (-DI)
}

// Indicator interface for common methods across all indicator types.
// This allows for polymorphic handling of different indicators.
type Indicator interface {
	GetInstrumentToken() uint32
	GetInterval() string
	GetTimestamp() time.Time
	GetIndicatorName() string // Added: Get the name of the indicator
}

// --- Implementations of the Indicator interface for each struct ---

func (s SMA) GetInstrumentToken() uint32 { return s.InstrumentToken }
func (s SMA) GetInterval() string        { return s.Interval }
func (s SMA) GetTimestamp() time.Time    { return s.Timestamp }
func (s SMA) GetIndicatorName() string   { return s.IndicatorName } // Added

func (e EMA) GetInstrumentToken() uint32 { return e.InstrumentToken }
func (e EMA) GetInterval() string        { return e.Interval }
func (e EMA) GetTimestamp() time.Time    { return e.Timestamp }
func (e EMA) GetIndicatorName() string   { return e.IndicatorName } // Added

func (m MACD) GetInstrumentToken() uint32 { return m.InstrumentToken }
func (m MACD) GetInterval() string        { return m.Interval }
func (m MACD) GetTimestamp() time.Time    { return m.Timestamp }
func (m MACD) GetIndicatorName() string   { return m.IndicatorName } // Added

func (a ATR) GetInstrumentToken() uint32 { return a.InstrumentToken }
func (a ATR) GetInterval() string        { return a.Interval }
func (a ATR) GetTimestamp() time.Time    { return a.Timestamp }
func (a ATR) GetIndicatorName() string   { return a.IndicatorName } // Added

func (r RSI) GetInstrumentToken() uint32 { return r.InstrumentToken }
func (r RSI) GetInterval() string        { return r.Interval }
func (r RSI) GetTimestamp() time.Time    { return r.Timestamp }
func (r RSI) GetIndicatorName() string   { return r.IndicatorName } // Added

func (s Stochastic) GetInstrumentToken() uint32 { return s.InstrumentToken }
func (s Stochastic) GetInterval() string        { return s.Interval }
func (s Stochastic) GetTimestamp() time.Time    { return s.Timestamp }
func (s Stochastic) GetIndicatorName() string   { return s.IndicatorName } // Added

func (b BollingerBands) GetInstrumentToken() uint32 { return b.InstrumentToken }
func (b BollingerBands) GetInterval() string        { return b.Interval }
func (b BollingerBands) GetTimestamp() time.Time    { return b.Timestamp }
func (b BollingerBands) GetIndicatorName() string   { return b.IndicatorName } // Added

func (o OBV) GetInstrumentToken() uint32 { return o.InstrumentToken }
func (o OBV) GetInterval() string        { return o.Interval }
func (o OBV) GetTimestamp() time.Time    { return o.Timestamp }
func (o OBV) GetIndicatorName() string   { return o.IndicatorName } // Added

func (v VWAP) GetInstrumentToken() uint32 { return v.InstrumentToken }
func (v VWAP) GetInterval() string        { return v.Interval }
func (v VWAP) GetTimestamp() time.Time    { return v.Timestamp }
func (v VWAP) GetIndicatorName() string   { return v.IndicatorName } // Added

func (a ADX) GetInstrumentToken() uint32 { return a.InstrumentToken }
func (a ADX) GetInterval() string        { return a.Interval }
func (a ADX) GetTimestamp() time.Time    { return a.Timestamp }
func (a ADX) GetIndicatorName() string   { return a.IndicatorName } // Added
