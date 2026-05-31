package db

import (
	"time"

	"gorm.io/gorm" // For gorm.Model and soft deletes
)

// User represents a user of the trading bot.
// This table stores core user authentication information.
type User struct {
	gorm.Model
	Email        string `gorm:"uniqueIndex;not null"` // User's unique email, used for login
	PasswordHash string `gorm:"not null"`             // Hashed password for security
	UserName     string
	IsActive     bool `gorm:"default:true"` // Account status
}

// UserBrokerAccount stores details for a user's linked broker account (e.g., Zerodha).
// Crucial for multi-user and multi-broker support. Access token is encrypted.
type UserBrokerAccount struct {
	gorm.Model
	UserID        uint      `gorm:"not null"` // Foreign key to User
	User          User      `gorm:"foreignKey:UserID"`
	BrokerType    string    `gorm:"not null"` // e.g., "ZERODHA", "FYERS", "ANGEL_ONE"
	APIKey        string    `gorm:"not null"` // Broker's public API key for this user
	AccessToken   []byte    `gorm:"not null"` // Encrypted access token (binary data)
	PublicToken   []byte    `gorm:"not null"` // Encrypted public token (binary data)
	RequestToken  string    // Temporary token from OAuth flow
	SessionExpiry time.Time // When the broker session expires
	IsActive      bool      `gorm:"default:true"` // Whether this broker account is active for trading
	AccountName   string    // e.g., "Zerodha - My Personal Account"
	BrokerUserID  string    // The user ID provided by the broker (e.g., Kite user ID)
}

// Instrument stores details about tradable instruments (stocks, indices, etc.).
// This is periodically refreshed from the broker.
type Instrument struct {
	gorm.Model
	InstrumentToken uint32 `gorm:"uniqueIndex;not null"`                                // Unique ID for the instrument (broker specific)
	Exchange        string `gorm:"not null"`                                            // e.g., "NSE", "BSE"
	Tradingsymbol   string `gorm:"uniqueIndex:idx_instrument_symbol_exchange;not null"` // Unique trading symbol (e.g., "RELIANCE")
	InstrumentType  string // e.g., "EQ" for equity, "IND" for index
	Name            string // Full name of the instrument (e.g., "Reliance Industries Ltd")
	Segment         string // e.g., "NSECM" for NSE Capital Market
	TickSize        float64
	LotSize         int
	Expiry          *time.Time // For F&O instruments
	Strike          *float64   // For F&O instruments
	OptionType      string     // "CE" (Call) or "PE" (Put) for F&O
	LastUpdated     time.Time  // Timestamp of the last time this instrument was updated
}

// MarketData stores aggregated OHLCV (Open-High-Low-Close-Volume) and technical indicator data.
// This table is designed to be a TimescaleDB hypertable for efficient time-series queries.
type MarketData struct {
	// Primary Keys for Time-Series Hypertable
	// These three fields form a composite primary key to ensure uniqueness for each tick.
	// Aligned InstrumentToken to uint32 as per kitemodels.Tick
	InstrumentToken uint32    `gorm:"primaryKey;not null"`
	Timestamp       time.Time `gorm:"primaryKey;not null;type:timestamptz"` // Exchange timestamp from kitemodels.Tick.Timestamp
	TickSequenceID  int       `gorm:"primaryKey;not null"`                  // Custom generated to ensure uniqueness at same timestamp

	// --- Core Tick Data (from kitemodels.Tick) ---
	LastPrice          float64 `gorm:"not null;type:numeric"`
	LastTradedQuantity uint32  `gorm:"not null"` // Aligned to uint32 as per kitemodels.Tick.LastTradedQuantity
	Volume             uint32  `gorm:"not null"` // Aligned to uint32 as per kitemodels.Tick.VolumeTraded
	AverageTradePrice  float64 `gorm:"not null;type:numeric"`
	NetChange          float64 `gorm:"not null;type:numeric"`

	// Daily OHLC (from kitemodels.Tick.OHLC)
	// These are typically the daily Open, High, Low, Close values.
	Open  float64 `gorm:"not null;type:numeric"`
	High  float64 `gorm:"not null;type:numeric"`
	Low   float64 `gorm:"not null;type:numeric"`
	Close float64 `gorm:"not null;type:numeric"`

	// Open Interest (from kitemodels.Tick.OI - will be 0 for equities)
	OpenInterest uint32 `gorm:"not null"` // Aligned to uint32 as per kitemodels.Tick.OI

	// --- Market Depth (Level 1 - Top 5 Bids and Asks, from kitemodels.Tick.Depth) ---
	// All quantities and orders are uint32 as per kitemodels.DepthItem

	// Bid Side (buyers)
	BidPrice1    float64 `gorm:"not null;type:numeric"`
	BidQuantity1 uint32  `gorm:"not null"`
	BidOrders1   uint32  `gorm:"not null"`

	BidPrice2    float64 `gorm:"not null;type:numeric"`
	BidQuantity2 uint32  `gorm:"not null"`
	BidOrders2   uint32  `gorm:"not null"`

	BidPrice3    float64 `gorm:"not null;type:numeric"`
	BidQuantity3 uint32  `gorm:"not null"`
	BidOrders3   uint32  `gorm:"not null"`

	BidPrice4    float64 `gorm:"not null;type:numeric"`
	BidQuantity4 uint32  `gorm:"not null"`
	BidOrders4   uint32  `gorm:"not null"`

	BidPrice5    float64 `gorm:"not null;type:numeric"`
	BidQuantity5 uint32  `gorm:"not null"`
	BidOrders5   uint32  `gorm:"not null"`

	// Ask Side (sellers)
	AskPrice1    float64 `gorm:"not null;type:numeric"`
	AskQuantity1 uint32  `gorm:"not null"`
	AskOrders1   uint32  `gorm:"not null"`

	AskPrice2    float64 `gorm:"not null;type:numeric"`
	AskQuantity2 uint32  `gorm:"not null"`
	AskOrders2   uint32  `gorm:"not null"`

	AskPrice3    float64 `gorm:"not null;type:numeric"`
	AskQuantity3 uint32  `gorm:"not null"`
	AskOrders3   uint32  `gorm:"not null"`

	AskPrice4    float64 `gorm:"not null;type:numeric"`
	AskQuantity4 uint32  `gorm:"not null"`
	AskOrders4   uint32  `gorm:"not null"`

	AskPrice5    float64 `gorm:"not null;type:numeric"`
	AskQuantity5 uint32  `gorm:"not null"`
	AskOrders5   uint32  `gorm:"not null"`

	// --- Other aggregated quantities (from kitemodels.Tick) ---
	TotalBuyQuantity  uint32 `gorm:"not null"` // Total aggregated buy quantity across all price levels
	TotalSellQuantity uint32 `gorm:"not null"` // Total aggregated sell quantity across all price levels

	// --- Fields from kitemodels.Tick that are typically NOT stored per tick in raw data ---
	// LastTradeTime      Time   // If needed, can add, but Timestamp is usually sufficient
	// Mode               string // More of a metadata field for the tick type itself
	// IsTradable         bool   // Instrument metadata, better in a separate instruments table
	// IsIndex            bool   // Instrument metadata, better in a separate instruments table
	// OIDayHigh          uint32 // Daily high/low for OI, typically derived or not per tick
	// OIDayLow           uint32 // Daily high/low for OI, typically derived or not per tick
}

// OHLCVCandle stores aggregated Open-High-Low-Close-Volume data for specific intervals.
// This will be a TimescaleDB hypertable partitionable by instrument and interval.
type OHLCVCandle struct {
	InstrumentToken uint32    `gorm:"primaryKey;column:instrument_token"`
	Interval        string    `gorm:"primaryKey;column:interval"`                                // e.g., "1m", "5m", "1h", "1d"
	Timestamp       time.Time `gorm:"primaryKey;column:timestamp;type:timestamp with time zone"` // Start time of the candle
	Open            float64   `gorm:"not null"`
	High            float64   `gorm:"not null"`
	Low             float64   `gorm:"not null"`
	Close           float64   `gorm:"not null"`
	Volume          float64   `gorm:"not null"` // Volume for the candle duration
	TradeCount      uint32    // Number of trades in this candle (optional)
	CreatedAt       time.Time `gorm:"autoCreateTime"`
	UpdatedAt       time.Time `gorm:"autoUpdateTime"`
}

// IndicatorSMA model
type IndicatorSMA struct {
	IndicatorName   string    `gorm:"-" json:"indicator_name"` // Added: For WebSocket identification
	InstrumentToken uint32    `gorm:"primaryKey;column:instrument_token"`
	Interval        string    `gorm:"primaryKey;column:interval"`                                // e.g., "1m", "5m"
	Period          int       `gorm:"primaryKey;column:period"`                                  // e.g., 20
	Timestamp       time.Time `gorm:"primaryKey;column:timestamp;type:timestamp with time zone"` // Timestamp of the candle for which this SMA is calculated
	Value           float64   `gorm:"not null;type:numeric"`
	CreatedAt       time.Time `gorm:"autoCreateTime"`
	UpdatedAt       time.Time `gorm:"autoUpdateTime"`
}

// IndicatorEMA model
type IndicatorEMA struct {
	IndicatorName   string    `gorm:"-" json:"indicator_name"` // Added
	InstrumentToken uint32    `gorm:"primaryKey;column:instrument_token"`
	Interval        string    `gorm:"primaryKey;column:interval"`
	Period          int       `gorm:"primaryKey;column:period"`
	Timestamp       time.Time `gorm:"primaryKey;column:timestamp;type:timestamp with time zone"`
	Value           float64   `gorm:"not null;type:numeric"`
	CreatedAt       time.Time `gorm:"autoCreateTime"`
	UpdatedAt       time.Time `gorm:"autoUpdateTime"`
}

// IndicatorMACD model
type IndicatorMACD struct {
	IndicatorName   string    `gorm:"-" json:"indicator_name"` // Added
	InstrumentToken uint32    `gorm:"primaryKey;column:instrument_token"`
	Interval        string    `gorm:"primaryKey;column:interval"`
	FastPeriod      int       `gorm:"primaryKey;column:fast_period"`
	SlowPeriod      int       `gorm:"primaryKey;column:slow_period"`
	SignalPeriod    int       `gorm:"primaryKey;column:signal_period"`
	Timestamp       time.Time `gorm:"primaryKey;column:timestamp;type:timestamp with time zone"`
	MACDLine        float64   `gorm:"not null;type:numeric"`
	SignalLine      float64   `gorm:"not null;type:numeric"`
	Histogram       float64   `gorm:"not null;type:numeric"`
	CreatedAt       time.Time `gorm:"autoCreateTime"`
	UpdatedAt       time.Time `gorm:"autoUpdateTime"`
}

// IndicatorATR model
type IndicatorATR struct {
	IndicatorName   string    `gorm:"-" json:"indicator_name"` // Added
	InstrumentToken uint32    `gorm:"primaryKey;column:instrument_token"`
	Interval        string    `gorm:"primaryKey;column:interval"`
	Period          int       `gorm:"primaryKey;column:period"`
	Timestamp       time.Time `gorm:"primaryKey;column:timestamp;type:timestamp with time zone"`
	Value           float64   `gorm:"not null;type:numeric"`
	CreatedAt       time.Time `gorm:"autoCreateTime"`
	UpdatedAt       time.Time `gorm:"autoUpdateTime"`
}

// IndicatorRSI model
type IndicatorRSI struct {
	IndicatorName   string    `gorm:"-" json:"indicator_name"` // Added
	InstrumentToken uint32    `gorm:"primaryKey;column:instrument_token"`
	Interval        string    `gorm:"primaryKey;column:interval"`
	Period          int       `gorm:"primaryKey;column:period"`
	Timestamp       time.Time `gorm:"primaryKey;column:timestamp;type:timestamp with time zone"`
	Value           float64   `gorm:"not null;type:numeric"`
	CreatedAt       time.Time `gorm:"autoCreateTime"`
	UpdatedAt       time.Time `gorm:"autoUpdateTime"`
}

// IndicatorStochastic model
type IndicatorStochastic struct {
	IndicatorName   string    `gorm:"-" json:"indicator_name"` // Added
	InstrumentToken uint32    `gorm:"primaryKey;column:instrument_token"`
	Interval        string    `gorm:"primaryKey;column:interval"`
	KPeriod         int       `gorm:"primaryKey;column:k_period"`
	DPeriod         int       `gorm:"primaryKey;column:d_period"`
	Timestamp       time.Time `gorm:"primaryKey;column:timestamp;type:timestamp with time zone"`
	KValue          float64   `gorm:"not null;type:numeric"`
	DValue          float64   `gorm:"not null;type:numeric"`
	CreatedAt       time.Time `gorm:"autoCreateTime"`
	UpdatedAt       time.Time `gorm:"autoUpdateTime"`
}

// IndicatorBollingerBands model
type IndicatorBollingerBands struct {
	IndicatorName   string    `gorm:"-" json:"indicator_name"` // Added
	InstrumentToken uint32    `gorm:"primaryKey;column:instrument_token"`
	Interval        string    `gorm:"primaryKey;column:interval"`
	Period          int       `gorm:"primaryKey;column:period"`
	NumStdDev       float64   `gorm:"primaryKey;column:num_std_dev;type:numeric(5,2)"` // Use specific numeric type for std dev if needed
	Timestamp       time.Time `gorm:"primaryKey;column:timestamp;type:timestamp with time zone"`
	UpperBand       float64   `gorm:"not null;type:numeric"`
	MiddleBand      float64   `gorm:"not null;type:numeric"`
	LowerBand       float64   `gorm:"not null;type:numeric"`
	CreatedAt       time.Time `gorm:"autoCreateTime"`
	UpdatedAt       time.Time `gorm:"autoUpdateTime"`
}

// IndicatorOBV model
type IndicatorOBV struct {
	IndicatorName   string    `gorm:"-" json:"indicator_name"` // Added
	InstrumentToken uint32    `gorm:"primaryKey;column:instrument_token"`
	Interval        string    `gorm:"primaryKey;column:interval"`
	Timestamp       time.Time `gorm:"primaryKey;column:timestamp;type:timestamp with time zone"`
	Value           float64   `gorm:"not null;type:numeric"`
	CreatedAt       time.Time `gorm:"autoCreateTime"`
	UpdatedAt       time.Time `gorm:"autoUpdateTime"`
}

// IndicatorVWAP model (VWAP is typically period-agnostic but reset daily, so no 'Period' column)
type IndicatorVWAP struct {
	IndicatorName   string    `gorm:"-" json:"indicator_name"` // Added
	InstrumentToken uint32    `gorm:"primaryKey;column:instrument_token"`
	Interval        string    `gorm:"primaryKey;column:interval"`
	Timestamp       time.Time `gorm:"primaryKey;column:timestamp;type:timestamp with time zone"` // Timestamp of the candle
	Value           float64   `gorm:"not null;type:numeric"`
	CreatedAt       time.Time `gorm:"autoCreateTime"`
	UpdatedAt       time.Time `gorm:"autoUpdateTime"`
}

// IndicatorADX model (assuming it's implemented)
type IndicatorADX struct {
	IndicatorName   string    `gorm:"-" json:"indicator_name"` // Added
	InstrumentToken uint32    `gorm:"primaryKey;column:instrument_token"`
	Interval        string    `gorm:"primaryKey;column:interval"`
	Period          int       `gorm:"primaryKey;column:period"`
	Timestamp       time.Time `gorm:"primaryKey;column:timestamp;type:timestamp with time zone"`
	ADXValue        float64   `gorm:"not null;type:numeric"`
	PlusDI          float64   `gorm:"not null;type:numeric"`
	MinusDI         float64   `gorm:"not null;type:numeric"`
	CreatedAt       time.Time `gorm:"autoCreateTime"`
	UpdatedAt       time.Time `gorm:"autoUpdateTime"`
}

// Order represents a placed order (buy/sell request) by the bot.
type Order struct {
	gorm.Model
	UserID          uint       `gorm:"not null"`
	User            User       `gorm:"foreignKey:UserID"`
	InstrumentToken uint32     `gorm:"not null"`
	Instrument      Instrument `gorm:"foreignKey:InstrumentToken"`
	BrokerOrderID   string     `gorm:"uniqueIndex;not null"` // Unique ID from the broker for this order
	StrategyName    string     `gorm:"not null"`             // e.g., "IntradayMomentum"
	OrderType       string     `gorm:"not null"`             // e.g., "MARKET", "LIMIT", "SL", "SL-M"
	TransactionType string     `gorm:"not null"`             // "BUY" or "SELL"
	Quantity        int        `gorm:"not null"`
	Price           float64    // Limit price if applicable
	TriggerPrice    float64    // Stop loss trigger price if applicable
	Status          string     `gorm:"not null"` // e.g., "PENDING", "OPEN", "FILLED", "CANCELLED", "REJECTED"
	PlacedAt        time.Time  `gorm:"not null"` // Timestamp when order was placed
	FilledQuantity  int
	FilledPrice     float64    // Average filled price
	ValidUntil      *time.Time // For GTT orders or day validity
	Product         string     // e.g., "MIS", "CNC", "NRML"
	ExchangeOrderID string     // Exchange specific order ID
	Tag             string     // Optional tag for tracking
}

// Trade represents a filled order or a part of a filled order (an execution).
// A single Order can result in multiple Trades if partially filled.
type Trade struct {
	gorm.Model
	OrderID         uint      `gorm:"not null"` // Foreign key to Order
	Order           Order     `gorm:"foreignKey:OrderID"`
	UserID          uint      `gorm:"not null"`             // Redundant for querying/indexing
	InstrumentToken uint32    `gorm:"not null"`             // Redundant for querying/indexing
	TradeID         string    `gorm:"uniqueIndex;not null"` // Unique ID for this trade (broker specific)
	TransactionType string    `gorm:"not null"`             // "BUY" or "SELL"
	Quantity        int       `gorm:"not null"`
	Price           float64   `gorm:"not null"` // Price at which this specific trade was executed
	TradeTime       time.Time `gorm:"not null"` // Timestamp of the trade execution
	Exchange        string    // Exchange on which the trade happened
}

// Position represents a user's current holding for an instrument.
// This is typically updated based on trades and reconciled with broker's live positions.
type Position struct {
	gorm.Model
	UserID          uint       `gorm:"not null"`
	User            User       `gorm:"foreignKey:UserID"`
	InstrumentToken uint32     `gorm:"not null"`
	Instrument      Instrument `gorm:"foreignKey:InstrumentToken"`
	TradingSymbol   string     `gorm:"not null"`
	Product         string     `gorm:"not null"` // e.g., "MIS", "CNC", "NRML"
	Quantity        int        `gorm:"not null"` // Current net quantity (positive for long, negative for short)
	AveragePrice    float64    `gorm:"not null"` // Average entry price
	LastPrice       float64    // Last traded price (for P&L calculation)
	RealizedPnL     float64    // Profit/Loss from closed positions
	UnrealizedPnL   float64    // Current P&L for open positions
	UpdatedAt       time.Time  `gorm:"not null"` // Timestamp of last update
}

// NewsArticle stores fetched news content and its metadata.
type NewsArticle struct {
	gorm.Model
	Source         string `gorm:"not null"` // e.g., "NewsAPI", "MarketAux"
	Title          string `gorm:"not null"`
	Description    string
	Content        string    `gorm:"type:text"` // Use text type for potentially long content
	PublishedAt    time.Time `gorm:"not null"`
	URL            string    `gorm:"uniqueIndex;not null"` // URL to the original article
	ImageURL       string
	SentimentScore float64   // Numerical sentiment score (e.g., -1.0 to 1.0)
	SentimentLabel string    // e.g., "Positive", "Neutral", "Negative"
	AnalyzedAt     time.Time // Timestamp when sentiment analysis was performed
}

// UserStrategy stores which strategies a user has enabled and their specific parameters.
type UserStrategy struct {
	gorm.Model
	UserID       uint   `gorm:"not null"` // Foreign key to User
	User         User   `gorm:"foreignKey:UserID"`
	StrategyName string `gorm:"not null"` // e.g., "IntradayMomentum", "Scalping"
	IsEnabled    bool   `gorm:"default:false"`
	Parameters   []byte `gorm:"type:jsonb"` // Store strategy-specific JSON parameters (e.g., {"rsi_period": 14, "sma_short": 20})
	LastUpdated  time.Time
}

// Metric represents a custom application metric logged to the database (for auditing/reporting).
// Note: This is separate from Prometheus metrics. Prometheus is for time-series health/performance.
// This could be for business-level metrics or audit trails.
type Metric struct {
	gorm.Model
	Name      string    `gorm:"not null"`
	Value     float64   `gorm:"not null"`
	Timestamp time.Time `gorm:"not null"`
	Labels    []byte    `gorm:"type:jsonb"` // Additional key-value pairs as JSONB (e.g., {"user_id": 123, "instrument": "RELIANCE"})
}
