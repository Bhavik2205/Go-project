// internal/db/models.go
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
	InstrumentToken uint   `gorm:"uniqueIndex;not null"`                                // Unique ID for the instrument (broker specific)
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
	// gorm.Model is typically NOT used for hypertables as it adds an 'ID' column
	// that might conflict with the time-series nature and compound primary keys.
	// We'll define primary key explicitly.
	InstrumentToken uint       `gorm:"primaryKey;not null"` // Composite primary key part
	Instrument      Instrument `gorm:"foreignKey:InstrumentToken"`
	Timestamp       time.Time  `gorm:"primaryKey;not null;type:timestamp with time zone"` // Composite primary key part, Time-series dimension for TimescaleDB
	Open            float64    `gorm:"not null"`
	High            float64    `gorm:"not null"`
	Low             float64    `gorm:"not null"`
	Close           float64    `gorm:"not null"`
	Volume          float64    `gorm:"not null"`
	// Technical Indicators (examples, add more as needed)
	SMA20      float64
	RSI14      float64
	MACD       float64
	MACDSignal float64
	MACDHist   float64
	// Add more indicators here...
}

// Order represents a placed order (buy/sell request) by the bot.
type Order struct {
	gorm.Model
	UserID          uint       `gorm:"not null"`
	User            User       `gorm:"foreignKey:UserID"`
	InstrumentToken uint       `gorm:"not null"`
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
	InstrumentToken uint      `gorm:"not null"`             // Redundant for querying/indexing
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
	InstrumentToken uint       `gorm:"not null"`
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
