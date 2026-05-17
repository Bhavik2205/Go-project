package db

import (
	"encoding/json"
	"time"
)

// UserSetting stores per-user, per-section settings as JSONB.
// Corresponds to migration 000012.
type UserSetting struct {
	ID           uint `gorm:"primaryKey"`
	CreatedAt    time.Time
	UpdatedAt    time.Time
	DeletedAt    *time.Time      `gorm:"index"`
	UserID       uint            `gorm:"not null;index"`
	Section      string          `gorm:"not null;size:100"`
	SettingsJSON json.RawMessage `gorm:"type:jsonb;not null;default:'{}'"`
}

func (UserSetting) TableName() string { return "user_settings" }

// Watchlist stores a named watchlist per user.
// Corresponds to migration 000013.
type Watchlist struct {
	ID        uint `gorm:"primaryKey"`
	CreatedAt time.Time
	UpdatedAt time.Time
	DeletedAt *time.Time      `gorm:"index"`
	UserID    uint            `gorm:"not null;index"`
	Name      string          `gorm:"not null;size:255;default:'Default'"`
	Items     []WatchlistItem `gorm:"foreignKey:WatchlistID"`
}

func (Watchlist) TableName() string { return "watchlists" }

// WatchlistItem stores a single instrument in a watchlist.
type WatchlistItem struct {
	ID              uint `gorm:"primaryKey"`
	CreatedAt       time.Time
	UpdatedAt       time.Time
	DeletedAt       *time.Time `gorm:"index"`
	WatchlistID     uint       `gorm:"not null;index"`
	InstrumentToken uint       `gorm:"not null"`
	Symbol          string     `gorm:"not null;size:255"`
}

func (WatchlistItem) TableName() string { return "watchlist_items" }

// BacktestJob stores an async backtest job submission and its result.
// Corresponds to migration 000014.
type BacktestJob struct {
	ID             uint `gorm:"primaryKey"`
	CreatedAt      time.Time
	UpdatedAt      time.Time
	DeletedAt      *time.Time      `gorm:"index"`
	UserID         uint            `gorm:"not null;index"`
	StrategyName   string          `gorm:"not null;size:255"`
	Symbols        json.RawMessage `gorm:"type:jsonb;not null;default:'[]'"`
	FromTime       time.Time       `gorm:"not null"`
	ToTime         time.Time       `gorm:"not null"`
	InitialCapital float64         `gorm:"type:numeric;not null;default:100000"`
	FeesConfig     json.RawMessage `gorm:"type:jsonb;not null;default:'{}'"`
	Parameters     json.RawMessage `gorm:"type:jsonb;not null;default:'{}'"`
	Status         string          `gorm:"not null;size:50;default:'PENDING'"`
	Result         json.RawMessage `gorm:"type:jsonb"`
	ErrorMessage   string          `gorm:"type:text"`
	StartedAt      *time.Time
	CompletedAt    *time.Time
}

func (BacktestJob) TableName() string { return "backtest_jobs" }

// NotificationChannel stores per-user, per-channel notification config.
// Corresponds to migration 000015.
type NotificationChannel struct {
	ID          uint `gorm:"primaryKey"`
	CreatedAt   time.Time
	UpdatedAt   time.Time
	DeletedAt   *time.Time      `gorm:"index"`
	UserID      uint            `gorm:"not null;index"`
	ChannelType string          `gorm:"not null;size:50"`
	IsEnabled   bool            `gorm:"not null;default:false"`
	Config      json.RawMessage `gorm:"type:jsonb;not null;default:'{}'"`
}

func (NotificationChannel) TableName() string { return "notification_channels" }

// NotificationHistory stores a delivery log for each notification attempt.
type NotificationHistory struct {
	ID                uint `gorm:"primaryKey"`
	CreatedAt         time.Time
	UserID            uint   `gorm:"not null;index"`
	ChannelType       string `gorm:"not null;size:50"`
	EventType         string `gorm:"not null;size:100"`
	Message           string `gorm:"type:text;not null"`
	Status            string `gorm:"not null;size:50;default:'PENDING'"`
	ProviderMessageID string `gorm:"size:255"`
	ErrorMessage      string `gorm:"type:text"`
	SentAt            *time.Time
}

func (NotificationHistory) TableName() string { return "notification_history" }

// AuditEvent stores an immutable audit trail for sensitive actions.
// Corresponds to migration 000016.
type AuditEvent struct {
	ID           uint            `gorm:"primaryKey"`
	CreatedAt    time.Time       `gorm:"index"`
	UserID       *uint           `gorm:"index"`
	EventType    string          `gorm:"not null;size:100;index"`
	ResourceType string          `gorm:"size:100"`
	ResourceID   string          `gorm:"size:255"`
	Action       string          `gorm:"not null;size:50"`
	Status       string          `gorm:"not null;size:50;default:'SUCCESS'"`
	IPAddress    string          `gorm:"size:45"`
	UserAgent    string          `gorm:"type:text"`
	RequestID    string          `gorm:"size:255"`
	Metadata     json.RawMessage `gorm:"type:jsonb"`
	ErrorMessage string          `gorm:"type:text"`
}

func (AuditEvent) TableName() string { return "audit_events" }

// OptionSnapshot stores real-time option market data + Greeks.
// Corresponds to migration 000017.
type OptionSnapshot struct {
	InstrumentToken   uint32    `gorm:"primaryKey;column:instrument_token"`
	Timestamp         time.Time `gorm:"primaryKey;column:timestamp;type:timestamptz"`
	LastPrice         float64   `gorm:"type:numeric;not null"`
	BidPrice          *float64  `gorm:"type:numeric"`
	AskPrice          *float64  `gorm:"type:numeric"`
	Volume            *int
	OpenInterest      *int
	ImpliedVolatility *float64 `gorm:"type:numeric"` // IV in percentage
	Delta             *float64 `gorm:"type:numeric"`
	Gamma             *float64 `gorm:"type:numeric"`
	Theta             *float64 `gorm:"type:numeric"`
	Vega              *float64 `gorm:"type:numeric"`
	Rho               *float64 `gorm:"type:numeric"`
	UnderlyingPrice   *float64 `gorm:"type:numeric"`
	TheoreticalPrice  *float64 `gorm:"type:numeric"`
}

func (OptionSnapshot) TableName() string { return "option_snapshots" }
