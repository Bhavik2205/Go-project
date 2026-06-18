package candles

import (
	"sync"
	"time"
)

// OpenCandle represents an in‑memory candle that is still being updated.
type OpenCandle struct {
	InstrumentToken uint32
	Interval        time.Duration // parsed duration from string (e.g., 1*time.Minute)
	IntervalStr     string        // original string like "1m"
	StartTime       time.Time
	EndTime         time.Time
	Open            int64
	High            int64
	Low             int64
	Close           int64
	Volume          int64
	TradeCount      uint32
	LastTickTime    time.Time
	mu              sync.RWMutex
}
