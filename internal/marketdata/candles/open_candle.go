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
	Open            float64
	High            float64
	Low             float64
	Close           float64
	Volume          float64
	TradeCount      uint32
	LastTickTime    time.Time
	mu              sync.RWMutex
}
