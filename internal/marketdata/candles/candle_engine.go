package candles

import (
	"context"
	"math"
	"sync"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/marketdata"
	"github.com/Bhavik2205/ML-Bot/internal/observability"
	"go.uber.org/zap"
)

const (
	marketOpenHour    = 9
	marketOpenMinute  = 15
	marketCloseHour   = 15
	marketCloseMinute = 30
	marketTimezone    = "Asia/Kolkata"
)

// EngineConfig holds dependencies and callbacks for the CandleEngine.
type EngineConfig struct {
	Timezone              *time.Location           // Asia/Kolkata
	Intervals             []string                 // e.g., ["1m","5m"]
	GracePeriod           time.Duration            // e.g., 100ms
	FinalizeInterval      time.Duration            // e.g., 1s
	OnFinalize            func(candle *OpenCandle) // called when a candle is finalised (for DB, WebSocket, indicators)
	DBFlushChannel        chan<- interface{}       // optional, can be integrated into OnFinalize
	IndicatorInputChannel chan<- interface{}       // optional
	SimulationMode        bool                     // if true, skip market hours check and use tick time as is
}

// CandleEngine manages open candles, session‑aware bucketing and watermark finalisation.
type CandleEngine struct {
	cfg         *EngineConfig
	mu          sync.RWMutex
	openCandles map[uint32]map[time.Duration]*OpenCandle // token -> interval -> candle

	// precomputed interval durations and their string representations
	intervals   []time.Duration
	intervalStr map[time.Duration]string
}

// NewCandleEngine creates a new engine.
func NewCandleEngine(cfg *EngineConfig) *CandleEngine {
	intervals := make([]time.Duration, 0, len(cfg.Intervals))
	intervalStr := make(map[time.Duration]string)
	for _, s := range cfg.Intervals {
		d, err := parseDuration(s)
		if err != nil {
			zap.L().Warn("Invalid interval, skipping", zap.String("interval", s), zap.Error(err))
			continue
		}
		intervals = append(intervals, d)
		intervalStr[d] = s
	}
	return &CandleEngine{
		cfg:         cfg,
		openCandles: make(map[uint32]map[time.Duration]*OpenCandle),
		intervals:   intervals,
		intervalStr: intervalStr,
	}
}

// parseDuration supports "1s", "5m", "1h", "30s", "2m30s", etc.
func parseDuration(s string) (time.Duration, error) {
	// For "1h" we treat as 1 hour, but alignment is special later.
	return time.ParseDuration(s)
}

// StartFinalizer runs the watermark‑based finalisation loop.
func (e *CandleEngine) StartFinalizer(ctx context.Context) {
	ticker := time.NewTicker(e.cfg.FinalizeInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			e.finalizeByWatermark()
		case <-ctx.Done():
			return
		}
	}
}

// finalizeByWatermark checks all open candles and finalises those whose EndTime is <= now - grace period.
func (e *CandleEngine) finalizeByWatermark() {
	watermark := time.Now().Add(-e.cfg.GracePeriod)
	e.mu.Lock()
	defer e.mu.Unlock()

	for token, byInterval := range e.openCandles {
		for interval, candle := range byInterval {
			if !candle.EndTime.After(watermark) {
				// Finalise this candle
				e.finalizeCandleLocked(candle)
				delete(byInterval, interval)
			}
		}
		if len(byInterval) == 0 {
			delete(e.openCandles, token)
		}
	}

	// Update open candles gauge
	count := 0
	for _, byInterval := range e.openCandles {
		count += len(byInterval)
	}
	observability.OpenCandlesCount.Set(float64(count))
}

// finalizeCandleLocked calls the OnFinalize callback (must be called with lock held).
func (e *CandleEngine) finalizeCandleLocked(candle *OpenCandle) {
	latency := time.Since(candle.EndTime).Milliseconds()
	observability.RecordCandleLatency(float64(latency))
	
	if e.cfg.OnFinalize != nil {
		e.cfg.OnFinalize(candle)
	}
}

// ProcessTick updates open candles for all configured intervals based on a NormalizedTick.
func (e *CandleEngine) ProcessTick(tick *marketdata.NormalizedTick) {
	tickTime := tick.EventTime
	// If not in simulation mode AND market is closed, skip the tick
	if !e.cfg.SimulationMode && !e.isMarketOpen(tick.EventTime) {
		return
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	token := tick.InstrumentToken
	// Ensure map exists
	if _, ok := e.openCandles[token]; !ok {
		e.openCandles[token] = make(map[time.Duration]*OpenCandle)
	}

	for _, interval := range e.intervals {
		startTime := e.getCandleStartTime(tickTime, interval)
		if startTime.IsZero() {
			continue
		}
		endTime := e.getCandleEndTime(startTime, interval)

		candle, exists := e.openCandles[token][interval]
		if !exists || !candle.StartTime.Equal(startTime) {
			// If old candle exists with different start, finalise it first
			if exists {
				e.finalizeCandleLocked(candle)
			}
			// Create new candle
			candle = &OpenCandle{
				InstrumentToken: token,
				Interval:        interval,
				IntervalStr:     e.intervalStr[interval],
				StartTime:       startTime,
				EndTime:         endTime,
				Open:            tick.LastPrice,
				High:            tick.LastPrice,
				Low:             tick.LastPrice,
				Close:           tick.LastPrice,
				Volume:          float64(tick.LastTradedQuantity),
				TradeCount:      1,
				LastTickTime:    tickTime,
			}
			e.openCandles[token][interval] = candle
			continue
		}
		// Update existing candle
		candle.mu.Lock()
		if tick.LastPrice > candle.High {
			candle.High = tick.LastPrice
		}
		if tick.LastPrice < candle.Low {
			candle.Low = tick.LastPrice
		}
		candle.Close = tick.LastPrice
		candle.Volume += float64(tick.LastTradedQuantity)
		candle.TradeCount++
		candle.LastTickTime = tickTime
		candle.mu.Unlock()
	}
}

// isMarketOpen checks if tickTime is within market hours (9:15 – 15:30 IST).
func (e *CandleEngine) isMarketOpen(t time.Time) bool {
	marketTime := t.In(e.cfg.Timezone)
	open := time.Date(marketTime.Year(), marketTime.Month(), marketTime.Day(), 9, 15, 0, 0, e.cfg.Timezone)
	close := time.Date(marketTime.Year(), marketTime.Month(), marketTime.Day(), 15, 30, 0, 0, e.cfg.Timezone)
	return marketTime.After(open) && marketTime.Before(close)
}

// getCandleStartTime returns the start time of the candle bucket for a given tick and interval.
// For intervals < 1 hour: truncate to the interval within the market session (e.g., 9:15:00.000).
// For interval == 1 hour: align to market open hours (9:15, 10:15, … 15:15).
// For interval > 1 hour: treat as 1d candle (market open to close).
func (e *CandleEngine) getCandleStartTime(tickTime time.Time, interval time.Duration) time.Time {
	t := tickTime.In(e.cfg.Timezone)

	// -------------------------
	// Simulation Mode
	// -------------------------
	if e.cfg.SimulationMode {
		// Daily candle
		if interval > time.Hour {
			return time.Date(
				t.Year(),
				t.Month(),
				t.Day(),
				0, 0, 0, 0,
				e.cfg.Timezone,
			)
		}

		// Current candle bucket
		return t.Truncate(interval)
	}

	// -------------------------
	// Live Market Mode
	// -------------------------
	marketOpen := time.Date(
		t.Year(),
		t.Month(),
		t.Day(),
		marketOpenHour,
		marketOpenMinute,
		0,
		0,
		e.cfg.Timezone,
	)

	// Before market open
	if t.Before(marketOpen) {
		return time.Time{}
	}

	// Daily candle
	if interval > time.Hour {
		return marketOpen
	}

	// 1-hour candle aligned to market open
	if interval == time.Hour {
		hoursSinceOpen := int(t.Sub(marketOpen) / time.Hour)
		return marketOpen.Add(time.Duration(hoursSinceOpen) * time.Hour)
	}

	// Current bucket aligned from market open
	elapsed := t.Sub(marketOpen)
	bucket := elapsed / interval

	return marketOpen.Add(bucket * interval)
}

// nextCandleStart returns the start time of the next complete candle.
// If t exactly aligns with a boundary, that candle is considered "next".
func nextCandleStart(t, marketOpen time.Time, interval time.Duration) time.Time {
	if t.Before(marketOpen) {
		return marketOpen
	}
	elapsed := t.Sub(marketOpen)
	// Ceil division gives the number of intervals that have started or are exactly at the boundary
	n := int(math.Ceil(float64(elapsed) / float64(interval)))
	return marketOpen.Add(time.Duration(n) * interval)
}

// getCandleEndTime returns the end time for a given start time and interval.
func (e *CandleEngine) getCandleEndTime(startTime time.Time, interval time.Duration) time.Time {
	if interval > time.Hour {
		// Daily candle ends at market close (15:30)
		marketClose := time.Date(startTime.Year(), startTime.Month(), startTime.Day(),
			15, 30, 0, 0, e.cfg.Timezone)
		return marketClose
	}
	return startTime.Add(interval)
}

// FinalizeAll forces finalisation of all open candles (for graceful shutdown).
func (e *CandleEngine) FinalizeAll() {
	e.mu.Lock()
	defer e.mu.Unlock()
	for _, byInterval := range e.openCandles {
		for _, candle := range byInterval {
			e.finalizeCandleLocked(candle)
		}
	}
	e.openCandles = make(map[uint32]map[time.Duration]*OpenCandle)
}
