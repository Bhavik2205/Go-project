package observability

import (
	"context"
	"sync/atomic"
	"time"
)

var (
	// Use atomic.Int64 for thread-safe tick timestamp updates
	// Stores unix nanoseconds for high precision
	lastTickTimeNano atomic.Int64
)

// UpdateLastTickTimestamp records when the last tick was received.
// Call this in the OnTick handler. Thread-safe.
func UpdateLastTickTimestamp(t time.Time) {
	lastTickTimeNano.Store(t.UnixNano())
	LastTickTimestamp.Set(float64(t.Unix()))
}

// UpdateTickStalenessStatus checks if no ticks have been received for >5s.
// Call this periodically from a background worker. Thread-safe.
func UpdateTickStalenessStatus() {
	nanoVal := lastTickTimeNano.Load()
	if nanoVal == 0 {
		TickFeedDead.Set(1)
		return
	}

	lastTickTime := time.Unix(0, nanoVal)
	since := time.Since(lastTickTime)
	if since > 5*time.Second {
		TickFeedDead.Set(1)
	} else {
		TickFeedDead.Set(0)
	}
}

// RecordTickLag records the latency from Zerodha timestamp to processing.
// latencyMs should be in milliseconds.
func RecordTickLag(latencyMs float64) {
	TickLag.Observe(latencyMs)
}

// RecordCandleLatency records the latency from candle EndTime to finalization.
// latencyMs should be in milliseconds.
func RecordCandleLatency(latencyMs float64) {
	CandleLatency.Observe(latencyMs)
	CandleFinalized.Inc()
}

// RecordIndicatorLatency records the time to compute indicators.
// latencyMs should be in milliseconds.
func RecordIndicatorLatency(latencyMs float64) {
	IndicatorLatency.Observe(latencyMs)
}

// RecordDBError increments the database error counter.
func RecordDBError() {
	DBErrors.Inc()
}

// RecordDBFlushDrop increments the database flush drop counter.
func RecordDBFlushDrop() {
	DBFlushDrops.Inc()
}

// RecordIndicatorError increments the indicator error counter.
func RecordIndicatorError() {
	IndicatorErrors.Inc()
}

// RecordTickGap increments the tick gap counter.
// Call when a gap in tick sequence is detected.
func RecordTickGap() {
	TickGapsDetected.Inc()
}

// RecordDuplicateTick increments the duplicate tick counter.
// Call when a duplicate tick is detected.
func RecordDuplicateTick() {
	DuplicateTicksDetected.Inc()
}

// RecordOutOfOrderTick increments the out-of-order tick counter.
func RecordOutOfOrderTick() {
	OutOfOrderTicksDetected.Inc()
}

// RecordLateTick increments the late tick counter.
// Call when a tick arrives after its candle has already closed.
func RecordLateTick() {
	LateTicks.Inc()
}

// RecordCandleDrop increments the candle drop counter.
func RecordCandleDrop() {
	CandleDrops.Inc()
}

// RecordCandleFinalizationFailure increments the finalization failure counter.
func RecordCandleFinalizationFailure() {
	CandleFinalizationFailures.Inc()
}

// RecordCandleRebuild increments the candle rebuild counter.
func RecordCandleRebuild() {
	CandleRebuilds.Inc()
}

// RecordIndicatorComputed increments the indicators computed counter.
func RecordIndicatorComputed() {
	IndicatorsComputed.Inc()
}

// SetDBUp sets the DB reachability gauge. Call from your health-check loop.
func SetDBUp(up bool) {
	if up {
		DBUp.Set(1)
	} else {
		DBUp.Set(0)
	}
}

// SetRedisUp sets the Redis reachability gauge. Call from your health-check loop.
func SetRedisUp(up bool) {
	if up {
		RedisUp.Set(1)
	} else {
		RedisUp.Set(0)
	}
}

// StartTickStalenessMonitor starts a background goroutine that monitors tick staleness
// and updates the TickFeedDead gauge. Ticks are considered stale if no ticks have been
// received for more than 5 seconds.
func StartTickStalenessMonitor(ctx context.Context) {
go func() {
ticker := time.NewTicker(1 * time.Second)
defer ticker.Stop()

for {
select {
case <-ticker.C:
UpdateTickStalenessStatus()
case <-ctx.Done():
return
}
}
}()
}
