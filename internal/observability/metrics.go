// Package observability provides centralized Prometheus metrics and health monitoring.
package observability

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	// Counters
	TicksReceived = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "trading_ticks_received_total",
		Help: "Total number of ticks received from Zerodha",
	})

	TicksProcessed = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "trading_ticks_processed_total",
		Help: "Total number of ticks successfully processed",
	})

	TicksDropped = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "trading_ticks_dropped_total",
		Help: "Total number of ticks dropped due to backpressure",
	})

	PanicCounter = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "trading_panics_total",
		Help: "Total number of panics recovered in the system",
	})

	CandleFinalized = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "trading_candles_finalized_total",
		Help: "Total number of candles finalized",
	})

	// Histograms (latencies in milliseconds)
	TickLag = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "trading_tick_lag_ms",
		Help:    "Latency from Zerodha timestamp to processing (milliseconds)",
		Buckets: []float64{1, 5, 10, 25, 50, 100, 250, 500, 1000},
	})

	CandleLatency = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "trading_candle_latency_ms",
		Help:    "Latency from candle EndTime to finalization (milliseconds)",
		Buckets: []float64{1, 5, 10, 25, 50, 100, 250, 500, 1000},
	})

	IndicatorLatency = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "trading_indicator_latency_ms",
		Help:    "Time to compute indicators (milliseconds)",
		Buckets: []float64{1, 5, 10, 25, 50, 100, 250, 500, 1000},
	})

	// Gauges
	GoroutineCount = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "trading_goroutines_current",
		Help: "Current number of active goroutines",
	})

	MemoryHeapAlloc = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "trading_memory_heap_alloc_bytes",
		Help: "Heap allocations in bytes",
	})

	MemoryHeapInuse = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "trading_memory_heap_inuse_bytes",
		Help: "In-use heap memory in bytes",
	})

	MemorySys = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "trading_memory_sys_bytes",
		Help: "Total system memory in bytes",
	})

	GCRunsTotal = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "trading_gc_runs_total",
		Help: "Total number of GC runs",
	})

	TickQueueDepth = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "trading_tick_queue_depth",
		Help: "Current depth of tick broadcast queue",
	})

	TickQueueCapacity = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "trading_tick_queue_capacity",
		Help: "Total capacity of tick broadcast queue",
	})

	CandleQueueDepth = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "trading_candle_queue_depth",
		Help: "Current depth of candle finalization queue",
	})

	CandleQueueCapacity = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "trading_candle_queue_capacity",
		Help: "Total capacity of candle finalization queue",
	})

	DBFlushQueueDepth = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "trading_db_flush_queue_depth",
		Help: "Current depth of database flush queue",
	})

	DBFlushQueueCapacity = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "trading_db_flush_queue_capacity",
		Help: "Total capacity of database flush queue",
	})

	IndicatorQueueDepth = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "trading_indicator_queue_depth",
		Help: "Current depth of indicator processing queue",
	})

	IndicatorQueueCapacity = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "trading_indicator_queue_capacity",
		Help: "Total capacity of indicator processing queue",
	})

	LastTickTimestamp = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "trading_last_tick_timestamp_seconds",
		Help: "Unix timestamp of last received tick",
	})

	TickFeedDead = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "trading_tick_feed_dead",
		Help: "1 if no ticks received for >5s, 0 otherwise (staleness alert)",
	})

	OpenCandlesCount = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "trading_open_candles_count",
		Help: "Current number of open candles",
	})

	DBErrors = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "trading_db_errors_total",
		Help: "Total number of database write errors",
	})

	DBFlushDrops = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "trading_db_flush_drops_total",
		Help: "Total number of DB flush drops due to timeout",
	})

	WebSocketBroadcasted = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "trading_websocket_broadcasts_total",
		Help: "Total number of market data broadcasts to WebSocket clients",
	})

	IndicatorErrors = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "trading_indicator_errors_total",
		Help: "Total number of indicator computation errors",
	})

	// WAL metrics
	WALAppendsTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "wal_appends_total",
		Help: "Total number of ticks appended to the WAL",
	})

	WALAppendErrorsTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "wal_append_errors_total",
		Help: "Total number of WAL append errors",
	})

	WALBytesWrittenTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "wal_bytes_written_total",
		Help: "Total bytes written to WAL segments",
	})

	WALSegmentsTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "wal_segments_total",
		Help: "Total number of WAL segments created",
	})

	WALReplayRecordsTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "wal_replay_records_total",
		Help: "Total number of records replayed from WAL",
	})

	// Tick quality metrics (for market data validation)
	TickGapsDetected = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "trading_tick_gaps_detected_total",
		Help: "Total number of gaps detected in tick sequence",
	})

	DuplicateTicksDetected = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "trading_duplicate_ticks_detected_total",
		Help: "Total number of duplicate ticks detected",
	})

	OutOfOrderTicksDetected = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "trading_out_of_order_ticks_detected_total",
		Help: "Total number of out-of-order ticks detected",
	})
)

var (
	metricsOnce sync.Once
	initErr     error
)

// allMetrics is the single source of truth for registration.
// AlreadyRegisteredError is tolerated so tests and integration environments
// that call InitMetrics more than once do not break startup.
var allMetrics = []prometheus.Collector{
	TicksReceived, TicksProcessed, TicksDropped, PanicCounter, CandleFinalized,
	DBErrors, DBFlushDrops, WebSocketBroadcasted, IndicatorErrors,
	WALAppendsTotal, WALAppendErrorsTotal, WALBytesWrittenTotal, WALSegmentsTotal, WALReplayRecordsTotal,
	TickGapsDetected, DuplicateTicksDetected, OutOfOrderTicksDetected,
	TickLag, CandleLatency, IndicatorLatency,
	GoroutineCount, MemoryHeapAlloc, MemoryHeapInuse, MemorySys, GCRunsTotal,
	TickQueueDepth, TickQueueCapacity,
	CandleQueueDepth, CandleQueueCapacity,
	DBFlushQueueDepth, DBFlushQueueCapacity,
	IndicatorQueueDepth, IndicatorQueueCapacity,
	LastTickTimestamp, TickFeedDead, OpenCandlesCount,
}

// InitMetrics registers all Prometheus metrics.
// AlreadyRegisteredError is treated as success — safe to call from tests.
func InitMetrics() error {
	metricsOnce.Do(func() {
		for _, m := range allMetrics {
			if err := prometheus.Register(m); err != nil {
				if _, ok := err.(prometheus.AlreadyRegisteredError); !ok {
					initErr = err
					return
				}
			}
		}
	})
	return initErr
}
