package observability

import "github.com/prometheus/client_golang/prometheus"

// Execution and paper-trading metrics. Registered via allMetrics in metrics.go.
var (
	// ExecutionOrdersPlaced counts successfully placed orders by mode.
	ExecutionOrdersPlaced = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "execution_orders_placed_total",
		Help: "Orders successfully placed, labelled PAPER or LIVE",
	}, []string{"mode"})

	// ExecutionOrdersFailed counts broker placement failures.
	ExecutionOrdersFailed = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "execution_orders_failed_total",
		Help: "Orders the broker rejected or failed to place",
	})

	// ExecutionLatency measures time spent inside Broker.Place.
	ExecutionLatency = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "execution_place_duration_microseconds",
		Help:    "Time spent in Broker.Place",
		Buckets: []float64{1, 5, 10, 50, 100, 500, 1000, 10000, 100000, 1000000},
	})

	// KillSwitchTripped is 1 while order placement is halted. Alert on this.
	KillSwitchTripped = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "execution_kill_switch_tripped",
		Help: "1 when the trading kill switch is tripped, 0 otherwise",
	})

	// PaperRealizedPnL / PaperUnrealizedPnL track the simulated account.
	PaperRealizedPnL = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "paper_realized_pnl",
		Help: "Realised PnL of the paper trading account",
	})
	PaperUnrealizedPnL = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "paper_unrealized_pnl",
		Help: "Unrealised PnL of the paper trading account",
	})
	PaperOpenPositions = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "paper_open_positions",
		Help: "Number of open paper positions",
	})
	PaperFeesPaid = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "paper_fees_paid",
		Help: "Cumulative simulated fees and charges",
	})

	// TradePersistence counters expose the async DB writer.
	TradePersistWritten = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "trade_persist_written_total",
		Help: "Trades successfully written to the database",
	})
	TradePersistDropped = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "trade_persist_dropped_total",
		Help: "Trades dropped because the persistence queue was full",
	})
	TradePersistFailed = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "trade_persist_failed_total",
		Help: "Trades that failed to persist because of a database error",
	})
)

// RecordExecutionLatency records Broker.Place duration in microseconds.
func RecordExecutionLatency(us float64) { ExecutionLatency.Observe(us) }

// executionMetrics is folded into allMetrics for registration.
var executionMetrics = []prometheus.Collector{
	ExecutionOrdersPlaced,
	ExecutionOrdersFailed,
	ExecutionLatency,
	KillSwitchTripped,
	PaperRealizedPnL,
	PaperUnrealizedPnL,
	PaperOpenPositions,
	PaperFeesPaid,
	TradePersistWritten,
	TradePersistDropped,
	TradePersistFailed,
}
