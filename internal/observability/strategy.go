package observability

import "github.com/prometheus/client_golang/prometheus"

// Strategy engine metrics. Registration happens in metrics.go via allMetrics.
var (
	// StrategyEvaluations counts finalized candles handed to the engine.
	StrategyEvaluations = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "strategy_evaluations_total",
		Help: "Total finalized candles evaluated by the strategy engine",
	})

	// StrategySignals counts actionable signals returned by strategies,
	// before risk checks.
	StrategySignals = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "strategy_signals_total",
		Help: "Actionable signals produced by strategies, before risk checks",
	}, []string{"strategy", "action", "side"})

	// StrategyOrders counts order intents emitted after risk sizing.
	StrategyOrders = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "strategy_orders_total",
		Help: "Order intents emitted by the strategy engine after risk sizing",
	}, []string{"strategy", "action", "side"})

	// StrategyRejects counts signals refused by the risk manager.
	StrategyRejects = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "strategy_rejects_total",
		Help: "Signals rejected by the risk manager, by reason",
	}, []string{"reason"})

	// StrategyPanics counts recovered panics inside strategy code. Any
	// non-zero value here is a bug and should alert.
	StrategyPanics = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "strategy_panics_total",
		Help: "Panics recovered inside strategy OnCandle implementations",
	})

	// StrategyOrderDrops counts orders lost to a full outbound queue.
	StrategyOrderDrops = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "strategy_order_drops_total",
		Help: "Order intents dropped because the execution queue was full",
	})

	// StrategyLatency measures engine evaluation time per finalized candle.
	StrategyLatency = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "strategy_evaluation_duration_microseconds",
		Help:    "Time spent evaluating all strategies for one finalized candle",
		Buckets: []float64{1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000},
	})

	// StrategyOrderQueueDepth / Capacity expose the execution queue.
	StrategyOrderQueueDepth = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "strategy_order_queue_depth",
		Help: "Current depth of the strategy order queue",
	})
	StrategyOrderQueueCapacity = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "strategy_order_queue_capacity",
		Help: "Capacity of the strategy order queue",
	})

	// StrategyOpenPositions is the number of positions the engine believes
	// are open.
	StrategyOpenPositions = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "strategy_open_positions",
		Help: "Positions currently open according to the strategy engine",
	})

	// StrategyRealizedPnL is the running realised PnL for the trading day.
	StrategyRealizedPnL = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "strategy_realized_pnl",
		Help: "Realised PnL for the current trading day",
	})
)

// RecordStrategyLatency records engine evaluation time in microseconds.
func RecordStrategyLatency(us float64) { StrategyLatency.Observe(us) }

// strategyMetrics is folded into allMetrics for registration.
var strategyMetrics = []prometheus.Collector{
	StrategyEvaluations,
	StrategySignals,
	StrategyOrders,
	StrategyRejects,
	StrategyPanics,
	StrategyOrderDrops,
	StrategyLatency,
	StrategyOrderQueueDepth,
	StrategyOrderQueueCapacity,
	StrategyOpenPositions,
	StrategyRealizedPnL,
}
