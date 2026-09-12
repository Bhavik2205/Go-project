// Package monitor provides system monitoring and order execution utilities.
package monitor

import (
	"context"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/observability"
	"github.com/Bhavik2205/ML-Bot/internal/strategy"
	"go.uber.org/zap"
)

// Broker places orders with an execution venue. Implementations must be safe
// for concurrent use.
//
// Both PaperBroker and LiveBroker implement it, so the executor is identical
// in simulation and in production and there is no untested code path that
// only runs with real money.
type Broker interface {
	Place(ctx context.Context, o strategy.Order) (string, error)
	// Mode reports the trade type recorded for fills: PAPER or LIVE.
	Mode() string
}

// Alerter is the subset of the notification dispatcher the executor needs.
// An interface keeps execution from importing the notifications package and
// lets the field stay nil when alerting is off.
type Alerter interface {
	TradeExecuted(mode, strategy, symbol, action, side string, qty int64, price float64)
	ErrorAlert(subject, detail string)
}

// OrderExecutor drains the strategy engine's order queue and forwards each
// intent to a Broker. It runs on its own goroutine so broker latency never
// propagates back into the market data pipeline.
type OrderExecutor struct {
	orders <-chan strategy.Order
	broker Broker

	// alerter is optional; nil means no notifications are raised.
	alerter Alerter

	// timeout bounds a single Place call.
	timeout time.Duration

	placed atomic.Uint64
	failed atomic.Uint64
}

// NewOrderExecutor builds an executor reading from orders.
func NewOrderExecutor(orders <-chan strategy.Order, broker Broker, timeout time.Duration) *OrderExecutor {
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	return &OrderExecutor{orders: orders, broker: broker, timeout: timeout}
}

// SetAlerter attaches a notification dispatcher. Call during startup, before
// Run, because the field is read without synchronisation.
func (x *OrderExecutor) SetAlerter(a Alerter) { x.alerter = a }

// Run consumes orders until ctx is cancelled or the channel closes.
func (x *OrderExecutor) Run(ctx context.Context) {
	defer observability.RecoverPanic("order-executor")
	zap.L().Info("✅ Order executor started", zap.String("mode", x.broker.Mode()))

	for {
		select {
		case o, ok := <-x.orders:
			if !ok {
				zap.L().Info("Order channel closed, stopping executor")
				return
			}
			x.place(ctx, o)
		case <-ctx.Done():
			zap.L().Info("Context cancelled, stopping order executor")
			return
		}
	}
}

func (x *OrderExecutor) place(ctx context.Context, o strategy.Order) {
	callCtx, cancel := context.WithTimeout(ctx, x.timeout)
	defer cancel()

	start := time.Now()
	id, err := x.broker.Place(callCtx, o)
	observability.RecordExecutionLatency(float64(time.Since(start).Microseconds()))

	if err != nil {
		n := x.failed.Add(1)
		observability.ExecutionOrdersFailed.Inc()
		if x.alerter != nil {
			// Alert on the first failure and then sparsely: a venue outage
			// would otherwise send one message per rejected order.
			if n == 1 || n%10 == 0 {
				x.alerter.ErrorAlert("Order placement failing",
					fmt.Sprintf("%s %s x%d: %v (failure #%d)", o.Side, o.Symbol, o.Quantity, err, n))
			}
		}
		zap.L().Error("order placement failed",
			zap.String("strategy", o.Strategy),
			zap.Uint32("token", o.Token),
			zap.String("action", o.Action.String()),
			zap.String("side", o.Side.String()),
			zap.Int64("quantity", o.Quantity),
			zap.Error(err),
		)
		return
	}

	x.placed.Add(1)
	observability.ExecutionOrdersPlaced.WithLabelValues(x.broker.Mode()).Inc()
	if x.alerter != nil {
		x.alerter.TradeExecuted(x.broker.Mode(), o.Strategy, o.Symbol,
			o.Action.String(), o.Side.String(), o.Quantity, o.Price)
	}
	zap.L().Info("order placed",
		zap.String("order_id", id),
		zap.String("mode", x.broker.Mode()),
		zap.String("strategy", o.Strategy),
		zap.String("symbol", o.Symbol),
		zap.String("action", o.Action.String()),
		zap.String("side", o.Side.String()),
		zap.Int64("quantity", o.Quantity),
		zap.Float64("price", o.Price),
	)
}

// ExecutorStats reports executor counters.
type ExecutorStats struct {
	Mode   string `json:"mode"`
	Placed uint64 `json:"placed"`
	Failed uint64 `json:"failed"`
}

// Stats returns executor counters for the monitoring API.
func (x *OrderExecutor) Stats() ExecutorStats {
	return ExecutorStats{
		Mode:   x.broker.Mode(),
		Placed: x.placed.Load(),
		Failed: x.failed.Load(),
	}
}
