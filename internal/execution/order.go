// Package monitor provides system monitoring and order execution utilities.
package monitor

import (
	"context"
	"sync/atomic"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/observability"
	"github.com/Bhavik2205/ML-Bot/internal/strategy"
	"go.uber.org/zap"
)

// Broker places orders with an upstream venue. Implementations must be
// safe for concurrent use.
type Broker interface {
	// Place submits an order and returns the broker's order id.
	Place(ctx context.Context, o strategy.Order) (string, error)
}

// PaperBroker records orders without sending them anywhere. It is the
// default so that enabling the strategy engine cannot place a live order by
// accident — switching to a real broker is an explicit wiring decision.
type PaperBroker struct {
	placed atomic.Uint64
}

// NewPaperBroker builds a paper broker.
func NewPaperBroker() *PaperBroker { return &PaperBroker{} }

// Place logs the intent and returns a synthetic order id.
func (b *PaperBroker) Place(_ context.Context, o strategy.Order) (string, error) {
	n := b.placed.Add(1)
	zap.L().Info("paper order",
		zap.String("strategy", o.Strategy),
		zap.Uint32("token", o.Token),
		zap.String("symbol", o.Symbol),
		zap.String("interval", o.Interval),
		zap.String("action", o.Action.String()),
		zap.String("side", o.Side.String()),
		zap.String("reason", o.Reason.String()),
		zap.Float64("price", o.Price),
		zap.Int64("quantity", o.Quantity),
		zap.Float64("stop", o.StopPrice),
		zap.Float64("target", o.TargetPrice),
		zap.Time("candle_time", o.Timestamp),
	)
	return paperOrderID(n), nil
}

// Placed is the number of orders this broker has accepted.
func (b *PaperBroker) Placed() uint64 { return b.placed.Load() }

func paperOrderID(n uint64) string {
	const prefix = "PAPER-"
	buf := make([]byte, 0, len(prefix)+20)
	buf = append(buf, prefix...)
	return string(appendUint(buf, n))
}

func appendUint(dst []byte, v uint64) []byte {
	if v == 0 {
		return append(dst, '0')
	}
	var tmp [20]byte
	i := len(tmp)
	for v > 0 {
		i--
		tmp[i] = byte('0' + v%10)
		v /= 10
	}
	return append(dst, tmp[i:]...)
}

// OrderExecutor drains the strategy engine's order queue and forwards each
// intent to a Broker. It runs on its own goroutine so that broker latency
// never propagates back into the market data pipeline.
type OrderExecutor struct {
	orders <-chan strategy.Order
	broker Broker

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

// Run consumes orders until ctx is cancelled or the channel closes.
func (x *OrderExecutor) Run(ctx context.Context) {
	defer observability.RecoverPanic("order-executor")
	zap.L().Info("✅ Order executor started")

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

	id, err := x.broker.Place(callCtx, o)
	if err != nil {
		x.failed.Add(1)
		zap.L().Error("order placement failed",
			zap.String("strategy", o.Strategy),
			zap.Uint32("token", o.Token),
			zap.String("action", o.Action.String()),
			zap.String("side", o.Side.String()),
			zap.Error(err),
		)
		return
	}
	x.placed.Add(1)
	zap.L().Debug("order placed", zap.String("order_id", id), zap.String("strategy", o.Strategy))
}

// Placed and Failed expose executor counters for the runtime endpoint.
func (x *OrderExecutor) Placed() uint64 { return x.placed.Load() }
func (x *OrderExecutor) Failed() uint64 { return x.failed.Load() }
