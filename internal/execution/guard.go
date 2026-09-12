package monitor

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/strategy"
	"go.uber.org/zap"
)

// GuardConfig bounds what any broker behind the guard may do in a single day.
type GuardConfig struct {
	// MaxNotionalPerDay caps the total rupee value transacted per trading day.
	MaxNotionalPerDay float64
	// MaxOrdersPerDay caps order count per trading day.
	MaxOrdersPerDay int
	// MaxNotionalPerOrder rejects any single oversized order, which is the
	// usual shape of a sizing bug.
	MaxNotionalPerOrder float64
	// Location is the exchange timezone used to roll the daily counters.
	Location *time.Location
}

// DefaultGuardConfig returns conservative caps.
func DefaultGuardConfig(loc *time.Location) GuardConfig {
	if loc == nil {
		loc = time.UTC
	}
	return GuardConfig{
		MaxNotionalPerDay:   1_000_000,
		MaxOrdersPerDay:     100,
		MaxNotionalPerOrder: 200_000,
		Location:            loc,
	}
}

// TripReason explains why the guard refused or halted.
type TripReason string

const (
	TripManual            TripReason = "manual"
	TripNotionalPerDay    TripReason = "max_notional_per_day"
	TripOrdersPerDay      TripReason = "max_orders_per_day"
	TripNotionalPerOrder  TripReason = "max_notional_per_order"
	TripBrokerErrorStreak TripReason = "broker_error_streak"
)

// Guard wraps a Broker with a kill switch and daily caps.
//
// Once tripped it stays tripped until Reset is called explicitly. That is
// deliberate: an automatic recovery would let whatever caused the trip start
// placing orders again unattended.
type Guard struct {
	inner Broker
	cfg   GuardConfig

	tripped   atomic.Bool
	reason    atomic.Value // TripReason
	trippedAt atomic.Value // time.Time

	mu            sync.Mutex
	day           int
	notionalToday float64
	ordersToday   int
	errorStreak   int

	placed  atomic.Uint64
	refused atomic.Uint64
}

// NewGuard wraps a broker.
func NewGuard(inner Broker, cfg GuardConfig) *Guard {
	if cfg.Location == nil {
		cfg.Location = time.UTC
	}
	g := &Guard{inner: inner, cfg: cfg, day: -1}
	g.reason.Store(TripReason(""))
	g.trippedAt.Store(time.Time{})
	return g
}

// Trip halts all order placement. Safe to call from an HTTP handler.
func (g *Guard) Trip(reason TripReason) {
	if g.tripped.CompareAndSwap(false, true) {
		g.reason.Store(reason)
		g.trippedAt.Store(time.Now())
		zap.L().Error("🛑 TRADING KILL SWITCH TRIPPED", zap.String("reason", string(reason)))
	}
}

// Reset re-arms the guard and clears the daily counters. This is the only way
// back from a trip, and it is a deliberate human action.
func (g *Guard) Reset() {
	g.mu.Lock()
	g.notionalToday = 0
	g.ordersToday = 0
	g.errorStreak = 0
	g.mu.Unlock()

	g.tripped.Store(false)
	g.reason.Store(TripReason(""))
	g.trippedAt.Store(time.Time{})
	zap.L().Warn("Trading kill switch reset; order placement re-armed")
}

// Tripped reports whether placement is currently halted.
func (g *Guard) Tripped() bool { return g.tripped.Load() }

// Mode reports the wrapped broker's trade type, so the Guard itself satisfies
// Broker and can be handed straight to the executor.
func (g *Guard) Mode() string { return g.inner.Mode() }

// Place applies the caps, then delegates.
func (g *Guard) Place(ctx context.Context, o strategy.Order) (string, error) {
	if g.tripped.Load() {
		g.refused.Add(1)
		return "", fmt.Errorf("trading halted by kill switch (%s)", g.reason.Load())
	}

	notional := o.Price * float64(o.Quantity)

	if g.cfg.MaxNotionalPerOrder > 0 && notional > g.cfg.MaxNotionalPerOrder {
		// A single oversized order is far more likely to be a sizing bug than
		// a real intent, so trip rather than merely refuse.
		g.refused.Add(1)
		g.Trip(TripNotionalPerOrder)
		return "", fmt.Errorf("order notional %.2f exceeds per-order cap %.2f", notional, g.cfg.MaxNotionalPerOrder)
	}

	g.mu.Lock()
	g.rollDayLocked(o.Timestamp)
	if g.cfg.MaxOrdersPerDay > 0 && g.ordersToday >= g.cfg.MaxOrdersPerDay {
		g.mu.Unlock()
		g.refused.Add(1)
		g.Trip(TripOrdersPerDay)
		return "", fmt.Errorf("daily order cap %d reached", g.cfg.MaxOrdersPerDay)
	}
	if g.cfg.MaxNotionalPerDay > 0 && g.notionalToday+notional > g.cfg.MaxNotionalPerDay {
		g.mu.Unlock()
		g.refused.Add(1)
		g.Trip(TripNotionalPerDay)
		return "", fmt.Errorf("daily notional cap %.2f would be exceeded", g.cfg.MaxNotionalPerDay)
	}
	// Reserve the budget before the call so concurrent placements cannot both
	// pass the check and jointly breach the cap.
	g.ordersToday++
	g.notionalToday += notional
	g.mu.Unlock()

	id, err := g.inner.Place(ctx, o)
	if err != nil {
		g.mu.Lock()
		// Release the reservation: nothing was actually transacted.
		g.ordersToday--
		g.notionalToday -= notional
		g.errorStreak++
		streak := g.errorStreak
		g.mu.Unlock()

		// A run of broker failures means something systemic; stop trying.
		if streak >= 5 {
			g.Trip(TripBrokerErrorStreak)
		}
		return "", err
	}

	g.mu.Lock()
	g.errorStreak = 0
	g.mu.Unlock()
	g.placed.Add(1)
	return id, nil
}

func (g *Guard) rollDayLocked(t time.Time) {
	if t.IsZero() {
		t = time.Now()
	}
	lt := t.In(g.cfg.Location)
	d := lt.Year()*1000 + lt.YearDay()
	if d != g.day {
		g.day = d
		g.notionalToday = 0
		g.ordersToday = 0
	}
}

// GuardStats is a snapshot for the monitoring API.
type GuardStats struct {
	Tripped       bool      `json:"tripped"`
	Reason        string    `json:"reason,omitempty"`
	TrippedAt     time.Time `json:"tripped_at,omitempty"`
	OrdersToday   int       `json:"orders_today"`
	NotionalToday float64   `json:"notional_today"`
	Placed        uint64    `json:"placed"`
	Refused       uint64    `json:"refused"`

	MaxOrdersPerDay     int     `json:"max_orders_per_day"`
	MaxNotionalPerDay   float64 `json:"max_notional_per_day"`
	MaxNotionalPerOrder float64 `json:"max_notional_per_order"`
}

// Stats returns the guard's current state.
func (g *Guard) Stats() GuardStats {
	g.mu.Lock()
	orders, notional := g.ordersToday, g.notionalToday
	g.mu.Unlock()

	reason, _ := g.reason.Load().(TripReason)
	at, _ := g.trippedAt.Load().(time.Time)

	return GuardStats{
		Tripped:             g.tripped.Load(),
		Reason:              string(reason),
		TrippedAt:           at,
		OrdersToday:         orders,
		NotionalToday:       notional,
		Placed:              g.placed.Load(),
		Refused:             g.refused.Load(),
		MaxOrdersPerDay:     g.cfg.MaxOrdersPerDay,
		MaxNotionalPerDay:   g.cfg.MaxNotionalPerDay,
		MaxNotionalPerOrder: g.cfg.MaxNotionalPerOrder,
	}
}
