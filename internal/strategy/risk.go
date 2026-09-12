package strategy

import (
	"sync"
	"time"
)

// RejectReason explains why the risk manager refused an order.
type RejectReason uint8

const (
	RejectNone RejectReason = iota
	RejectOutsideSession
	RejectShortDisallowed
	RejectMaxTradesPerDay
	RejectMaxOpenPositions
	RejectDailyLossLimit
	RejectZeroQuantity
)

var rejectNames = [...]string{
	RejectNone:             "none",
	RejectOutsideSession:   "outside_session",
	RejectShortDisallowed:  "short_disallowed",
	RejectMaxTradesPerDay:  "max_trades_per_day",
	RejectMaxOpenPositions: "max_open_positions",
	RejectDailyLossLimit:   "daily_loss_limit",
	RejectZeroQuantity:     "zero_quantity",
}

func (r RejectReason) String() string {
	if int(r) < len(rejectNames) {
		return rejectNames[r]
	}
	return "unknown"
}

// RiskManager enforces account-level limits and sizes positions. It is
// consulted only for actionable signals, which are rare relative to candles,
// so a single mutex here never shows up on the hot path.
//
// Exits are never blocked. A risk limit must not trap the account in a
// position it is trying to leave.
type RiskManager struct {
	cfg *Config

	mu sync.Mutex
	// day identifies the trading day the counters below belong to, so they
	// reset automatically on the first signal of a new day.
	day           int
	tradesToday   int
	openPositions int
	realizedPnL   float64
}

// NewRiskManager builds a risk manager bound to cfg.
func NewRiskManager(cfg *Config) *RiskManager {
	return &RiskManager{cfg: cfg, day: -1}
}

// rollDayLocked resets the daily counters when the calendar day changes.
// openPositions deliberately survives the roll: a position held overnight is
// still open.
func (r *RiskManager) rollDayLocked(t time.Time) {
	lt := t.In(r.cfg.Session.loc)
	d := lt.Year()*1000 + lt.YearDay()
	if d != r.day {
		r.day = d
		r.tradesToday = 0
		r.realizedPnL = 0
	}
}

// Evaluate applies risk rules to a signal and returns the sized quantity.
// A zero quantity with a non-zero RejectReason means the order is refused.
func (r *RiskManager) Evaluate(sig Signal, f *Frame) (qty int64, reject RejectReason) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.rollDayLocked(f.Timestamp)

	// Exits always pass. Size them at the caller's discretion (the engine
	// passes the open quantity).
	if sig.Action == ActionExit {
		return 0, RejectNone
	}

	if !r.cfg.Session.AllowsEntry(f.Timestamp) {
		return 0, RejectOutsideSession
	}
	if sig.Side == SideShort && !r.cfg.Risk.AllowShort {
		return 0, RejectShortDisallowed
	}
	if r.tradesToday >= r.cfg.Risk.MaxTradesPerDay {
		return 0, RejectMaxTradesPerDay
	}
	if r.openPositions >= r.cfg.Risk.MaxOpenPositions {
		return 0, RejectMaxOpenPositions
	}
	// realizedPnL is negative on a losing day; compare against the budget.
	if lossBudget := -r.cfg.Capital * r.cfg.Risk.MaxLossPerDayPercentage; r.realizedPnL <= lossBudget {
		return 0, RejectDailyLossLimit
	}

	qty = r.size(sig, f)
	if qty <= 0 {
		return 0, RejectZeroQuantity
	}
	return qty, RejectNone
}

// size converts a signal into a whole-lot quantity.
func (r *RiskManager) size(sig Signal, f *Frame) int64 {
	price := sig.Price
	if price <= 0 {
		price = f.Close
	}
	if price <= 0 {
		return 0
	}

	strength := sig.Strength
	if strength <= 0 || strength > 1 {
		strength = 1
	}

	notional := r.cfg.Capital * r.cfg.Risk.TradeSizePercentageOfCapital * strength
	return int64(notional / price)
}

// OnEntryFilled records that an entry was accepted.
func (r *RiskManager) OnEntryFilled() {
	r.mu.Lock()
	r.tradesToday++
	r.openPositions++
	r.mu.Unlock()
}

// OnExitFilled records a closed position and books its realised PnL.
func (r *RiskManager) OnExitFilled(pnl float64) {
	r.mu.Lock()
	if r.openPositions > 0 {
		r.openPositions--
	}
	r.realizedPnL += pnl
	r.mu.Unlock()
}

// Stats is a point-in-time snapshot for the runtime/metrics endpoints.
type Stats struct {
	TradesToday   int     `json:"trades_today"`
	OpenPositions int     `json:"open_positions"`
	RealizedPnL   float64 `json:"realized_pnl"`
}

// Stats returns the current risk counters.
func (r *RiskManager) Stats() Stats {
	r.mu.Lock()
	defer r.mu.Unlock()
	return Stats{
		TradesToday:   r.tradesToday,
		OpenPositions: r.openPositions,
		RealizedPnL:   r.realizedPnL,
	}
}
