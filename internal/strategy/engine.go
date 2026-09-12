package strategy

import (
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/Bhavik2205/ML-Bot/internal/observability"
)

// instance is one live strategy bound to one (instrument, interval).
type instance struct {
	strat  Strategy
	name   string
	req    Mask
	warmup int
	seen   int32 // finalized candles observed by this instance
	pos    Position
}

// slot holds every strategy instance for one (instrument, interval).
type slot struct {
	instances []*instance
}

type key struct {
	token    uint32
	interval string
}

// Engine evaluates registered strategies against finalized candles and emits
// risk-checked Orders.
//
// Concurrency: OnCandle may be called from any goroutine. In the current
// pipeline it is driven by the single IndicatorManager goroutine, so the
// read path below is uncontended; the write lock is only taken the first
// time a given (instrument, interval) is seen.
type Engine struct {
	cfg  *Config
	risk *RiskManager
	sink Sink

	mu    sync.RWMutex
	slots map[key]*slot

	// templates are the enabled instances resolved once from config.
	templates []InstanceConfig

	// counters, read by the runtime endpoint.
	evaluated atomic.Uint64
	signals   atomic.Uint64
	orders    atomic.Uint64
	rejects   atomic.Uint64
	panics    atomic.Uint64
}

// NewEngine builds an engine from config. It fails fast if a configured
// strategy cannot be constructed, so a bad config never reaches market hours.
func NewEngine(cfg *Config, sink Sink) (*Engine, error) {
	if cfg == nil {
		return nil, fmt.Errorf("strategy: nil config")
	}
	if sink == nil {
		return nil, fmt.Errorf("strategy: nil sink")
	}

	templates := cfg.EnabledStrategies()

	// Construct each strategy once up front purely to validate its params.
	// The throwaway instances are discarded; real ones are built per slot.
	for _, t := range templates {
		if _, err := New(t.Name, t.Params); err != nil {
			return nil, fmt.Errorf("strategy: validating %q: %w", t.Name, err)
		}
	}

	return &Engine{
		cfg:       cfg,
		risk:      NewRiskManager(cfg),
		sink:      sink,
		slots:     make(map[key]*slot),
		templates: templates,
	}, nil
}

// Risk exposes the risk manager for status reporting.
func (e *Engine) Risk() *RiskManager { return e.risk }

// EngineStats is a snapshot of engine counters.
type EngineStats struct {
	Evaluated uint64 `json:"evaluated"`
	Signals   uint64 `json:"signals"`
	Orders    uint64 `json:"orders"`
	Rejects   uint64 `json:"rejects"`
	Panics    uint64 `json:"panics"`
	Slots     int    `json:"slots"`
	Risk      Stats  `json:"risk"`
}

// Stats returns engine counters for the runtime endpoint.
func (e *Engine) Stats() EngineStats {
	e.mu.RLock()
	n := len(e.slots)
	e.mu.RUnlock()
	return EngineStats{
		Evaluated: e.evaluated.Load(),
		Signals:   e.signals.Load(),
		Orders:    e.orders.Load(),
		Rejects:   e.rejects.Load(),
		Panics:    e.panics.Load(),
		Slots:     n,
		Risk:      e.risk.Stats(),
	}
}

// slotFor returns the slot for a key, building it on first use.
func (e *Engine) slotFor(k key) *slot {
	e.mu.RLock()
	s, ok := e.slots[k]
	e.mu.RUnlock()
	if ok {
		return s
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	if s, ok := e.slots[k]; ok { // lost the race
		return s
	}

	s = &slot{}
	for _, t := range e.templates {
		if !intervalMatches(t.Intervals, k.interval) {
			continue
		}
		if len(t.Instruments) > 0 && !containsToken(t.Instruments, k.token) {
			continue
		}
		st, err := New(t.Name, t.Params)
		if err != nil {
			// Config was validated in NewEngine, so this is unreachable in
			// practice. Skip rather than panic on the data path.
			continue
		}
		s.instances = append(s.instances, &instance{
			strat:  st,
			name:   t.Name,
			req:    st.Requires(),
			warmup: st.Warmup(),
		})
	}
	e.slots[k] = s
	return s
}

func intervalMatches(list []string, interval string) bool {
	for _, v := range list {
		if v == interval || v == "*" {
			return true
		}
	}
	return false
}

func containsToken(list []uint32, tok uint32) bool {
	for _, v := range list {
		if v == tok {
			return true
		}
	}
	return false
}

// OnCandle evaluates every strategy bound to this instrument and interval.
// It is called once per finalized candle and never blocks: a full sink drops
// the order and increments a counter rather than stalling market data.
func (e *Engine) OnCandle(f *Frame) {
	if !e.cfg.Enabled {
		return
	}
	s := e.slotFor(key{token: f.Token, interval: f.Interval})
	if len(s.instances) == 0 {
		return
	}

	e.evaluated.Add(1)
	observability.StrategyEvaluations.Inc()

	for _, inst := range s.instances {
		e.evalInstance(inst, f)
	}
}

// evalInstance runs protective exits first, then the strategy itself.
// Protective levels are enforced by the engine so that every strategy gets
// stop-loss and square-off behaviour without reimplementing it.
func (e *Engine) evalInstance(inst *instance, f *Frame) {
	inst.seen++
	if inst.pos.IsOpen() {
		inst.pos.BarsHeld++
	}

	// 1. Engine-owned protective exits take precedence over strategy logic.
	if inst.pos.IsOpen() {
		if sig, hit := e.protectiveExit(inst, f); hit {
			e.handleSignal(inst, f, sig)
			return
		}
	}

	// 2. Warm-up and indicator readiness gates.
	if int(inst.seen) <= inst.warmup {
		return
	}
	if !f.Ind.Present.Has(inst.req) {
		return
	}

	// 3. Strategy evaluation, isolated from panics.
	sig := e.safeOnCandle(inst, f)
	if !sig.IsActionable() {
		return
	}
	e.signals.Add(1)
	observability.StrategySignals.WithLabelValues(inst.name, sig.Action.String(), sig.Side.String()).Inc()
	e.handleSignal(inst, f, sig)
}

// protectiveExit checks square-off, stop-loss and target in that order.
func (e *Engine) protectiveExit(inst *instance, f *Frame) (Signal, bool) {
	p := &inst.pos

	if e.cfg.Session.IsSquareOff(f.Timestamp) {
		return Signal{Action: ActionExit, Side: p.Side, Reason: ReasonSessionEnd, Price: f.Close}, true
	}

	// Stop is checked before target: within a single candle we cannot know
	// which level was touched first, so assume the adverse one.
	switch p.Side {
	case SideLong:
		if p.StopPrice > 0 && f.Low <= p.StopPrice {
			return Signal{Action: ActionExit, Side: SideLong, Reason: ReasonStopLoss, Price: p.StopPrice}, true
		}
		if p.TargetPrice > 0 && f.High >= p.TargetPrice {
			return Signal{Action: ActionExit, Side: SideLong, Reason: ReasonTakeProfit, Price: p.TargetPrice}, true
		}
	case SideShort:
		if p.StopPrice > 0 && f.High >= p.StopPrice {
			return Signal{Action: ActionExit, Side: SideShort, Reason: ReasonStopLoss, Price: p.StopPrice}, true
		}
		if p.TargetPrice > 0 && f.Low <= p.TargetPrice {
			return Signal{Action: ActionExit, Side: SideShort, Reason: ReasonTakeProfit, Price: p.TargetPrice}, true
		}
	}
	return None, false
}

// safeOnCandle isolates a misbehaving strategy so one panic cannot take down
// the market data pipeline. The deferred recover costs tens of nanoseconds
// and runs once per finalized candle, not once per tick.
func (e *Engine) safeOnCandle(inst *instance, f *Frame) (sig Signal) {
	defer func() {
		if r := recover(); r != nil {
			e.panics.Add(1)
			observability.StrategyPanics.Inc()
			sig = None
		}
	}()
	return inst.strat.OnCandle(f, &inst.pos)
}

// handleSignal applies a signal to the position and emits an order.
func (e *Engine) handleSignal(inst *instance, f *Frame, sig Signal) {
	p := &inst.pos

	switch sig.Action {
	case ActionExit:
		if !p.IsOpen() {
			return
		}
		price := sig.Price
		if price <= 0 {
			price = f.Close
		}
		qty := p.Quantity
		pnl := realizedPnL(p.Side, p.EntryPrice, price, qty)

		e.emit(Order{
			Token: f.Token, Symbol: f.Symbol, Interval: f.Interval,
			Strategy: inst.name, Action: ActionExit, Side: p.Side,
			Reason: sig.Reason, Price: price, Quantity: qty,
			Timestamp: f.Timestamp, BarVolume: f.Volume,
		})
		e.risk.OnExitFilled(pnl)
		e.publishRiskGauges()
		*p = Position{} // flat

	case ActionEntry:
		if p.IsOpen() {
			return // already exposed; no pyramiding
		}
		qty, reject := e.risk.Evaluate(sig, f)
		if reject != RejectNone {
			e.rejects.Add(1)
			observability.StrategyRejects.WithLabelValues(reject.String()).Inc()
			return
		}
		price := sig.Price
		if price <= 0 {
			price = f.Close
		}
		stop, target := e.protectiveLevels(sig, price)

		p.Side = sig.Side
		p.EntryPrice = price
		p.Quantity = qty
		p.EntryTime = f.Timestamp
		p.BarsHeld = 0
		p.StopPrice = stop
		p.TargetPrice = target

		e.emit(Order{
			Token: f.Token, Symbol: f.Symbol, Interval: f.Interval,
			Strategy: inst.name, Action: ActionEntry, Side: sig.Side,
			Reason: sig.Reason, Price: price, Quantity: qty,
			StopPrice: stop, TargetPrice: target,
			Timestamp: f.Timestamp, BarVolume: f.Volume,
		})
		e.risk.OnEntryFilled()
		e.publishRiskGauges()
	}
}

// protectiveLevels fills in stop and target from config when the strategy did
// not specify them.
func (e *Engine) protectiveLevels(sig Signal, price float64) (stop, target float64) {
	stop, target = sig.StopPrice, sig.TargetPrice
	slPct := e.cfg.Risk.DefaultStopLossPercentage
	tpPct := e.cfg.Risk.DefaultTargetPercentage

	if sig.Side == SideLong {
		if stop <= 0 {
			stop = price * (1 - slPct)
		}
		if target <= 0 {
			target = price * (1 + tpPct)
		}
	} else {
		if stop <= 0 {
			stop = price * (1 + slPct)
		}
		if target <= 0 {
			target = price * (1 - tpPct)
		}
	}
	return stop, target
}

func realizedPnL(side Side, entry, exit float64, qty int64) float64 {
	switch side {
	case SideLong:
		return (exit - entry) * float64(qty)
	case SideShort:
		return (entry - exit) * float64(qty)
	default:
		return 0
	}
}

func (e *Engine) emit(o Order) {
	if e.sink.Emit(o) {
		e.orders.Add(1)
		observability.StrategyOrders.WithLabelValues(o.Strategy, o.Action.String(), o.Side.String()).Inc()
	} else {
		observability.StrategyOrderDrops.Inc()
	}
}

// publishRiskGauges mirrors the risk counters into Prometheus. Call it after
// the risk manager has been updated, never before.
func (e *Engine) publishRiskGauges() {
	st := e.risk.Stats()
	observability.StrategyOpenPositions.Set(float64(st.OpenPositions))
	observability.StrategyRealizedPnL.Set(st.RealizedPnL)
}
