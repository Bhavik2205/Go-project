package strategy

import "fmt"

// Params carries the tunable configuration for a single strategy instance.
// Values come from configs/strategy.yaml; a strategy reads what it needs in
// Init and stores it in typed fields so the hot path never touches a map.
type Params map[string]float64

// Float returns the parameter or def when absent.
func (p Params) Float(key string, def float64) float64 {
	if v, ok := p[key]; ok {
		return v
	}
	return def
}

// Int returns the parameter as an int, or def when absent.
func (p Params) Int(key string, def int) int {
	if v, ok := p[key]; ok {
		return int(v)
	}
	return def
}

// Bool treats any non-zero value as true.
func (p Params) Bool(key string, def bool) bool {
	if v, ok := p[key]; ok {
		return v != 0
	}
	return def
}

// Strategy is the contract every trading strategy implements.
//
// Implementations must obey three rules, because OnCandle runs on the market
// data hot path:
//
//  1. OnCandle must not allocate, block, log, or perform I/O.
//  2. OnCandle must be deterministic given (Frame, Position) and its own state.
//  3. All state mutation happens inside OnCandle only. The engine guarantees
//     that OnCandle is never called concurrently for the same instance, so
//     implementations need no internal locking.
type Strategy interface {
	// Name is the registry key, e.g. "intraday_trend".
	Name() string

	// Init applies configuration. It is called once before any OnCandle.
	Init(p Params) error

	// Requires returns the indicators this strategy needs. The engine skips
	// evaluation entirely until all of them are present in the snapshot,
	// which keeps warm-up handling out of every strategy.
	Requires() Mask

	// Warmup is the number of finalized candles to observe before the
	// strategy is allowed to signal, in addition to indicator readiness.
	Warmup() int

	// OnCandle evaluates one finalized candle and returns a signal.
	// Return None (the zero Signal) to do nothing, which is the common case.
	OnCandle(f *Frame, pos *Position) Signal
}

// Factory constructs a fresh, uninitialised Strategy instance. Each
// (instrument, interval) gets its own instance so indicator state and
// position state never bleed across symbols.
type Factory func() Strategy

// base provides the Warmup/Requires defaults so concrete strategies only
// override what they actually care about.
type base struct {
	warmup   int
	requires Mask
}

func (b *base) Warmup() int    { return b.warmup }
func (b *base) Requires() Mask { return b.requires }

// validate checks a freshly built strategy for obvious misconfiguration.
func validate(s Strategy) error {
	if s.Name() == "" {
		return fmt.Errorf("strategy has empty Name()")
	}
	if s.Warmup() < 0 {
		return fmt.Errorf("strategy %s: negative warmup %d", s.Name(), s.Warmup())
	}
	return nil
}
