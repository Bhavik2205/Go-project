package strategy

import "time"

// Side is the direction of a trade signal.
type Side uint8

const (
	SideNone Side = iota // no action
	SideLong
	SideShort
)

func (s Side) String() string {
	switch s {
	case SideLong:
		return "LONG"
	case SideShort:
		return "SHORT"
	default:
		return "NONE"
	}
}

// Action distinguishes opening from closing exposure.
type Action uint8

const (
	ActionNone Action = iota
	ActionEntry
	ActionExit
)

func (a Action) String() string {
	switch a {
	case ActionEntry:
		return "ENTRY"
	case ActionExit:
		return "EXIT"
	default:
		return "NONE"
	}
}

// Reason is an enumerated signal cause. It is a uint8 rather than a string so
// that emitting a signal on the hot path never allocates.
type Reason uint8

const (
	ReasonNone Reason = iota
	ReasonTrendUp
	ReasonTrendDown
	ReasonMeanReversionLong
	ReasonMeanReversionShort
	ReasonMomentumCrossUp
	ReasonMomentumCrossDown
	ReasonStopLoss
	ReasonTakeProfit
	ReasonTimeExit
	ReasonSessionEnd
	ReasonRiskLimit
)

var reasonNames = [...]string{
	ReasonNone:               "none",
	ReasonTrendUp:            "trend_up",
	ReasonTrendDown:          "trend_down",
	ReasonMeanReversionLong:  "mean_reversion_long",
	ReasonMeanReversionShort: "mean_reversion_short",
	ReasonMomentumCrossUp:    "momentum_cross_up",
	ReasonMomentumCrossDown:  "momentum_cross_down",
	ReasonStopLoss:           "stop_loss",
	ReasonTakeProfit:         "take_profit",
	ReasonTimeExit:           "time_exit",
	ReasonSessionEnd:         "session_end",
	ReasonRiskLimit:          "risk_limit",
}

func (r Reason) String() string {
	if int(r) < len(reasonNames) {
		return reasonNames[r]
	}
	return "unknown"
}

// Mask marks which fields of an IndicatorSnapshot carry a valid value.
// Stateful indicators need a warm-up period, so "present" and "zero" are
// different things and must not be confused.
type Mask uint16

const (
	HasSMA Mask = 1 << iota
	HasEMA
	HasRSI
	HasMACD
	HasATR
	HasBollinger
	HasStochastic
	HasOBV
	HasVWAP
	HasADX
)

// Has reports whether every indicator in m is present.
func (m Mask) Has(want Mask) bool { return m&want == want }

// IndicatorSnapshot is a flat, fixed-size view of the indicator values for one
// finalized candle. It is deliberately a value type with no maps, slices or
// pointers: copying it is a handful of MOVs and it never escapes to the heap.
type IndicatorSnapshot struct {
	Present Mask

	SMA  float64
	EMA  float64
	RSI  float64
	ATR  float64
	OBV  float64
	VWAP float64

	MACDLine   float64
	MACDSignal float64
	MACDHist   float64

	BBUpper  float64
	BBMiddle float64
	BBLower  float64

	StochK float64
	StochD float64

	ADX     float64
	PlusDI  float64
	MinusDI float64
}

// Frame is the complete input to a strategy evaluation for one finalized
// candle. It is passed by pointer purely to avoid copying ~200 bytes per
// strategy; strategies must treat it as read-only.
type Frame struct {
	Token    uint32
	Interval string
	Symbol   string

	Timestamp time.Time

	Open   float64
	High   float64
	Low    float64
	Close  float64
	Volume float64

	Ind IndicatorSnapshot
}

// Position is the engine-owned view of current exposure for one
// (instrument, interval, strategy) triple, handed to the strategy read-only.
type Position struct {
	Side       Side
	EntryPrice float64
	Quantity   int64
	EntryTime  time.Time
	BarsHeld   int32
	// StopPrice and TargetPrice are set by the engine from the strategy's
	// entry signal and re-checked on every bar.
	StopPrice   float64
	TargetPrice float64
}

// IsOpen reports whether there is live exposure.
func (p *Position) IsOpen() bool { return p.Side != SideNone }

// Signal is what a strategy returns from OnCandle. The zero value means
// "do nothing", which is the overwhelmingly common case.
type Signal struct {
	Action Action
	Side   Side
	Reason Reason

	// Price is the reference price for the decision (normally the candle close).
	Price float64

	// StopPrice and TargetPrice are optional protective levels for an entry.
	// Zero means "not specified"; the engine then falls back to config defaults.
	StopPrice   float64
	TargetPrice float64

	// Strength is a 0..1 confidence used for position sizing. Zero is treated
	// as 1.0 by the risk manager.
	Strength float64
}

// None is the do-nothing signal.
var None = Signal{}

// IsActionable reports whether the signal asks for anything to happen.
func (s Signal) IsActionable() bool { return s.Action != ActionNone && s.Side != SideNone }

// Order is the engine's output after risk sizing: a concrete, executable
// intent. The execution layer consumes these.
type Order struct {
	Token       uint32
	Symbol      string
	Interval    string
	Strategy    string
	Action      Action
	Side        Side
	Reason      Reason
	Price       float64
	Quantity    int64
	StopPrice   float64
	TargetPrice float64
	Timestamp   time.Time

	// BarVolume is the volume of the candle that produced this order. The
	// paper broker uses it to cap fills and to size market impact; assuming a
	// whole bar's volume is available at its close invents liquidity that was
	// never there.
	BarVolume float64
}
