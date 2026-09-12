package strategy

import "fmt"

// Swing is a MACD-crossover momentum strategy for higher intervals.
//
// It enters on a MACD/signal crossover confirmed by ADX trend strength and
// holds for multiple bars, exiting on the opposite crossover or after a
// configured holding period. Unlike the intraday strategies it does not fade
// moves; it rides them.
type Swing struct {
	base

	minADX  float64
	maxBars int32
	// requireHistogram demands the histogram agree with the crossover, which
	// filters crossovers that immediately stall.
	requireHistogram bool

	prevHist float64
	hasPrev  bool
}

func init() {
	Register("swing_macd", func() Strategy { return &Swing{} })
}

// Name is the registry key.
func (s *Swing) Name() string { return "swing_macd" }

// Init reads tunables from configs/strategy.yaml.
func (s *Swing) Init(p Params) error {
	s.minADX = p.Float("min_adx", 25)
	s.maxBars = int32(p.Int("max_bars_in_trade", 200))
	s.requireHistogram = p.Bool("require_histogram", true)

	if s.minADX < 0 {
		return fmt.Errorf("min_adx must not be negative, got %.2f", s.minADX)
	}
	if s.maxBars <= 0 {
		return fmt.Errorf("max_bars_in_trade must be positive, got %d", s.maxBars)
	}

	s.base = base{
		warmup:   p.Int("warmup", 10),
		requires: HasMACD | HasADX,
	}
	return nil
}

// OnCandle evaluates one finalized candle. Allocation-free.
func (s *Swing) OnCandle(f *Frame, pos *Position) Signal {
	ind := &f.Ind

	prev, hadPrev := s.prevHist, s.hasPrev
	s.prevHist, s.hasPrev = ind.MACDHist, true

	// A crossover needs a previous bar to compare against.
	if !hadPrev {
		return None
	}

	crossUp := prev <= 0 && ind.MACDHist > 0
	crossDown := prev >= 0 && ind.MACDHist < 0

	// --- Exits: opposite crossover, or holding period elapsed ---
	if pos.IsOpen() {
		if pos.BarsHeld >= s.maxBars {
			return Signal{Action: ActionExit, Side: pos.Side, Reason: ReasonTimeExit, Price: f.Close}
		}
		if pos.Side == SideLong && crossDown {
			return Signal{Action: ActionExit, Side: SideLong, Reason: ReasonMomentumCrossDown, Price: f.Close}
		}
		if pos.Side == SideShort && crossUp {
			return Signal{Action: ActionExit, Side: SideShort, Reason: ReasonMomentumCrossUp, Price: f.Close}
		}
		return None
	}

	// --- Entries ---
	if ind.ADX < s.minADX {
		return None
	}

	if crossUp && (!s.requireHistogram || ind.MACDLine > ind.MACDSignal) {
		return Signal{
			Action:   ActionEntry,
			Side:     SideLong,
			Reason:   ReasonMomentumCrossUp,
			Price:    f.Close,
			Strength: trendStrength(ind, s.minADX),
		}
	}

	if crossDown && (!s.requireHistogram || ind.MACDLine < ind.MACDSignal) {
		return Signal{
			Action:   ActionEntry,
			Side:     SideShort,
			Reason:   ReasonMomentumCrossDown,
			Price:    f.Close,
			Strength: trendStrength(ind, s.minADX),
		}
	}

	return None
}
