package strategy

import "fmt"

// IntradayTrend is a VWAP/EMA trend-following strategy with an RSI filter.
//
// Long when price holds above both VWAP and the EMA while RSI confirms
// momentum without being stretched; short is the mirror image. Protective
// levels are derived from ATR so the stop widens with realised volatility
// instead of using a fixed percentage.
type IntradayTrend struct {
	base

	rsiLongMin   float64
	rsiLongMax   float64
	rsiShortMin  float64
	rsiShortMax  float64
	atrStopMult  float64
	atrTargetMul float64
	minADX       float64

	// prevAboveVWAP tracks the previous bar's side of VWAP so an exit fires
	// on the cross rather than on every bar spent below it.
	prevAboveVWAP bool
	hasPrev       bool
}

func init() {
	Register("intraday_trend", func() Strategy { return &IntradayTrend{} })
}

// Name is the registry key.
func (s *IntradayTrend) Name() string { return "intraday_trend" }

// Init reads tunables from configs/strategy.yaml.
func (s *IntradayTrend) Init(p Params) error {
	s.rsiLongMin = p.Float("rsi_long_min", 55)
	s.rsiLongMax = p.Float("rsi_long_max", 75)
	s.rsiShortMin = p.Float("rsi_short_min", 25)
	s.rsiShortMax = p.Float("rsi_short_max", 45)
	s.atrStopMult = p.Float("atr_stop_multiplier", 1.5)
	s.atrTargetMul = p.Float("atr_target_multiplier", 3.0)
	s.minADX = p.Float("min_adx", 20)

	if s.rsiLongMin >= s.rsiLongMax {
		return fmt.Errorf("rsi_long_min (%.2f) must be below rsi_long_max (%.2f)", s.rsiLongMin, s.rsiLongMax)
	}
	if s.rsiShortMin >= s.rsiShortMax {
		return fmt.Errorf("rsi_short_min (%.2f) must be below rsi_short_max (%.2f)", s.rsiShortMin, s.rsiShortMax)
	}
	if s.atrStopMult <= 0 {
		return fmt.Errorf("atr_stop_multiplier must be positive, got %.2f", s.atrStopMult)
	}
	if s.atrTargetMul <= 0 {
		return fmt.Errorf("atr_target_multiplier must be positive, got %.2f", s.atrTargetMul)
	}

	s.base = base{
		warmup:   p.Int("warmup", 5),
		requires: HasEMA | HasRSI | HasVWAP | HasATR,
	}
	return nil
}

// OnCandle evaluates one finalized candle. Allocation-free.
func (s *IntradayTrend) OnCandle(f *Frame, pos *Position) Signal {
	ind := &f.Ind
	aboveVWAP := f.Close > ind.VWAP

	prevAbove, hadPrev := s.prevAboveVWAP, s.hasPrev
	s.prevAboveVWAP, s.hasPrev = aboveVWAP, true

	// --- Exits: leave when price loses the VWAP side that justified entry ---
	if pos.IsOpen() {
		if !hadPrev {
			return None
		}
		if pos.Side == SideLong && prevAbove && !aboveVWAP {
			return Signal{Action: ActionExit, Side: SideLong, Reason: ReasonTrendDown, Price: f.Close}
		}
		if pos.Side == SideShort && !prevAbove && aboveVWAP {
			return Signal{Action: ActionExit, Side: SideShort, Reason: ReasonTrendUp, Price: f.Close}
		}
		return None
	}

	// --- Entries ---
	// A weak trend produces whipsaws, so require ADX confirmation when it is
	// available. When ADX is not configured the filter is simply skipped.
	if ind.Present.Has(HasADX) && ind.ADX < s.minADX {
		return None
	}

	if aboveVWAP && f.Close > ind.EMA &&
		ind.RSI >= s.rsiLongMin && ind.RSI <= s.rsiLongMax {
		return Signal{
			Action:      ActionEntry,
			Side:        SideLong,
			Reason:      ReasonTrendUp,
			Price:       f.Close,
			StopPrice:   f.Close - s.atrStopMult*ind.ATR,
			TargetPrice: f.Close + s.atrTargetMul*ind.ATR,
			Strength:    trendStrength(ind, s.minADX),
		}
	}

	if !aboveVWAP && f.Close < ind.EMA &&
		ind.RSI >= s.rsiShortMin && ind.RSI <= s.rsiShortMax {
		return Signal{
			Action:      ActionEntry,
			Side:        SideShort,
			Reason:      ReasonTrendDown,
			Price:       f.Close,
			StopPrice:   f.Close + s.atrStopMult*ind.ATR,
			TargetPrice: f.Close - s.atrTargetMul*ind.ATR,
			Strength:    trendStrength(ind, s.minADX),
		}
	}

	return None
}

// trendStrength maps ADX onto a 0.5..1.0 sizing multiplier. Without ADX the
// strategy sizes at full weight.
func trendStrength(ind *IndicatorSnapshot, minADX float64) float64 {
	if !ind.Present.Has(HasADX) {
		return 1
	}
	// ADX at minADX -> 0.5, ADX at 50 or above -> 1.0
	span := 50 - minADX
	if span <= 0 {
		return 1
	}
	r := (ind.ADX - minADX) / span
	if r < 0 {
		r = 0
	}
	if r > 1 {
		r = 1
	}
	return 0.5 + 0.5*r
}
