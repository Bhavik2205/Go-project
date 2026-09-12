package strategy

import "fmt"

// Scalping is a Bollinger-band mean-reversion strategy for fast intervals.
//
// It fades stretched moves: buy when price closes below the lower band with
// RSI oversold, sell when it closes above the upper band with RSI overbought,
// and take profit back at the middle band. Exposure is deliberately short
// lived — MaxBars forces a time-based exit so a failed reversion does not
// turn into an accidental position trade.
type Scalping struct {
	base

	rsiOversold   float64
	rsiOverbought float64
	maxBars       int32
	// minBandWidthPct filters out flat, low-volatility regimes where the
	// bands hug price and every bar looks like a signal.
	minBandWidthPct float64
}

func init() {
	Register("scalping_mean_reversion", func() Strategy { return &Scalping{} })
}

// Name is the registry key.
func (s *Scalping) Name() string { return "scalping_mean_reversion" }

// Init reads tunables from configs/strategy.yaml.
func (s *Scalping) Init(p Params) error {
	s.rsiOversold = p.Float("rsi_oversold", 30)
	s.rsiOverbought = p.Float("rsi_overbought", 70)
	s.maxBars = int32(p.Int("max_bars_in_trade", 10))
	s.minBandWidthPct = p.Float("min_band_width_pct", 0.002) // 0.2% of price

	if s.rsiOversold >= s.rsiOverbought {
		return fmt.Errorf("rsi_oversold (%.2f) must be below rsi_overbought (%.2f)", s.rsiOversold, s.rsiOverbought)
	}
	if s.maxBars <= 0 {
		return fmt.Errorf("max_bars_in_trade must be positive, got %d", s.maxBars)
	}
	if s.minBandWidthPct < 0 {
		return fmt.Errorf("min_band_width_pct must not be negative, got %.4f", s.minBandWidthPct)
	}

	s.base = base{
		warmup:   p.Int("warmup", 3),
		requires: HasBollinger | HasRSI,
	}
	return nil
}

// OnCandle evaluates one finalized candle. Allocation-free.
func (s *Scalping) OnCandle(f *Frame, pos *Position) Signal {
	ind := &f.Ind

	// --- Exits: revert to the mean, or bail out on time ---
	if pos.IsOpen() {
		if pos.BarsHeld >= s.maxBars {
			return Signal{Action: ActionExit, Side: pos.Side, Reason: ReasonTimeExit, Price: f.Close}
		}
		if pos.Side == SideLong && f.Close >= ind.BBMiddle {
			return Signal{Action: ActionExit, Side: SideLong, Reason: ReasonTakeProfit, Price: f.Close}
		}
		if pos.Side == SideShort && f.Close <= ind.BBMiddle {
			return Signal{Action: ActionExit, Side: SideShort, Reason: ReasonTakeProfit, Price: f.Close}
		}
		return None
	}

	// --- Entries ---
	if ind.BBMiddle <= 0 {
		return None
	}
	// Skip regimes too quiet for the round trip to cover costs.
	if width := (ind.BBUpper - ind.BBLower) / ind.BBMiddle; width < s.minBandWidthPct {
		return None
	}

	if f.Close <= ind.BBLower && ind.RSI <= s.rsiOversold {
		return Signal{
			Action:      ActionEntry,
			Side:        SideLong,
			Reason:      ReasonMeanReversionLong,
			Price:       f.Close,
			TargetPrice: ind.BBMiddle,
			Strength:    stretchStrength(ind.RSI, s.rsiOversold, true),
		}
	}

	if f.Close >= ind.BBUpper && ind.RSI >= s.rsiOverbought {
		return Signal{
			Action:      ActionEntry,
			Side:        SideShort,
			Reason:      ReasonMeanReversionShort,
			Price:       f.Close,
			TargetPrice: ind.BBMiddle,
			Strength:    stretchStrength(ind.RSI, s.rsiOverbought, false),
		}
	}

	return None
}

// stretchStrength sizes further from the threshold more aggressively, capped
// at full weight. threshold is the RSI trigger; long is true for oversold.
func stretchStrength(rsi, threshold float64, long bool) float64 {
	var excess float64
	if long {
		excess = threshold - rsi // deeper oversold -> larger
	} else {
		excess = rsi - threshold
	}
	if excess <= 0 {
		return 0.5
	}
	// 0 points past the threshold -> 0.5, 20 points past -> 1.0
	r := excess / 20
	if r > 1 {
		r = 1
	}
	return 0.5 + 0.5*r
}
