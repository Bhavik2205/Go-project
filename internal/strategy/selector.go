package strategy

// Regime is a coarse classification of current market conditions.
type Regime uint8

const (
	RegimeUnknown Regime = iota
	RegimeTrending
	RegimeRanging
	RegimeVolatile
)

var regimeNames = [...]string{
	RegimeUnknown:  "unknown",
	RegimeTrending: "trending",
	RegimeRanging:  "ranging",
	RegimeVolatile: "volatile",
}

func (r Regime) String() string {
	if int(r) < len(regimeNames) {
		return regimeNames[r]
	}
	return "unknown"
}

// SelectorConfig tunes regime classification thresholds.
type SelectorConfig struct {
	// TrendADX and above is considered trending.
	TrendADX float64
	// RangeADX and below is considered ranging.
	RangeADX float64
	// VolatileATRPct is the ATR-as-fraction-of-price above which conditions
	// are treated as volatile regardless of trend strength.
	VolatileATRPct float64
}

// DefaultSelectorConfig returns sane defaults for Indian equity intraday data.
func DefaultSelectorConfig() SelectorConfig {
	return SelectorConfig{
		TrendADX:       25,
		RangeADX:       20,
		VolatileATRPct: 0.02, // ATR worth 2% of price
	}
}

// Selector classifies market regime from an indicator snapshot. It is a
// read-only helper: the engine runs every enabled strategy regardless, and
// Selector exists so operators and dashboards can see which family of
// strategy current conditions favour.
type Selector struct {
	cfg SelectorConfig
}

// NewSelector builds a Selector.
func NewSelector(cfg SelectorConfig) *Selector {
	if cfg.TrendADX <= 0 {
		cfg.TrendADX = 25
	}
	if cfg.RangeADX <= 0 {
		cfg.RangeADX = 20
	}
	if cfg.VolatileATRPct <= 0 {
		cfg.VolatileATRPct = 0.02
	}
	return &Selector{cfg: cfg}
}

// Classify returns the regime implied by a frame. It returns RegimeUnknown
// when the indicators it needs have not warmed up.
func (s *Selector) Classify(f *Frame) Regime {
	ind := &f.Ind

	// Volatility dominates: a violently volatile tape is not a clean trend
	// even when ADX is high.
	if ind.Present.Has(HasATR) && f.Close > 0 {
		if ind.ATR/f.Close >= s.cfg.VolatileATRPct {
			return RegimeVolatile
		}
	}

	if !ind.Present.Has(HasADX) {
		return RegimeUnknown
	}
	switch {
	case ind.ADX >= s.cfg.TrendADX:
		return RegimeTrending
	case ind.ADX <= s.cfg.RangeADX:
		return RegimeRanging
	default:
		return RegimeUnknown
	}
}

// Favoured returns the built-in strategy names suited to a regime. Callers
// should treat this as advisory; enabling a strategy remains a config
// decision in configs/strategy.yaml.
func (s *Selector) Favoured(r Regime) []string {
	switch r {
	case RegimeTrending:
		return []string{"intraday_trend", "swing_macd"}
	case RegimeRanging:
		return []string{"scalping_mean_reversion"}
	case RegimeVolatile:
		// Mean reversion into a volatile tape is how accounts die; prefer
		// trend following with its ATR-scaled stop, or stand aside.
		return []string{"intraday_trend"}
	default:
		return nil
	}
}
