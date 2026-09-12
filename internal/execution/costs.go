package monitor

import "math"

// CostModel reproduces the charges an Indian equity intraday (MIS) trade
// actually incurs. Ignoring these is the single biggest reason a strategy that
// looks profitable on paper loses money live: on a ₹50,000 intraday round trip
// the statutory charges alone are larger than most per-trade edges.
//
// Rates are configurable because they change by regulation and by broker.
// The defaults below are Zerodha-equivalent equity intraday as of 2025 and
// should be re-checked before being relied on.
type CostModel struct {
	// BrokeragePct is charged per executed order, e.g. 0.0003 for 0.03%.
	BrokeragePct float64 `yaml:"brokerage_pct"`
	// BrokerageCap is the per-order rupee ceiling, e.g. 20. Zero means no cap.
	BrokerageCap float64 `yaml:"brokerage_cap"`

	// STTPct is Securities Transaction Tax. For equity intraday it applies to
	// the SELL side only, which makes a round trip asymmetric.
	STTPct float64 `yaml:"stt_pct"`

	// ExchangeTxnPct is the exchange transaction charge, both sides.
	ExchangeTxnPct float64 `yaml:"exchange_txn_pct"`

	// SEBIPct is the SEBI turnover fee, both sides.
	SEBIPct float64 `yaml:"sebi_pct"`

	// StampDutyPct applies to the BUY side only.
	StampDutyPct float64 `yaml:"stamp_duty_pct"`

	// GSTPct is charged on brokerage + exchange + SEBI charges, not on STT
	// or stamp duty.
	GSTPct float64 `yaml:"gst_pct"`
}

// DefaultCostModel returns Zerodha-equivalent equity intraday rates.
func DefaultCostModel() CostModel {
	return CostModel{
		BrokeragePct:   0.0003,    // 0.03%
		BrokerageCap:   20,        // ₹20 per order
		STTPct:         0.00025,   // 0.025%, sell side only
		ExchangeTxnPct: 0.0000297, // NSE 0.00297%
		SEBIPct:        0.000001,  // ₹10 per crore
		StampDutyPct:   0.00003,   // 0.003%, buy side only
		GSTPct:         0.18,      // 18% on brokerage + exchange + SEBI
	}
}

// ZeroCostModel charges nothing. Useful for isolating strategy logic in tests,
// never for evaluating whether a strategy is actually profitable.
func ZeroCostModel() CostModel { return CostModel{} }

// CostBreakdown itemises the charges on one fill, so a backtest can show
// where the money went rather than a single opaque "fees" figure.
type CostBreakdown struct {
	Brokerage   float64 `json:"brokerage"`
	STT         float64 `json:"stt"`
	ExchangeTxn float64 `json:"exchange_txn"`
	SEBI        float64 `json:"sebi"`
	StampDuty   float64 `json:"stamp_duty"`
	GST         float64 `json:"gst"`
	Total       float64 `json:"total"`
}

// Charges computes the cost of one fill. buy selects the side, because STT and
// stamp duty are one-sided.
func (c CostModel) Charges(turnover float64, buy bool) CostBreakdown {
	if turnover <= 0 {
		return CostBreakdown{}
	}

	var b CostBreakdown

	b.Brokerage = turnover * c.BrokeragePct
	if c.BrokerageCap > 0 && b.Brokerage > c.BrokerageCap {
		b.Brokerage = c.BrokerageCap
	}

	b.ExchangeTxn = turnover * c.ExchangeTxnPct
	b.SEBI = turnover * c.SEBIPct

	if buy {
		b.StampDuty = turnover * c.StampDutyPct
	} else {
		// STT on intraday equity is charged on the sell side only.
		b.STT = turnover * c.STTPct
	}

	// GST applies to brokerage and the exchange/SEBI charges, not to STT or
	// stamp duty, which are themselves taxes.
	b.GST = (b.Brokerage + b.ExchangeTxn + b.SEBI) * c.GSTPct

	b.Total = b.Brokerage + b.STT + b.ExchangeTxn + b.SEBI + b.StampDuty + b.GST
	return b
}

// Add accumulates a breakdown into the receiver.
func (b *CostBreakdown) Add(o CostBreakdown) {
	b.Brokerage += o.Brokerage
	b.STT += o.STT
	b.ExchangeTxn += o.ExchangeTxn
	b.SEBI += o.SEBI
	b.StampDuty += o.StampDuty
	b.GST += o.GST
	b.Total += o.Total
}

// SlippageModel converts an order into an execution price worse than the
// reference. It has two parts because real slippage has two causes: crossing
// the bid-ask spread, which every order pays, and market impact, which only
// large orders pay.
type SlippageModel struct {
	// HalfSpreadBps is the cost of crossing the spread, paid on every fill.
	HalfSpreadBps float64 `yaml:"half_spread_bps"`

	// ImpactCoefficient scales market impact. Impact is modelled as
	// coefficient * sqrt(participation), the standard square-root law: taking
	// 25% of a bar's volume costs roughly half of what taking 100% would.
	ImpactCoefficient float64 `yaml:"impact_coefficient_bps"`

	// MaxParticipation caps the fraction of a bar's volume one order may take.
	// Assuming you can buy an entire bar's volume at its close is the most
	// common way a backtest invents liquidity that does not exist.
	MaxParticipation float64 `yaml:"max_participation"`
}

// DefaultSlippageModel returns conservative defaults for liquid NSE equities.
func DefaultSlippageModel() SlippageModel {
	return SlippageModel{
		HalfSpreadBps:     2,    // 0.02%
		ImpactCoefficient: 10,   // 10bps at 100% participation
		MaxParticipation:  0.10, // never take more than 10% of a bar
	}
}

// FillPlan is the outcome of applying the slippage model to a request.
type FillPlan struct {
	// Quantity is what actually fills, which may be less than requested when
	// the bar lacked the volume.
	Quantity int64
	// Price is the execution price after spread and impact.
	Price float64
	// SlippageBps is the total adverse move, for reporting.
	SlippageBps float64
	// Capped reports that liquidity limited the fill.
	Capped bool
}

// Plan computes the fill for an order against one bar.
//
// barVolume of zero means volume is unknown; the participation cap is then
// skipped rather than silently rejecting the order, because refusing every
// fill on a feed without volume would be worse than ignoring impact.
func (s SlippageModel) Plan(ref float64, want int64, barVolume float64, buy bool) FillPlan {
	if ref <= 0 || want <= 0 {
		return FillPlan{}
	}

	qty := want
	capped := false

	if barVolume > 0 && s.MaxParticipation > 0 {
		maxQty := int64(barVolume * s.MaxParticipation)
		if maxQty < 1 {
			// The bar is too thin to support even one share at the cap.
			return FillPlan{Capped: true}
		}
		if qty > maxQty {
			qty = maxQty
			capped = true
		}
	}

	bps := s.HalfSpreadBps

	if barVolume > 0 && s.ImpactCoefficient > 0 {
		participation := float64(qty) / barVolume
		if participation > 1 {
			participation = 1
		}
		bps += s.ImpactCoefficient * math.Sqrt(participation)
	}

	adj := ref * bps / 10_000
	price := ref - adj
	if buy {
		price = ref + adj
	}
	if price <= 0 {
		// A slippage model should never produce a non-positive price; fall
		// back to the reference rather than booking a nonsensical fill.
		price = ref
	}

	return FillPlan{Quantity: qty, Price: price, SlippageBps: bps, Capped: capped}
}
