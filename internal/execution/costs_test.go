package monitor

import (
	"context"
	"math"
	"testing"

	"github.com/Bhavik2205/ML-Bot/internal/strategy"
)

// ---------------------------------------------------------------------------
// Cost model
// ---------------------------------------------------------------------------

func TestCostModelBrokerageCap(t *testing.T) {
	c := CostModel{BrokeragePct: 0.0003, BrokerageCap: 20}

	// 0.03% of 100,000 is 30, above the cap.
	if got := c.Charges(100_000, true).Brokerage; got != 20 {
		t.Errorf("brokerage = %v, want the 20 cap", got)
	}
	// 0.03% of 10,000 is 3, below the cap, so the percentage applies.
	// Compared with a tolerance because 10000*0.0003 is not exact in binary.
	if got := c.Charges(10_000, true).Brokerage; math.Abs(got-3) > 1e-9 {
		t.Errorf("brokerage = %v, want 3", got)
	}
}

func TestCostModelSTTIsSellSideOnly(t *testing.T) {
	c := DefaultCostModel()

	if got := c.Charges(100_000, true).STT; got != 0 {
		t.Errorf("buy STT = %v, want 0 (intraday STT is sell-side only)", got)
	}
	if got := c.Charges(100_000, false).STT; got <= 0 {
		t.Errorf("sell STT = %v, want positive", got)
	}
}

func TestCostModelStampDutyIsBuySideOnly(t *testing.T) {
	c := DefaultCostModel()

	if got := c.Charges(100_000, true).StampDuty; got <= 0 {
		t.Errorf("buy stamp duty = %v, want positive", got)
	}
	if got := c.Charges(100_000, false).StampDuty; got != 0 {
		t.Errorf("sell stamp duty = %v, want 0 (buy side only)", got)
	}
}

func TestCostModelGSTExcludesSTTAndStampDuty(t *testing.T) {
	c := DefaultCostModel()
	b := c.Charges(100_000, false) // sell side, so STT applies

	// GST is levied on brokerage + exchange + SEBI only. Including STT would
	// materially overstate the bill, so this guards a real mistake.
	want := (b.Brokerage + b.ExchangeTxn + b.SEBI) * c.GSTPct
	if math.Abs(b.GST-want) > 1e-9 {
		t.Errorf("GST = %v, want %v (STT and stamp duty must be excluded)", b.GST, want)
	}
}

func TestCostModelTotalIsSumOfParts(t *testing.T) {
	b := DefaultCostModel().Charges(250_000, false)
	sum := b.Brokerage + b.STT + b.ExchangeTxn + b.SEBI + b.StampDuty + b.GST
	if math.Abs(b.Total-sum) > 1e-9 {
		t.Errorf("total = %v, want %v", b.Total, sum)
	}
}

func TestCostModelZeroTurnoverIsFree(t *testing.T) {
	if got := DefaultCostModel().Charges(0, true).Total; got != 0 {
		t.Errorf("charges on zero turnover = %v, want 0", got)
	}
}

func TestZeroCostModelChargesNothing(t *testing.T) {
	if got := ZeroCostModel().Charges(1_000_000, true).Total; got != 0 {
		t.Errorf("zero cost model charged %v", got)
	}
}

func TestCostBreakdownAdd(t *testing.T) {
	c := DefaultCostModel()
	var acc CostBreakdown
	acc.Add(c.Charges(100_000, true))
	acc.Add(c.Charges(100_000, false))

	// A round trip pays stamp duty on the buy and STT on the sell.
	if acc.StampDuty <= 0 || acc.STT <= 0 {
		t.Errorf("round trip should incur both stamp duty and STT, got %+v", acc)
	}
}

// ---------------------------------------------------------------------------
// Slippage and liquidity
// ---------------------------------------------------------------------------

func TestSlippageCapsParticipation(t *testing.T) {
	m := SlippageModel{HalfSpreadBps: 1, MaxParticipation: 0.10}

	// 5000 shares against a 10,000-share bar at a 10 percent cap fills 1000.
	plan := m.Plan(100, 5000, 10_000, true)
	if plan.Quantity != 1000 {
		t.Errorf("quantity = %d, want 1000", plan.Quantity)
	}
	if !plan.Capped {
		t.Error("expected Capped when liquidity limits the fill")
	}
}

func TestSlippageRejectsWhenBarTooThin(t *testing.T) {
	m := SlippageModel{MaxParticipation: 0.10}

	// A 5-share bar at a 10 percent cap cannot support even one share.
	if plan := m.Plan(100, 10, 5, true); plan.Quantity != 0 {
		t.Errorf("quantity = %d, want 0 on a bar too thin to fill", plan.Quantity)
	}
}

func TestSlippageImpactGrowsWithParticipation(t *testing.T) {
	m := SlippageModel{HalfSpreadBps: 0, ImpactCoefficient: 100, MaxParticipation: 1}

	small := m.Plan(100, 100, 10_000, true)   // 1 percent participation
	large := m.Plan(100, 9_000, 10_000, true) // 90 percent participation

	if large.SlippageBps <= small.SlippageBps {
		t.Errorf("impact did not grow with size: %v vs %v", large.SlippageBps, small.SlippageBps)
	}
	// Square-root law: 90x the participation costs about 10x, not 90x.
	if large.SlippageBps > small.SlippageBps*20 {
		t.Errorf("impact grew faster than the square-root law allows: %v vs %v",
			large.SlippageBps, small.SlippageBps)
	}
}

func TestSlippageWithoutVolumeSkipsCap(t *testing.T) {
	m := DefaultSlippageModel()

	// A feed without volume must still fill, or every order would be refused.
	plan := m.Plan(100, 1_000_000, 0, true)
	if plan.Quantity != 1_000_000 {
		t.Errorf("quantity = %d, want the full request when volume is unknown", plan.Quantity)
	}
	if plan.Capped {
		t.Error("should not report capped when volume is unknown")
	}
}

func TestSlippageRejectsBadInput(t *testing.T) {
	m := DefaultSlippageModel()
	if plan := m.Plan(0, 10, 1000, true); plan.Quantity != 0 {
		t.Error("expected no fill at a non-positive reference price")
	}
	if plan := m.Plan(100, 0, 1000, true); plan.Quantity != 0 {
		t.Error("expected no fill for zero quantity")
	}
}

// ---------------------------------------------------------------------------
// Paper broker integration with the two models
// ---------------------------------------------------------------------------

func TestPaperBrokerPartialFillIsRecorded(t *testing.T) {
	cfg := FrictionlessPaperConfig()
	cfg.Slippage.MaxParticipation = 0.10
	b := NewPaperBroker(cfg, 1, nil)

	o := order(strategy.ActionEntry, strategy.SideLong, 5000, 100)
	o.BarVolume = 10_000

	if _, err := b.Place(context.Background(), o); err != nil {
		t.Fatalf("place: %v", err)
	}

	pos, _ := b.Book().Position(1)
	if pos.Quantity != 1000 {
		t.Errorf("filled %d, want 1000 (capped at 10 percent of the bar)", pos.Quantity)
	}
	if st := b.Stats(); st.PartialFills != 1 {
		t.Errorf("partial fills = %d, want 1", st.PartialFills)
	}
}

func TestPaperBrokerRejectsIlliquidBar(t *testing.T) {
	cfg := FrictionlessPaperConfig()
	cfg.Slippage.MaxParticipation = 0.10
	b := NewPaperBroker(cfg, 1, nil)

	o := order(strategy.ActionEntry, strategy.SideLong, 100, 100)
	o.BarVolume = 5 // 10 percent of 5 is less than one share

	if _, err := b.Place(context.Background(), o); err == nil {
		t.Fatal("expected a rejection on a bar with no usable liquidity")
	}
	if st := b.Stats(); st.RejectedFills != 1 {
		t.Errorf("rejected fills = %d, want 1", st.RejectedFills)
	}
}

func TestPaperBrokerReportsItemisedCosts(t *testing.T) {
	b := NewPaperBroker(DefaultPaperConfig(), 1, nil)
	ctx := context.Background()

	buy := order(strategy.ActionEntry, strategy.SideLong, 100, 1000)
	if _, err := b.Place(ctx, buy); err != nil {
		t.Fatalf("buy: %v", err)
	}
	sell := order(strategy.ActionExit, strategy.SideLong, 100, 1010)
	if _, err := b.Place(ctx, sell); err != nil {
		t.Fatalf("sell: %v", err)
	}

	st := b.Stats()
	if st.Costs.Total <= 0 {
		t.Fatal("a round trip should incur charges")
	}
	if st.Costs.Brokerage <= 0 || st.Costs.GST <= 0 {
		t.Errorf("expected brokerage and GST: %+v", st.Costs)
	}
	if st.Costs.STT <= 0 {
		t.Error("the sell leg should have incurred STT")
	}
	if st.Costs.StampDuty <= 0 {
		t.Error("the buy leg should have incurred stamp duty")
	}
	if st.Turnover <= 0 {
		t.Error("turnover should be recorded")
	}
	if st.AvgSlippageBps <= 0 {
		t.Error("average slippage should be positive with the default model")
	}
}

func TestPaperBrokerCostsMakeAMarginalTradeLose(t *testing.T) {
	// The whole point of the cost model: a round trip whose gross profit is
	// smaller than the charges must show a net loss.
	b := NewPaperBroker(DefaultPaperConfig(), 1, nil)
	ctx := context.Background()

	// Buy 100 at 1000, sell at 1000.20: gross +20 before costs.
	if _, err := b.Place(ctx, order(strategy.ActionEntry, strategy.SideLong, 100, 1000)); err != nil {
		t.Fatalf("buy: %v", err)
	}
	if _, err := b.Place(ctx, order(strategy.ActionExit, strategy.SideLong, 100, 1000.20)); err != nil {
		t.Fatalf("sell: %v", err)
	}

	if st := b.Stats(); st.NetPnL >= 0 {
		t.Errorf("net PnL = %v; a 0.02%% move should not survive slippage and charges", st.NetPnL)
	}
}
