package backtest

import (
	"math"
	"testing"
	"time"
)

func TestRequestValidateDefaults(t *testing.T) {
	r := Request{StrategyName: "intraday_trend", Instruments: []uint32{1}}
	if err := r.Validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
	if r.Interval != "5m" {
		t.Errorf("interval = %q, want 5m", r.Interval)
	}
	if r.InitialCapital != 1_000_000 {
		t.Errorf("capital = %v, want 1000000", r.InitialCapital)
	}
	if !r.From.Before(r.To) {
		t.Errorf("default window is not chronological: %v..%v", r.From, r.To)
	}
	if r.Params == nil {
		t.Error("params should default to an empty map, not nil")
	}
}

func TestRequestValidateRejections(t *testing.T) {
	now := time.Now()

	cases := []struct {
		name string
		req  Request
	}{
		{"no strategy", Request{Instruments: []uint32{1}}},
		{"unregistered strategy", Request{StrategyName: "nope", Instruments: []uint32{1}}},
		{"no instruments", Request{StrategyName: "intraday_trend"}},
		{
			"inverted window",
			Request{StrategyName: "intraday_trend", Instruments: []uint32{1}, From: now, To: now.Add(-time.Hour)},
		},
		{
			"negative slippage",
			Request{StrategyName: "intraday_trend", Instruments: []uint32{1}, SlippageBps: -1},
		},
	}

	for _, c := range cases {
		req := c.req
		if err := req.Validate(); err == nil {
			t.Errorf("%s: expected an error, got nil", c.name)
		}
	}
}

func TestCountOutcomesLongWinAndLoss(t *testing.T) {
	trades := []TradeLog{
		{InstrumentToken: 1, Action: "ENTRY", Side: "LONG", Quantity: 10, Price: 100},
		{InstrumentToken: 1, Action: "EXIT", Side: "LONG", Quantity: 10, Price: 110}, // +100
		{InstrumentToken: 1, Action: "ENTRY", Side: "LONG", Quantity: 10, Price: 100},
		{InstrumentToken: 1, Action: "EXIT", Side: "LONG", Quantity: 10, Price: 90}, // -100
	}
	wins, losses, rt := countOutcomes(trades)
	if rt != 2 {
		t.Errorf("round trips = %d, want 2", rt)
	}
	if wins != 1 || losses != 1 {
		t.Errorf("wins/losses = %d/%d, want 1/1", wins, losses)
	}
}

func TestCountOutcomesShortDirection(t *testing.T) {
	// A short that closes lower is a win.
	trades := []TradeLog{
		{InstrumentToken: 2, Action: "ENTRY", Side: "SHORT", Quantity: 5, Price: 100},
		{InstrumentToken: 2, Action: "EXIT", Side: "SHORT", Quantity: 5, Price: 90},
	}
	wins, losses, rt := countOutcomes(trades)
	if rt != 1 || wins != 1 || losses != 0 {
		t.Errorf("short win: got rt=%d wins=%d losses=%d, want 1/1/0", rt, wins, losses)
	}
}

func TestCountOutcomesIgnoresStillOpenTrade(t *testing.T) {
	// An entry with no exit has an unknown outcome and must not be counted.
	trades := []TradeLog{
		{InstrumentToken: 1, Action: "ENTRY", Side: "LONG", Quantity: 10, Price: 100},
	}
	wins, losses, rt := countOutcomes(trades)
	if rt != 0 || wins != 0 || losses != 0 {
		t.Errorf("open trade counted: rt=%d wins=%d losses=%d", rt, wins, losses)
	}
}

func TestCountOutcomesIgnoresExitWithoutEntry(t *testing.T) {
	trades := []TradeLog{
		{InstrumentToken: 1, Action: "EXIT", Side: "LONG", Quantity: 10, Price: 100},
	}
	if _, _, rt := countOutcomes(trades); rt != 0 {
		t.Errorf("round trips = %d, want 0", rt)
	}
}

func TestCountOutcomesSeparatesInstruments(t *testing.T) {
	// Two instruments open at once must not close against each other.
	trades := []TradeLog{
		{InstrumentToken: 1, Action: "ENTRY", Side: "LONG", Quantity: 10, Price: 100},
		{InstrumentToken: 2, Action: "ENTRY", Side: "LONG", Quantity: 10, Price: 200},
		{InstrumentToken: 1, Action: "EXIT", Side: "LONG", Quantity: 10, Price: 110},
		{InstrumentToken: 2, Action: "EXIT", Side: "LONG", Quantity: 10, Price: 190},
	}
	wins, losses, rt := countOutcomes(trades)
	if rt != 2 || wins != 1 || losses != 1 {
		t.Errorf("got rt=%d wins=%d losses=%d, want 2/1/1", rt, wins, losses)
	}
}

func TestSharpeZeroForFlatReturns(t *testing.T) {
	// No variance means no risk-adjusted signal; dividing by zero must not
	// produce Inf or NaN.
	if got := sharpe([]float64{0, 0, 0, 0}, "5m"); got != 0 {
		t.Errorf("sharpe = %v, want 0 for flat returns", got)
	}
}

func TestSharpeNeedsTwoPoints(t *testing.T) {
	if got := sharpe([]float64{0.01}, "5m"); got != 0 {
		t.Errorf("sharpe = %v, want 0 with a single return", got)
	}
}

func TestSharpeSignFollowsMean(t *testing.T) {
	up := sharpe([]float64{0.01, 0.02, 0.015, 0.005}, "5m")
	if up <= 0 {
		t.Errorf("sharpe = %v, want positive for consistently positive returns", up)
	}
	down := sharpe([]float64{-0.01, -0.02, -0.015, -0.005}, "5m")
	if down >= 0 {
		t.Errorf("sharpe = %v, want negative for consistently negative returns", down)
	}
	if math.IsNaN(up) || math.IsInf(up, 0) {
		t.Errorf("sharpe is not finite: %v", up)
	}
}

func TestBarsPerYearScalesWithInterval(t *testing.T) {
	oneMin := barsPerYear("1m")
	fiveMin := barsPerYear("5m")
	if oneMin <= fiveMin {
		t.Errorf("1m bars/year (%v) should exceed 5m (%v)", oneMin, fiveMin)
	}
	// A malformed interval must fall back rather than return zero and make
	// the Sharpe annualisation collapse.
	if got := barsPerYear("garbage"); got != 252 {
		t.Errorf("fallback bars/year = %v, want 252", got)
	}
}

func TestDownsampleKeepsWithinLimitAndEndpoint(t *testing.T) {
	base := time.Now()
	pts := make([]EquityPoint, 1000)
	for i := range pts {
		pts[i] = EquityPoint{Timestamp: base.Add(time.Duration(i) * time.Minute), Equity: float64(i)}
	}

	out := downsample(pts, 100)
	if len(out) > 101 { // 100 samples plus a possible appended final point
		t.Errorf("downsample returned %d points, want <= 101", len(out))
	}
	// The true final equity must survive: it is the headline number.
	if out[len(out)-1].Equity != pts[len(pts)-1].Equity {
		t.Errorf("final equity = %v, want %v", out[len(out)-1].Equity, pts[len(pts)-1].Equity)
	}
	if out[0].Equity != pts[0].Equity {
		t.Errorf("first equity = %v, want %v", out[0].Equity, pts[0].Equity)
	}
}

func TestDownsampleShortInputUnchanged(t *testing.T) {
	pts := []EquityPoint{{Equity: 1}, {Equity: 2}}
	if got := downsample(pts, 100); len(got) != 2 {
		t.Errorf("len = %d, want 2 (input already below the limit)", len(got))
	}
}
