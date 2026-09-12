package monitor

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/strategy"
)

func fill(qty int64, price float64) Fill {
	return Fill{InstrumentToken: 1, Symbol: "NSE:TEST", SignedQuantity: qty, Price: price, Time: time.Now()}
}

// ---------------------------------------------------------------------------
// Position book netting
// ---------------------------------------------------------------------------

func TestBookOpenLong(t *testing.T) {
	b := NewBook()
	realized, pos := b.Apply(fill(100, 50))

	if realized != 0 {
		t.Errorf("opening a position realised %v, want 0", realized)
	}
	if pos.Quantity != 100 || pos.AveragePrice != 50 {
		t.Errorf("pos = qty %d @ %v, want 100 @ 50", pos.Quantity, pos.AveragePrice)
	}
}

func TestBookAddToLongAveragesEntry(t *testing.T) {
	b := NewBook()
	b.Apply(fill(100, 50))
	realized, pos := b.Apply(fill(100, 60))

	if realized != 0 {
		t.Errorf("adding to a position realised %v, want 0", realized)
	}
	if pos.Quantity != 200 {
		t.Errorf("qty = %d, want 200", pos.Quantity)
	}
	if pos.AveragePrice != 55 { // (100*50 + 100*60) / 200
		t.Errorf("avg = %v, want 55", pos.AveragePrice)
	}
}

func TestBookPartialCloseRealisesAgainstAverage(t *testing.T) {
	b := NewBook()
	b.Apply(fill(100, 50))
	realized, pos := b.Apply(fill(-40, 60)) // sell 40 at 60

	if realized != 400 { // 40 * (60-50)
		t.Errorf("realised = %v, want 400", realized)
	}
	if pos.Quantity != 60 {
		t.Errorf("qty = %d, want 60", pos.Quantity)
	}
	// A partial close must not move the average entry of what remains.
	if pos.AveragePrice != 50 {
		t.Errorf("avg = %v, want 50 (unchanged by a partial close)", pos.AveragePrice)
	}
}

func TestBookFullCloseGoesFlat(t *testing.T) {
	b := NewBook()
	b.Apply(fill(100, 50))
	realized, pos := b.Apply(fill(-100, 55))

	if realized != 500 {
		t.Errorf("realised = %v, want 500", realized)
	}
	if pos.IsOpen() {
		t.Errorf("position still open with qty %d", pos.Quantity)
	}
	if pos.AveragePrice != 0 {
		t.Errorf("avg = %v, want 0 when flat", pos.AveragePrice)
	}
}

func TestBookFlipThroughZero(t *testing.T) {
	b := NewBook()
	b.Apply(fill(100, 50))
	// Sell 150 at 60: closes 100 long (+1000) and opens 50 short at 60.
	realized, pos := b.Apply(fill(-150, 60))

	if realized != 1000 {
		t.Errorf("realised = %v, want 1000 (only the closed 100)", realized)
	}
	if pos.Quantity != -50 {
		t.Errorf("qty = %d, want -50", pos.Quantity)
	}
	if pos.AveragePrice != 60 {
		t.Errorf("avg = %v, want 60 (the flip price)", pos.AveragePrice)
	}
}

func TestBookShortProfitsWhenPriceFalls(t *testing.T) {
	b := NewBook()
	b.Apply(fill(-100, 50)) // short 100 at 50
	realized, _ := b.Apply(fill(100, 40))

	if realized != 1000 { // 100 * (50-40)
		t.Errorf("short realised = %v, want 1000", realized)
	}
}

func TestBookUnrealizedPnL(t *testing.T) {
	b := NewBook()
	b.Apply(fill(100, 50))
	b.Mark(1, 55, time.Now())

	pos, _ := b.Position(1)
	if got := pos.UnrealizedPnL(); got != 500 {
		t.Errorf("unrealised = %v, want 500", got)
	}

	realized, unrealized := b.Totals()
	if realized != 0 || unrealized != 500 {
		t.Errorf("totals = (%v, %v), want (0, 500)", realized, unrealized)
	}
}

func TestBookShortUnrealizedPnL(t *testing.T) {
	b := NewBook()
	b.Apply(fill(-100, 50))
	b.Mark(1, 45, time.Now())

	pos, _ := b.Position(1)
	// Short 100 from 50, now 45: (45-50) * -100 = +500.
	if got := pos.UnrealizedPnL(); got != 500 {
		t.Errorf("short unrealised = %v, want 500", got)
	}
}

func TestBookOpenPositionsExcludesFlat(t *testing.T) {
	b := NewBook()
	b.Apply(fill(100, 50))
	b.Apply(fill(-100, 50)) // back to flat

	if got := len(b.OpenPositions()); got != 0 {
		t.Errorf("open positions = %d, want 0", got)
	}
	if got := len(b.Positions()); got != 1 {
		t.Errorf("all positions = %d, want 1 (flat rows are retained)", got)
	}
}

// ---------------------------------------------------------------------------
// Paper broker
// ---------------------------------------------------------------------------

func order(action strategy.Action, side strategy.Side, qty int64, price float64) strategy.Order {
	return strategy.Order{
		Token: 1, Symbol: "NSE:TEST", Strategy: "test",
		Action: action, Side: side, Quantity: qty, Price: price,
		Timestamp: time.Now(),
	}
}

func TestIsBuyMapping(t *testing.T) {
	cases := []struct {
		name string
		o    strategy.Order
		want bool
	}{
		{"enter long buys", order(strategy.ActionEntry, strategy.SideLong, 1, 10), true},
		{"enter short sells", order(strategy.ActionEntry, strategy.SideShort, 1, 10), false},
		{"exit long sells", order(strategy.ActionExit, strategy.SideLong, 1, 10), false},
		{"exit short buys back", order(strategy.ActionExit, strategy.SideShort, 1, 10), true},
	}
	for _, c := range cases {
		if got := isBuy(c.o); got != c.want {
			t.Errorf("%s: isBuy = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestPaperSlippageIsAlwaysAdverse(t *testing.T) {
	// A buy must never fill below the reference, and a sell never above.
	if got := applySlippage(100, 50, true); got <= 100 {
		t.Errorf("buy fill %v should be above reference 100", got)
	}
	if got := applySlippage(100, 50, false); got >= 100 {
		t.Errorf("sell fill %v should be below reference 100", got)
	}
}

func TestPaperBrokerFillsAndNets(t *testing.T) {
	b := NewPaperBroker(PaperConfig{SlippageBps: 0, FeeBps: 0}, 1, nil)
	ctx := context.Background()

	if _, err := b.Place(ctx, order(strategy.ActionEntry, strategy.SideLong, 100, 50)); err != nil {
		t.Fatalf("entry: %v", err)
	}
	pos, ok := b.Book().Position(1)
	if !ok || pos.Quantity != 100 {
		t.Fatalf("after entry pos = %+v", pos)
	}

	if _, err := b.Place(ctx, order(strategy.ActionExit, strategy.SideLong, 100, 60)); err != nil {
		t.Fatalf("exit: %v", err)
	}
	st := b.Stats()
	if st.RealizedPnL != 1000 {
		t.Errorf("realised = %v, want 1000", st.RealizedPnL)
	}
	if len(st.OpenPositions) != 0 {
		t.Errorf("expected flat, got %+v", st.OpenPositions)
	}
}

func TestPaperBrokerChargesFees(t *testing.T) {
	b := NewPaperBroker(PaperConfig{SlippageBps: 0, FeeBps: 10}, 1, nil) // 0.1%
	ctx := context.Background()

	if _, err := b.Place(ctx, order(strategy.ActionEntry, strategy.SideLong, 100, 50)); err != nil {
		t.Fatalf("entry: %v", err)
	}
	// notional 5000 * 0.1% = 5
	if got := b.FeesPaid(); got != 5 {
		t.Errorf("fees = %v, want 5", got)
	}
	if st := b.Stats(); st.NetPnL != -5 {
		t.Errorf("net pnl = %v, want -5 (fees only)", st.NetPnL)
	}
}

func TestPaperBrokerRejectsBadOrders(t *testing.T) {
	b := NewPaperBroker(DefaultPaperConfig(), 1, nil)
	ctx := context.Background()

	if _, err := b.Place(ctx, order(strategy.ActionEntry, strategy.SideLong, 100, 0)); err == nil {
		t.Error("expected rejection at zero reference price")
	}
	if _, err := b.Place(ctx, order(strategy.ActionEntry, strategy.SideLong, 0, 50)); err == nil {
		t.Error("expected rejection at zero quantity")
	}
}

// ---------------------------------------------------------------------------
// Guard / kill switch
// ---------------------------------------------------------------------------

type stubBroker struct {
	calls int
	err   error
}

func (s *stubBroker) Place(context.Context, strategy.Order) (string, error) {
	s.calls++
	if s.err != nil {
		return "", s.err
	}
	return "OK", nil
}
func (s *stubBroker) Mode() string { return TradeTypePaper }

func guardCfg() GuardConfig {
	return GuardConfig{
		MaxNotionalPerDay:   10_000,
		MaxOrdersPerDay:     3,
		MaxNotionalPerOrder: 5_000,
		Location:            time.UTC,
	}
}

func TestGuardBlocksWhenTripped(t *testing.T) {
	inner := &stubBroker{}
	g := NewGuard(inner, guardCfg())
	g.Trip(TripManual)

	if _, err := g.Place(context.Background(), order(strategy.ActionEntry, strategy.SideLong, 1, 10)); err == nil {
		t.Fatal("expected placement to be refused while tripped")
	}
	if inner.calls != 0 {
		t.Errorf("inner broker was called %d times while tripped", inner.calls)
	}
}

func TestGuardResetRearms(t *testing.T) {
	inner := &stubBroker{}
	g := NewGuard(inner, guardCfg())
	g.Trip(TripManual)
	g.Reset()

	if g.Tripped() {
		t.Fatal("guard still tripped after reset")
	}
	if _, err := g.Place(context.Background(), order(strategy.ActionEntry, strategy.SideLong, 1, 10)); err != nil {
		t.Fatalf("placement after reset: %v", err)
	}
}

func TestGuardPerOrderNotionalCapTrips(t *testing.T) {
	inner := &stubBroker{}
	g := NewGuard(inner, guardCfg())

	// 1000 * 10 = 10,000 notional, above the 5,000 per-order cap.
	if _, err := g.Place(context.Background(), order(strategy.ActionEntry, strategy.SideLong, 1000, 10)); err == nil {
		t.Fatal("expected per-order cap to refuse")
	}
	if !g.Tripped() {
		t.Error("an oversized order should trip the kill switch, not just refuse")
	}
	if inner.calls != 0 {
		t.Errorf("inner broker called %d times despite cap breach", inner.calls)
	}
}

func TestGuardDailyOrderCap(t *testing.T) {
	inner := &stubBroker{}
	g := NewGuard(inner, guardCfg()) // 3 orders/day
	ctx := context.Background()
	o := order(strategy.ActionEntry, strategy.SideLong, 10, 10) // 100 notional

	for i := 0; i < 3; i++ {
		if _, err := g.Place(ctx, o); err != nil {
			t.Fatalf("order %d refused early: %v", i+1, err)
		}
	}
	if _, err := g.Place(ctx, o); err == nil {
		t.Fatal("expected the 4th order to breach the daily cap")
	}
	if inner.calls != 3 {
		t.Errorf("inner calls = %d, want 3", inner.calls)
	}
}

func TestGuardDailyNotionalCap(t *testing.T) {
	inner := &stubBroker{}
	cfg := guardCfg()
	cfg.MaxOrdersPerDay = 100 // isolate the notional cap
	g := NewGuard(inner, cfg) // 10,000/day, 5,000/order
	ctx := context.Background()
	o := order(strategy.ActionEntry, strategy.SideLong, 400, 10) // 4,000 each

	for i := 0; i < 2; i++ {
		if _, err := g.Place(ctx, o); err != nil {
			t.Fatalf("order %d refused early: %v", i+1, err)
		}
	}
	// Two at 4,000 = 8,000; a third would reach 12,000.
	if _, err := g.Place(ctx, o); err == nil {
		t.Fatal("expected the daily notional cap to refuse")
	}
}

func TestGuardReleasesBudgetOnBrokerError(t *testing.T) {
	inner := &stubBroker{err: errors.New("venue down")}
	cfg := guardCfg()
	cfg.MaxOrdersPerDay = 2
	g := NewGuard(inner, cfg)
	ctx := context.Background()
	o := order(strategy.ActionEntry, strategy.SideLong, 10, 10)

	// A failed placement transacted nothing, so it must not consume budget.
	if _, err := g.Place(ctx, o); err == nil {
		t.Fatal("expected the broker error to surface")
	}
	st := g.Stats()
	if st.OrdersToday != 0 {
		t.Errorf("orders_today = %d after a failure, want 0", st.OrdersToday)
	}
	if st.NotionalToday != 0 {
		t.Errorf("notional_today = %v after a failure, want 0", st.NotionalToday)
	}
}

func TestGuardTripsOnBrokerErrorStreak(t *testing.T) {
	inner := &stubBroker{err: errors.New("venue down")}
	cfg := guardCfg()
	cfg.MaxOrdersPerDay = 100
	g := NewGuard(inner, cfg)
	ctx := context.Background()
	o := order(strategy.ActionEntry, strategy.SideLong, 10, 10)

	for i := 0; i < 5; i++ {
		g.Place(ctx, o)
	}
	if !g.Tripped() {
		t.Error("expected a run of broker failures to trip the kill switch")
	}
}

func TestGuardModeDelegates(t *testing.T) {
	g := NewGuard(&stubBroker{}, guardCfg())
	if got := g.Mode(); got != TradeTypePaper {
		t.Errorf("Mode = %q, want %q", got, TradeTypePaper)
	}
}

// ---------------------------------------------------------------------------
// Live trading gate
// ---------------------------------------------------------------------------

func TestLiveDisabledByDefault(t *testing.T) {
	t.Setenv(EnvTradingMode, "")
	t.Setenv(EnvLiveConfirm, "")

	if ok, _ := LiveEnabled(); ok {
		t.Fatal("live trading must be disabled when nothing is configured")
	}
	if got := TradingMode(); got != TradeTypePaper {
		t.Errorf("TradingMode = %q, want PAPER", got)
	}
}

func TestLiveNeedsBothSwitches(t *testing.T) {
	// Mode alone is not enough.
	t.Setenv(EnvTradingMode, "live")
	t.Setenv(EnvLiveConfirm, "")
	if ok, _ := LiveEnabled(); ok {
		t.Error("live enabled with TRADING_MODE alone; the confirmation phrase must also be required")
	}

	// Confirmation alone is not enough.
	t.Setenv(EnvTradingMode, "paper")
	t.Setenv(EnvLiveConfirm, LiveConfirmPhrase)
	if ok, _ := LiveEnabled(); ok {
		t.Error("live enabled with the confirmation phrase alone")
	}

	// A truthy-but-wrong confirmation must not work.
	t.Setenv(EnvTradingMode, "live")
	t.Setenv(EnvLiveConfirm, "true")
	if ok, _ := LiveEnabled(); ok {
		t.Error("live enabled by LIVE_TRADING_CONFIRM=true; only the exact phrase should work")
	}

	// Both correct.
	t.Setenv(EnvLiveConfirm, LiveConfirmPhrase)
	if ok, why := LiveEnabled(); !ok {
		t.Errorf("live should be enabled with both switches set: %s", why)
	}
}

func TestNewLiveBrokerRefusesWhenDisabled(t *testing.T) {
	t.Setenv(EnvTradingMode, "paper")
	t.Setenv(EnvLiveConfirm, "")

	if _, err := NewLiveBroker(nil, nil, "MIS", 1); err == nil {
		t.Fatal("expected NewLiveBroker to refuse while live trading is disabled")
	}
}

func TestNewBrokerForModeFallsBackToPaper(t *testing.T) {
	t.Setenv(EnvTradingMode, "paper")
	t.Setenv(EnvLiveConfirm, "")

	paper := NewPaperBroker(DefaultPaperConfig(), 1, nil)
	g, mode := NewBrokerForMode(nil, nil, paper, guardCfg(), 1)
	if mode != TradeTypePaper {
		t.Errorf("mode = %q, want PAPER", mode)
	}
	if g == nil {
		t.Fatal("expected a guard even in paper mode")
	}
}

func TestTagForIsWithinKiteLimit(t *testing.T) {
	got := tagFor("a_very_long_strategy_name_beyond_the_limit")
	if len(got) > 20 {
		t.Errorf("tag %q is %d chars, Kite allows 20", got, len(got))
	}
}
