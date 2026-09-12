package strategy

import (
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

// scripted is a test strategy that returns whatever signal it is told to.
type scripted struct {
	base
	next Signal
	// panicOnCandle makes OnCandle panic, to exercise engine isolation.
	panicOnCandle bool
	calls         int
}

func (s *scripted) Name() string { return "scripted" }

func (s *scripted) Init(p Params) error {
	s.base = base{warmup: p.Int("warmup", 0), requires: 0}
	return nil
}

func (s *scripted) OnCandle(f *Frame, pos *Position) Signal {
	s.calls++
	if s.panicOnCandle {
		panic("boom")
	}
	return s.next
}

// current holds the instance the registry hands out, so a test can script it.
var current *scripted

func init() {
	Register("scripted", func() Strategy {
		current = &scripted{}
		return current
	})
}

// testConfig returns a config with one enabled scripted strategy.
func testConfig(t *testing.T) *Config {
	t.Helper()
	cfg := &Config{
		Enabled: true,
		Capital: 1_000_000,
		Session: SessionConfig{Timezone: "Asia/Kolkata", Open: "09:15", Close: "15:30", SquareOff: "15:15"},
		Risk: RiskConfig{
			MaxTradesPerDay:              10,
			MaxOpenPositions:             5,
			TradeSizePercentageOfCapital: 0.05,
			MaxLossPerDayPercentage:      0.02,
			DefaultStopLossPercentage:    0.01,
			DefaultTargetPercentage:      0.02,
			AllowShort:                   true,
		},
		Strategies: []InstanceConfig{{
			Name: "scripted", Enabled: true, Intervals: []string{"5m"},
		}},
	}
	if err := cfg.Normalize(); err != nil {
		t.Fatalf("normalize: %v", err)
	}
	return cfg
}

// at builds a timestamp inside the trading session on a fixed date.
func at(t *testing.T, hhmm string) time.Time {
	t.Helper()
	loc, err := time.LoadLocation("Asia/Kolkata")
	if err != nil {
		t.Fatalf("load tz: %v", err)
	}
	ts, err := time.ParseInLocation("2006-01-02 15:04", "2026-09-11 "+hhmm, loc)
	if err != nil {
		t.Fatalf("parse time: %v", err)
	}
	return ts
}

func frame(t *testing.T, hhmm string, o, h, l, c float64) *Frame {
	t.Helper()
	return &Frame{
		Token: 111, Interval: "5m", Symbol: "NSE:TEST",
		Timestamp: at(t, hhmm),
		Open:      o, High: h, Low: l, Close: c, Volume: 1000,
	}
}

// collect builds an engine whose sink appends to a slice.
func collect(t *testing.T, cfg *Config) (*Engine, *[]Order) {
	t.Helper()
	got := new([]Order)
	sink := FuncSink(func(o Order) bool {
		*got = append(*got, o)
		return true
	})
	eng, err := NewEngine(cfg, sink)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	return eng, got
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

func TestRegistryBuiltinsAreRegistered(t *testing.T) {
	for _, want := range []string{"intraday_trend", "scalping_mean_reversion", "swing_macd"} {
		if !IsRegistered(want) {
			t.Errorf("built-in strategy %q is not registered; known: %v", want, Names())
		}
	}
}

func TestRegistryUnknownStrategy(t *testing.T) {
	if _, err := New("does_not_exist", nil); err == nil {
		t.Fatal("expected error for unknown strategy, got nil")
	}
}

func TestRegistryDuplicatePanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on duplicate registration")
		}
	}()
	Register("intraday_trend", func() Strategy { return &IntradayTrend{} })
}

func TestNewAppliesParams(t *testing.T) {
	s, err := New("intraday_trend", Params{"min_adx": 33})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := s.(*IntradayTrend).minADX; got != 33 {
		t.Errorf("min_adx = %v, want 33", got)
	}
}

func TestInitRejectsBadParams(t *testing.T) {
	// rsi_long_min above rsi_long_max is contradictory and must not start.
	if _, err := New("intraday_trend", Params{"rsi_long_min": 80, "rsi_long_max": 60}); err == nil {
		t.Fatal("expected error for inverted RSI bounds, got nil")
	}
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

func TestConfigDefaults(t *testing.T) {
	cfg := &Config{}
	if err := cfg.Normalize(); err != nil {
		t.Fatalf("normalize: %v", err)
	}
	if cfg.SignalBuffer != 4096 {
		t.Errorf("SignalBuffer = %d, want 4096", cfg.SignalBuffer)
	}
	if cfg.Capital != 1_000_000 {
		t.Errorf("Capital = %v, want 1000000", cfg.Capital)
	}
	if cfg.Session.Timezone != "Asia/Kolkata" {
		t.Errorf("Timezone = %q, want Asia/Kolkata", cfg.Session.Timezone)
	}
}

func TestConfigRejectsUnknownStrategy(t *testing.T) {
	cfg := &Config{Strategies: []InstanceConfig{{Name: "nope", Enabled: true, Intervals: []string{"1m"}}}}
	if err := cfg.Normalize(); err == nil {
		t.Fatal("expected error for unregistered strategy, got nil")
	}
}

func TestConfigRejectsMissingIntervals(t *testing.T) {
	cfg := &Config{Strategies: []InstanceConfig{{Name: "scripted", Enabled: true}}}
	if err := cfg.Normalize(); err == nil {
		t.Fatal("expected error for missing intervals, got nil")
	}
}

func TestConfigDisabledStrategyIsNotValidated(t *testing.T) {
	// A disabled entry may reference anything; it never runs.
	cfg := &Config{Strategies: []InstanceConfig{{Name: "nope", Enabled: false}}}
	if err := cfg.Normalize(); err != nil {
		t.Fatalf("disabled strategy should not be validated: %v", err)
	}
}

func TestSessionWindow(t *testing.T) {
	cfg := testConfig(t)
	s := &cfg.Session

	if s.AllowsEntry(at(t, "09:00")) {
		t.Error("09:00 is before the open but AllowsEntry returned true")
	}
	if !s.AllowsEntry(at(t, "10:00")) {
		t.Error("10:00 is mid-session but AllowsEntry returned false")
	}
	if s.AllowsEntry(at(t, "15:20")) {
		t.Error("15:20 is past square-off but AllowsEntry returned true")
	}
	if !s.IsSquareOff(at(t, "15:15")) {
		t.Error("15:15 is the square-off time but IsSquareOff returned false")
	}
	if s.IsSquareOff(at(t, "14:00")) {
		t.Error("14:00 is mid-session but IsSquareOff returned true")
	}
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

func TestEngineEntryThenTargetExit(t *testing.T) {
	cfg := testConfig(t)
	eng, got := collect(t, cfg)

	// Entry at 100. Default target is +2% => 102.
	eng.OnCandle(frame(t, "10:00", 100, 100, 100, 100)) // builds the slot
	// The slot built its own instance; script it via the package-level hook.
	current.next = Signal{Action: ActionEntry, Side: SideLong, Reason: ReasonTrendUp, Price: 100}

	eng.OnCandle(frame(t, "10:05", 100, 100, 100, 100))
	if len(*got) != 1 {
		t.Fatalf("expected 1 order after entry, got %d: %+v", len(*got), *got)
	}
	entry := (*got)[0]
	if entry.Action != ActionEntry || entry.Side != SideLong {
		t.Fatalf("unexpected entry order: %+v", entry)
	}
	if entry.Quantity <= 0 {
		t.Errorf("entry quantity = %d, want positive", entry.Quantity)
	}
	// 5% of 1,000,000 at price 100 = 500 shares.
	if entry.Quantity != 500 {
		t.Errorf("entry quantity = %d, want 500", entry.Quantity)
	}
	if entry.TargetPrice != 102 {
		t.Errorf("target = %v, want 102", entry.TargetPrice)
	}

	// Next candle trades through the target.
	current.next = None
	eng.OnCandle(frame(t, "10:10", 100, 103, 100, 103))
	if len(*got) != 2 {
		t.Fatalf("expected exit order, got %d orders", len(*got))
	}
	exit := (*got)[1]
	if exit.Action != ActionExit || exit.Reason != ReasonTakeProfit {
		t.Errorf("unexpected exit: action=%v reason=%v", exit.Action, exit.Reason)
	}
	if exit.Price != 102 {
		t.Errorf("exit price = %v, want 102 (the target)", exit.Price)
	}
}

func TestEngineStopLossTakesPrecedenceOverTarget(t *testing.T) {
	cfg := testConfig(t)
	eng, got := collect(t, cfg)

	eng.OnCandle(frame(t, "10:00", 100, 100, 100, 100))
	current.next = Signal{Action: ActionEntry, Side: SideLong, Price: 100}
	eng.OnCandle(frame(t, "10:05", 100, 100, 100, 100))
	current.next = None

	// This candle touches both the stop (99) and the target (102). The engine
	// must assume the adverse fill.
	eng.OnCandle(frame(t, "10:10", 100, 103, 98, 101))
	if len(*got) != 2 {
		t.Fatalf("expected 2 orders, got %d", len(*got))
	}
	if r := (*got)[1].Reason; r != ReasonStopLoss {
		t.Errorf("exit reason = %v, want stop_loss", r)
	}
}

func TestEngineSquareOffClosesPosition(t *testing.T) {
	cfg := testConfig(t)
	eng, got := collect(t, cfg)

	eng.OnCandle(frame(t, "10:00", 100, 100, 100, 100))
	current.next = Signal{Action: ActionEntry, Side: SideLong, Price: 100}
	eng.OnCandle(frame(t, "10:05", 100, 100, 100, 100))
	current.next = None

	eng.OnCandle(frame(t, "15:20", 100, 100, 100, 100))
	if len(*got) != 2 {
		t.Fatalf("expected square-off exit, got %d orders", len(*got))
	}
	if r := (*got)[1].Reason; r != ReasonSessionEnd {
		t.Errorf("exit reason = %v, want session_end", r)
	}
}

func TestEngineRejectsEntryOutsideSession(t *testing.T) {
	cfg := testConfig(t)
	eng, got := collect(t, cfg)

	eng.OnCandle(frame(t, "15:20", 100, 100, 100, 100))
	current.next = Signal{Action: ActionEntry, Side: SideLong, Price: 100}
	eng.OnCandle(frame(t, "15:25", 100, 100, 100, 100))

	if len(*got) != 0 {
		t.Fatalf("expected no orders outside session, got %+v", *got)
	}
	if eng.Stats().Rejects == 0 {
		t.Error("expected a risk rejection to be counted")
	}
}

func TestEngineRejectsShortWhenDisallowed(t *testing.T) {
	cfg := testConfig(t)
	cfg.Risk.AllowShort = false
	eng, got := collect(t, cfg)

	eng.OnCandle(frame(t, "10:00", 100, 100, 100, 100))
	current.next = Signal{Action: ActionEntry, Side: SideShort, Price: 100}
	eng.OnCandle(frame(t, "10:05", 100, 100, 100, 100))

	if len(*got) != 0 {
		t.Fatalf("expected short to be refused, got %+v", *got)
	}
}

func TestEngineEnforcesMaxTradesPerDay(t *testing.T) {
	cfg := testConfig(t)
	cfg.Risk.MaxTradesPerDay = 2
	eng, got := collect(t, cfg)

	eng.OnCandle(frame(t, "10:00", 100, 100, 100, 100))

	// Three round trips; only two entries may be accepted.
	for i := 0; i < 3; i++ {
		current.next = Signal{Action: ActionEntry, Side: SideLong, Price: 100}
		eng.OnCandle(frame(t, "10:05", 100, 100, 100, 100))
		current.next = Signal{Action: ActionExit, Side: SideLong, Price: 100}
		eng.OnCandle(frame(t, "10:10", 100, 100, 100, 100))
	}

	entries := 0
	for _, o := range *got {
		if o.Action == ActionEntry {
			entries++
		}
	}
	if entries != 2 {
		t.Errorf("entries = %d, want 2 (daily cap)", entries)
	}
}

func TestEngineNoPyramiding(t *testing.T) {
	cfg := testConfig(t)
	eng, got := collect(t, cfg)

	eng.OnCandle(frame(t, "10:00", 100, 100, 100, 100))
	current.next = Signal{Action: ActionEntry, Side: SideLong, Price: 100}
	eng.OnCandle(frame(t, "10:05", 100, 100, 100, 100))
	// Same entry signal again while already long.
	eng.OnCandle(frame(t, "10:06", 100, 100, 100, 100))

	entries := 0
	for _, o := range *got {
		if o.Action == ActionEntry {
			entries++
		}
	}
	if entries != 1 {
		t.Errorf("entries = %d, want 1 (no pyramiding)", entries)
	}
}

func TestEngineIsolatesStrategyPanic(t *testing.T) {
	cfg := testConfig(t)
	eng, got := collect(t, cfg)

	eng.OnCandle(frame(t, "10:00", 100, 100, 100, 100))
	current.panicOnCandle = true

	// Must not propagate; the market data goroutine has to survive.
	eng.OnCandle(frame(t, "10:05", 100, 100, 100, 100))

	if len(*got) != 0 {
		t.Errorf("panicking strategy produced orders: %+v", *got)
	}
	if eng.Stats().Panics != 1 {
		t.Errorf("panics = %d, want 1", eng.Stats().Panics)
	}
	current.panicOnCandle = false
}

func TestEngineRespectsWarmup(t *testing.T) {
	cfg := testConfig(t)
	cfg.Strategies[0].Params = Params{"warmup": 3}
	eng, got := collect(t, cfg)

	eng.OnCandle(frame(t, "10:00", 100, 100, 100, 100))
	current.next = Signal{Action: ActionEntry, Side: SideLong, Price: 100}

	// Bars 2 and 3 are still within warm-up (bar 1 built the slot).
	eng.OnCandle(frame(t, "10:05", 100, 100, 100, 100))
	eng.OnCandle(frame(t, "10:10", 100, 100, 100, 100))
	if len(*got) != 0 {
		t.Fatalf("signals emitted during warm-up: %+v", *got)
	}

	// Bar 4 is past warm-up.
	eng.OnCandle(frame(t, "10:15", 100, 100, 100, 100))
	if len(*got) != 1 {
		t.Errorf("expected entry after warm-up, got %d orders", len(*got))
	}
}

func TestEngineSkipsUntilIndicatorsReady(t *testing.T) {
	cfg := testConfig(t)
	eng, got := collect(t, cfg)
	eng.OnCandle(frame(t, "10:00", 100, 100, 100, 100))

	// Demand an indicator the frames never carry.
	for _, inst := range eng.slots[key{token: 111, interval: "5m"}].instances {
		inst.req = HasADX
	}
	current.next = Signal{Action: ActionEntry, Side: SideLong, Price: 100}
	eng.OnCandle(frame(t, "10:05", 100, 100, 100, 100))

	if len(*got) != 0 {
		t.Errorf("emitted orders before indicators were ready: %+v", *got)
	}
}

func TestEngineDisabledDoesNothing(t *testing.T) {
	cfg := testConfig(t)
	cfg.Enabled = false
	eng, got := collect(t, cfg)

	eng.OnCandle(frame(t, "10:00", 100, 100, 100, 100))
	eng.OnCandle(frame(t, "10:05", 100, 100, 100, 100))
	if len(*got) != 0 {
		t.Errorf("disabled engine emitted orders: %+v", *got)
	}
}

func TestEngineIntervalRouting(t *testing.T) {
	cfg := testConfig(t) // scripted runs on "5m" only
	eng, got := collect(t, cfg)

	f := frame(t, "10:00", 100, 100, 100, 100)
	f.Interval = "1m"
	eng.OnCandle(f)
	eng.OnCandle(f)

	if len(*got) != 0 {
		t.Errorf("strategy ran on an interval it was not configured for: %+v", *got)
	}
}

func TestRealizedPnL(t *testing.T) {
	if got := realizedPnL(SideLong, 100, 110, 10); got != 100 {
		t.Errorf("long pnl = %v, want 100", got)
	}
	if got := realizedPnL(SideShort, 100, 90, 10); got != 100 {
		t.Errorf("short pnl = %v, want 100", got)
	}
	if got := realizedPnL(SideNone, 100, 90, 10); got != 0 {
		t.Errorf("flat pnl = %v, want 0", got)
	}
}

// ---------------------------------------------------------------------------
// Sink
// ---------------------------------------------------------------------------

func TestChannelSinkDropsWhenFull(t *testing.T) {
	s := NewChannelSink(1)
	if !s.Emit(Order{}) {
		t.Fatal("first emit should be accepted")
	}
	if s.Emit(Order{}) {
		t.Fatal("second emit should be dropped on a full buffer")
	}
	if s.Dropped() != 1 {
		t.Errorf("dropped = %d, want 1", s.Dropped())
	}
}

// ---------------------------------------------------------------------------
// Built-in strategies
// ---------------------------------------------------------------------------

func TestIntradayTrendGoesLong(t *testing.T) {
	s, err := New("intraday_trend", Params{"min_adx": 20})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	f := frame(t, "10:00", 100, 101, 99, 101)
	f.Ind = IndicatorSnapshot{
		Present: HasEMA | HasRSI | HasVWAP | HasATR | HasADX,
		EMA:     99.5, RSI: 60, VWAP: 100, ATR: 1, ADX: 30,
	}
	sig := s.OnCandle(f, &Position{})
	if sig.Action != ActionEntry || sig.Side != SideLong {
		t.Fatalf("expected long entry, got %+v", sig)
	}
	if sig.StopPrice != 101-1.5 {
		t.Errorf("stop = %v, want %v (ATR-scaled)", sig.StopPrice, 101-1.5)
	}
}

func TestIntradayTrendSkipsWeakTrend(t *testing.T) {
	s, _ := New("intraday_trend", Params{"min_adx": 25})
	f := frame(t, "10:00", 100, 101, 99, 101)
	f.Ind = IndicatorSnapshot{
		Present: HasEMA | HasRSI | HasVWAP | HasATR | HasADX,
		EMA:     99.5, RSI: 60, VWAP: 100, ATR: 1, ADX: 10, // ADX below threshold
	}
	if sig := s.OnCandle(f, &Position{}); sig.IsActionable() {
		t.Errorf("expected no signal in a weak trend, got %+v", sig)
	}
}

func TestScalpingFadesLowerBand(t *testing.T) {
	s, err := New("scalping_mean_reversion", nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	f := frame(t, "10:00", 100, 100, 97, 97)
	f.Ind = IndicatorSnapshot{
		Present: HasBollinger | HasRSI,
		BBUpper: 103, BBMiddle: 100, BBLower: 97,
		RSI: 25,
	}
	sig := s.OnCandle(f, &Position{})
	if sig.Action != ActionEntry || sig.Side != SideLong {
		t.Fatalf("expected long mean-reversion entry, got %+v", sig)
	}
	if sig.TargetPrice != 100 {
		t.Errorf("target = %v, want the middle band 100", sig.TargetPrice)
	}
}

func TestScalpingTimeExit(t *testing.T) {
	s, _ := New("scalping_mean_reversion", Params{"max_bars_in_trade": 3})
	f := frame(t, "10:00", 100, 100, 97, 97)
	f.Ind = IndicatorSnapshot{Present: HasBollinger | HasRSI, BBUpper: 103, BBMiddle: 100, BBLower: 97, RSI: 25}

	pos := &Position{Side: SideLong, EntryPrice: 97, Quantity: 10, BarsHeld: 3}
	sig := s.OnCandle(f, pos)
	if sig.Action != ActionExit || sig.Reason != ReasonTimeExit {
		t.Errorf("expected a time exit, got %+v", sig)
	}
}

func TestSwingNeedsTwoBarsForCrossover(t *testing.T) {
	s, _ := New("swing_macd", nil)
	f := frame(t, "10:00", 100, 101, 99, 101)
	f.Ind = IndicatorSnapshot{Present: HasMACD | HasADX, MACDHist: 0.5, MACDLine: 1, MACDSignal: 0.5, ADX: 30}

	// First bar establishes the baseline; it must not signal.
	if sig := s.OnCandle(f, &Position{}); sig.IsActionable() {
		t.Errorf("first bar signalled without a previous value: %+v", sig)
	}
}

func TestSwingCrossoverEntry(t *testing.T) {
	s, _ := New("swing_macd", nil)
	f := frame(t, "10:00", 100, 101, 99, 101)

	// Bar 1: histogram negative.
	f.Ind = IndicatorSnapshot{Present: HasMACD | HasADX, MACDHist: -0.2, MACDLine: 0.1, MACDSignal: 0.3, ADX: 30}
	s.OnCandle(f, &Position{})

	// Bar 2: histogram flips positive => cross up.
	f.Ind = IndicatorSnapshot{Present: HasMACD | HasADX, MACDHist: 0.4, MACDLine: 0.7, MACDSignal: 0.3, ADX: 30}
	sig := s.OnCandle(f, &Position{})
	if sig.Action != ActionEntry || sig.Side != SideLong {
		t.Fatalf("expected long entry on MACD cross up, got %+v", sig)
	}
}

// ---------------------------------------------------------------------------
// Selector
// ---------------------------------------------------------------------------

func TestSelectorClassify(t *testing.T) {
	sel := NewSelector(DefaultSelectorConfig())

	trending := &Frame{Close: 100, Ind: IndicatorSnapshot{Present: HasADX | HasATR, ADX: 30, ATR: 0.5}}
	if got := sel.Classify(trending); got != RegimeTrending {
		t.Errorf("classify = %v, want trending", got)
	}

	ranging := &Frame{Close: 100, Ind: IndicatorSnapshot{Present: HasADX | HasATR, ADX: 15, ATR: 0.5}}
	if got := sel.Classify(ranging); got != RegimeRanging {
		t.Errorf("classify = %v, want ranging", got)
	}

	// High ATR dominates even with a strong ADX.
	volatile := &Frame{Close: 100, Ind: IndicatorSnapshot{Present: HasADX | HasATR, ADX: 40, ATR: 5}}
	if got := sel.Classify(volatile); got != RegimeVolatile {
		t.Errorf("classify = %v, want volatile", got)
	}

	unknown := &Frame{Close: 100, Ind: IndicatorSnapshot{}}
	if got := sel.Classify(unknown); got != RegimeUnknown {
		t.Errorf("classify = %v, want unknown", got)
	}
}

// ---------------------------------------------------------------------------
// Benchmarks — the hot path must stay allocation-free
// ---------------------------------------------------------------------------

// BenchmarkEngineOnCandleNoSignal measures the common case: a candle that
// produces no order. This runs once per finalized candle per instrument.
func BenchmarkEngineOnCandleNoSignal(b *testing.B) {
	cfg := &Config{
		Enabled: true,
		Session: SessionConfig{Timezone: "Asia/Kolkata"},
		Strategies: []InstanceConfig{
			{Name: "intraday_trend", Enabled: true, Intervals: []string{"5m"}},
		},
	}
	if err := cfg.Normalize(); err != nil {
		b.Fatalf("normalize: %v", err)
	}
	eng, err := NewEngine(cfg, DiscardSink)
	if err != nil {
		b.Fatalf("NewEngine: %v", err)
	}

	loc, _ := time.LoadLocation("Asia/Kolkata")
	f := &Frame{
		Token: 1, Interval: "5m",
		Timestamp: time.Date(2026, 9, 11, 10, 0, 0, 0, loc),
		Open:      100, High: 101, Low: 99, Close: 100, Volume: 1000,
		Ind: IndicatorSnapshot{
			Present: HasEMA | HasRSI | HasVWAP | HasATR | HasADX,
			// RSI of 50 is outside both entry bands, so no signal fires.
			EMA: 100, RSI: 50, VWAP: 100, ATR: 1, ADX: 30,
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		eng.OnCandle(f)
	}
}

// BenchmarkIntradayOnCandle isolates a single strategy evaluation.
func BenchmarkIntradayOnCandle(b *testing.B) {
	s, err := New("intraday_trend", nil)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	f := &Frame{
		Open: 100, High: 101, Low: 99, Close: 100,
		Ind: IndicatorSnapshot{
			Present: HasEMA | HasRSI | HasVWAP | HasATR | HasADX,
			EMA:     100, RSI: 50, VWAP: 100, ATR: 1, ADX: 30,
		},
	}
	pos := &Position{}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.OnCandle(f, pos)
	}
}
