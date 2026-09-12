// Package backtest replays historical candles through the live strategy
// engine and paper broker.
//
// The point of reusing those components rather than reimplementing them is
// that a backtest result only means something if the code under test is the
// code that will trade. A separate simulation path is a separate set of bugs.
package backtest

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/db"
	monitor "github.com/Bhavik2205/ML-Bot/internal/execution"
	"github.com/Bhavik2205/ML-Bot/internal/indicators"
	"github.com/Bhavik2205/ML-Bot/internal/strategy"
	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"gorm.io/gorm"
)

// priceScale matches candles.PriceScale.
const priceScale = 10000

// Request describes one backtest run.
type Request struct {
	StrategyName   string          `json:"strategy_name"`
	Instruments    []uint32        `json:"instruments"`
	Interval       string          `json:"interval"`
	From           time.Time       `json:"from"`
	To             time.Time       `json:"to"`
	InitialCapital float64         `json:"initial_capital"`
	Params         strategy.Params `json:"params"`
	// SlippageBps and FeeBps override the paper defaults.
	SlippageBps float64 `json:"slippage_bps"`
	FeeBps      float64 `json:"fee_bps"`
}

// Validate applies defaults and rejects incoherent requests.
func (r *Request) Validate() error {
	if r.StrategyName == "" {
		return fmt.Errorf("strategy_name is required")
	}
	if !strategy.IsRegistered(r.StrategyName) {
		return fmt.Errorf("strategy %q is not registered (known: %v)", r.StrategyName, strategy.Names())
	}
	if len(r.Instruments) == 0 {
		return fmt.Errorf("at least one instrument is required")
	}
	if r.Interval == "" {
		r.Interval = "5m"
	}
	if r.InitialCapital <= 0 {
		r.InitialCapital = 1_000_000
	}
	if r.To.IsZero() {
		r.To = time.Now()
	}
	if r.From.IsZero() {
		r.From = r.To.AddDate(0, 0, -30)
	}
	if !r.From.Before(r.To) {
		return fmt.Errorf("from (%s) must be before to (%s)", r.From.Format(time.RFC3339), r.To.Format(time.RFC3339))
	}
	if r.SlippageBps < 0 || r.FeeBps < 0 {
		return fmt.Errorf("slippage_bps and fee_bps must not be negative")
	}
	if r.Params == nil {
		r.Params = strategy.Params{}
	}
	return nil
}

// TradeLog is one simulated fill.
type TradeLog struct {
	Timestamp       time.Time `json:"timestamp"`
	InstrumentToken uint32    `json:"instrument_token"`
	Symbol          string    `json:"symbol"`
	Action          string    `json:"action"`
	Side            string    `json:"side"`
	Reason          string    `json:"reason"`
	Quantity        int64     `json:"quantity"`
	Price           float64   `json:"price"`
}

// EquityPoint is the account value at a point in time.
type EquityPoint struct {
	Timestamp time.Time `json:"timestamp"`
	Equity    float64   `json:"equity"`
}

// Result summarises a completed run.
type Result struct {
	StrategyName   string    `json:"strategy_name"`
	Interval       string    `json:"interval"`
	From           time.Time `json:"from"`
	To             time.Time `json:"to"`
	InitialCapital float64   `json:"initial_capital"`
	FinalEquity    float64   `json:"final_equity"`

	CandlesReplayed int `json:"candles_replayed"`
	TotalTrades     int `json:"total_trades"`
	// RoundTrips counts completed entry+exit pairs; a trade left open at the
	// end is not counted, because its outcome is not known.
	RoundTrips int `json:"round_trips"`
	Wins       int `json:"wins"`
	Losses     int `json:"losses"`

	RealizedPnL    float64 `json:"realized_pnl"`
	UnrealizedPnL  float64 `json:"unrealized_pnl"`
	FeesPaid       float64 `json:"fees_paid"`
	NetPnL         float64 `json:"net_pnl"`
	ReturnPct      float64 `json:"return_pct"`
	WinRatePct     float64 `json:"win_rate_pct"`
	MaxDrawdownPct float64 `json:"max_drawdown_pct"`
	// Sharpe is annualised from per-bar returns. It is meaningless on a
	// handful of trades; treat it as indicative only.
	Sharpe float64 `json:"sharpe"`

	Trades      []TradeLog    `json:"trades"`
	EquityCurve []EquityPoint `json:"equity_curve"`

	DurationMs int64 `json:"duration_ms"`
}

// Runner executes backtests against candles stored in the database.
type Runner struct {
	db            *gorm.DB
	indicatorsCfg *utils.IndicatorsConfig
}

// NewRunner builds a Runner.
func NewRunner(dbc *db.DBClient, indicatorsCfg *utils.IndicatorsConfig) *Runner {
	return &Runner{db: dbc.DB, indicatorsCfg: indicatorsCfg}
}

// maxCandles bounds a single run so a careless date range cannot exhaust
// memory or block the request for minutes.
const maxCandles = 200_000

// Run replays history and returns the result.
func (rn *Runner) Run(ctx context.Context, req Request) (*Result, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	start := time.Now()

	rows, err := rn.loadCandles(ctx, req)
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("no candles found for instruments %v interval %s between %s and %s",
			req.Instruments, req.Interval, req.From.Format(time.RFC3339), req.To.Format(time.RFC3339))
	}

	// Build the engine exactly as production does, with the session window
	// wide open: historical bars carry their own timestamps and we are not
	// re-litigating market hours here.
	cfg := &strategy.Config{
		Enabled: true,
		Capital: req.InitialCapital,
		Session: strategy.SessionConfig{
			Timezone: "Asia/Kolkata", Open: "00:00", Close: "23:59", SquareOff: "23:58",
		},
		Risk: strategy.RiskConfig{
			MaxTradesPerDay:              1_000_000,
			MaxOpenPositions:             len(req.Instruments),
			TradeSizePercentageOfCapital: 0.05,
			MaxLossPerDayPercentage:      1.0,
			AllowShort:                   true,
		},
		Strategies: []strategy.InstanceConfig{{
			Name:        req.StrategyName,
			Enabled:     true,
			Intervals:   []string{req.Interval},
			Instruments: req.Instruments,
			Params:      req.Params,
		}},
	}
	if err := cfg.Normalize(); err != nil {
		return nil, fmt.Errorf("build engine config: %w", err)
	}

	paperCfg := monitor.PaperConfig{
		SlippageBps:       req.SlippageBps,
		FeeBps:            req.FeeBps,
		RejectOutsideBand: true,
	}
	paper := monitor.NewPaperBroker(paperCfg, 0, nil)

	var trades []TradeLog
	symbols := rn.symbolsFor(ctx, req.Instruments)

	// The sink applies each order to the paper broker synchronously, so the
	// position book is correct before the next bar is replayed.
	sink := strategy.FuncSink(func(o strategy.Order) bool {
		if o.Symbol == "" {
			o.Symbol = symbols[o.Token]
		}
		if _, err := paper.Place(ctx, o); err != nil {
			return false
		}
		trades = append(trades, TradeLog{
			Timestamp: o.Timestamp, InstrumentToken: o.Token, Symbol: o.Symbol,
			Action: o.Action.String(), Side: o.Side.String(), Reason: o.Reason.String(),
			Quantity: o.Quantity, Price: o.Price,
		})
		return true
	})

	eng, err := strategy.NewEngine(cfg, sink)
	if err != nil {
		return nil, fmt.Errorf("build engine: %w", err)
	}

	// Per-instrument indicator state, fed bar by bar exactly as the live
	// IndicatorManager does.
	states := make(map[uint32]*indicators.IndicatorStateSet, len(req.Instruments))

	var equity []EquityPoint
	peak := req.InitialCapital
	maxDD := 0.0
	var barReturns []float64
	prevEquity := req.InitialCapital

	for i := range rows {
		c := &rows[i]
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		st, ok := states[c.InstrumentToken]
		if !ok {
			st = rn.newStateSet()
			states[c.InstrumentToken] = st
		}

		frame := strategy.Frame{
			Token:     c.InstrumentToken,
			Interval:  c.Interval,
			Symbol:    symbols[c.InstrumentToken],
			Timestamp: c.Timestamp,
			Open:      float64(c.Open) / priceScale,
			High:      float64(c.High) / priceScale,
			Low:       float64(c.Low) / priceScale,
			Close:     float64(c.Close) / priceScale,
			Volume:    float64(c.Volume),
		}
		frame.Ind = rn.updateIndicators(st, &frame)

		// Mark before evaluating so unrealised PnL and any protective exit
		// use this bar's price, not the previous one's.
		paper.Mark(c.InstrumentToken, frame.Close, c.Timestamp)
		eng.OnCandle(&frame)

		realized, unrealized := paper.Book().Totals()
		eq := req.InitialCapital + realized + unrealized - paper.FeesPaid()
		equity = append(equity, EquityPoint{Timestamp: c.Timestamp, Equity: eq})

		if eq > peak {
			peak = eq
		}
		if peak > 0 {
			if dd := (peak - eq) / peak * 100; dd > maxDD {
				maxDD = dd
			}
		}
		if prevEquity > 0 {
			barReturns = append(barReturns, (eq-prevEquity)/prevEquity)
		}
		prevEquity = eq
	}

	stats := paper.Stats()
	wins, losses, roundTrips := countOutcomes(trades)

	final := req.InitialCapital + stats.RealizedPnL + stats.UnrealizedPnL - stats.FeesPaid
	res := &Result{
		StrategyName:   req.StrategyName,
		Interval:       req.Interval,
		From:           req.From,
		To:             req.To,
		InitialCapital: req.InitialCapital,
		FinalEquity:    final,

		CandlesReplayed: len(rows),
		TotalTrades:     len(trades),
		RoundTrips:      roundTrips,
		Wins:            wins,
		Losses:          losses,

		RealizedPnL:   stats.RealizedPnL,
		UnrealizedPnL: stats.UnrealizedPnL,
		FeesPaid:      stats.FeesPaid,
		NetPnL:        stats.RealizedPnL + stats.UnrealizedPnL - stats.FeesPaid,

		MaxDrawdownPct: maxDD,
		Sharpe:         sharpe(barReturns, req.Interval),

		Trades:      trades,
		EquityCurve: downsample(equity, 500),
		DurationMs:  time.Since(start).Milliseconds(),
	}
	if req.InitialCapital > 0 {
		res.ReturnPct = (final - req.InitialCapital) / req.InitialCapital * 100
	}
	if roundTrips > 0 {
		res.WinRatePct = float64(wins) / float64(roundTrips) * 100
	}
	return res, nil
}

// loadCandles fetches the replay window ordered by time, so instruments are
// interleaved chronologically rather than replayed one symbol at a time.
func (rn *Runner) loadCandles(ctx context.Context, req Request) ([]db.OHLCVCandle, error) {
	var rows []db.OHLCVCandle
	err := rn.db.WithContext(ctx).
		Where("instrument_token IN ? AND interval = ? AND timestamp >= ? AND timestamp <= ?",
			req.Instruments, req.Interval, req.From, req.To).
		Order("timestamp ASC").
		Limit(maxCandles).
		Find(&rows).Error
	if err != nil {
		return nil, fmt.Errorf("load candles: %w", err)
	}
	return rows, nil
}

// symbolsFor resolves display symbols once up front.
func (rn *Runner) symbolsFor(ctx context.Context, tokens []uint32) map[uint32]string {
	out := make(map[uint32]string, len(tokens))
	var rows []db.Instrument
	if err := rn.db.WithContext(ctx).Where("instrument_token IN ?", tokens).Find(&rows).Error; err == nil {
		for _, in := range rows {
			out[in.InstrumentToken] = in.Tradingsymbol
		}
	}
	return out
}

func (rn *Runner) newStateSet() *indicators.IndicatorStateSet {
	cfg := rn.indicatorsCfg
	s := &indicators.IndicatorStateSet{}
	if cfg == nil {
		return s
	}
	if cfg.SMA.Enabled {
		s.SMA = indicators.NewSMAState(cfg.SMA.Period)
	}
	if cfg.EMA.Enabled {
		s.EMA = indicators.NewEMAState(cfg.EMA.ShortPeriod)
	}
	if cfg.RSI.Enabled {
		s.RSI = indicators.NewRSIState(cfg.RSI.Period)
	}
	if cfg.MACD.Enabled {
		s.MACD = indicators.NewMACDState(cfg.MACD.FastPeriod, cfg.MACD.SlowPeriod, cfg.MACD.SignalPeriod)
	}
	if cfg.ATR.Enabled {
		s.ATR = indicators.NewATRState(cfg.ATR.Period)
	}
	if cfg.BollingerBands.Enabled {
		s.Bollinger = indicators.NewBollingerState(cfg.BollingerBands.Period, cfg.BollingerBands.NumStdDev)
	}
	if cfg.Stochastic.Enabled {
		s.Stochastic = indicators.NewStochasticState(cfg.Stochastic.KPeriod, cfg.Stochastic.DPeriod)
	}
	if cfg.OBV.Enabled {
		s.OBV = indicators.NewOBVState()
	}
	if cfg.VWAP.Enabled {
		s.VWAP = indicators.NewVWAPState()
	}
	if cfg.ADX.Enabled {
		s.ADX = indicators.NewADXState(cfg.ADX.Period)
	}
	return s
}

// updateIndicators advances every configured indicator by one bar and returns
// the snapshot, mirroring IndicatorManager.processCandle.
func (rn *Runner) updateIndicators(s *indicators.IndicatorStateSet, f *strategy.Frame) strategy.IndicatorSnapshot {
	var snap strategy.IndicatorSnapshot

	if s.SMA != nil {
		if v, ok := s.SMA.Update(f.Close); ok {
			snap.SMA, snap.Present = v, snap.Present|strategy.HasSMA
		}
	}
	if s.EMA != nil {
		if v, ok := s.EMA.Update(f.Close); ok {
			snap.EMA, snap.Present = v, snap.Present|strategy.HasEMA
		}
	}
	if s.RSI != nil {
		if v, ok := s.RSI.Update(f.Close); ok {
			snap.RSI, snap.Present = v, snap.Present|strategy.HasRSI
		}
	}
	if s.MACD != nil {
		if line, sig, hist, ok := s.MACD.Update(f.Close); ok {
			snap.MACDLine, snap.MACDSignal, snap.MACDHist = line, sig, hist
			snap.Present |= strategy.HasMACD
		}
	}
	if s.ATR != nil {
		if v, ok := s.ATR.Update(f.High, f.Low, f.Close); ok {
			snap.ATR, snap.Present = v, snap.Present|strategy.HasATR
		}
	}
	if s.Bollinger != nil {
		if up, mid, low, ok := s.Bollinger.Update(f.Close); ok {
			snap.BBUpper, snap.BBMiddle, snap.BBLower = up, mid, low
			snap.Present |= strategy.HasBollinger
		}
	}
	if s.Stochastic != nil {
		if k, d, ok := s.Stochastic.Update(f.High, f.Low, f.Close); ok {
			snap.StochK, snap.StochD = k, d
			snap.Present |= strategy.HasStochastic
		}
	}
	if s.OBV != nil {
		if v, ok := s.OBV.Update(f.Close, f.Volume); ok {
			snap.OBV, snap.Present = v, snap.Present|strategy.HasOBV
		}
	}
	if s.VWAP != nil {
		if v, ok := s.VWAP.Update(f.Open, f.High, f.Low, f.Close, f.Volume); ok {
			snap.VWAP, snap.Present = v, snap.Present|strategy.HasVWAP
		}
	}
	if s.ADX != nil {
		if adx, plus, minus, ok := s.ADX.Update(f.High, f.Low, f.Close); ok {
			snap.ADX, snap.PlusDI, snap.MinusDI = adx, plus, minus
			snap.Present |= strategy.HasADX
		}
	}
	return snap
}

// countOutcomes pairs entries with their exits per instrument to classify
// round trips. An entry still open at the end of the run is not counted,
// because its outcome is unknown.
func countOutcomes(trades []TradeLog) (wins, losses, roundTrips int) {
	type openPos struct {
		side  string
		price float64
		qty   int64
	}
	open := make(map[uint32]*openPos)

	for _, t := range trades {
		switch t.Action {
		case "ENTRY":
			open[t.InstrumentToken] = &openPos{side: t.Side, price: t.Price, qty: t.Quantity}
		case "EXIT":
			p := open[t.InstrumentToken]
			if p == nil {
				continue
			}
			var pnl float64
			if p.side == "LONG" {
				pnl = (t.Price - p.price) * float64(p.qty)
			} else {
				pnl = (p.price - t.Price) * float64(p.qty)
			}
			roundTrips++
			if pnl > 0 {
				wins++
			} else if pnl < 0 {
				losses++
			}
			delete(open, t.InstrumentToken)
		}
	}
	return wins, losses, roundTrips
}

// sharpe annualises the mean/stddev of per-bar returns.
func sharpe(returns []float64, interval string) float64 {
	if len(returns) < 2 {
		return 0
	}
	var sum float64
	for _, r := range returns {
		sum += r
	}
	mean := sum / float64(len(returns))

	var variance float64
	for _, r := range returns {
		d := r - mean
		variance += d * d
	}
	variance /= float64(len(returns) - 1)
	sd := math.Sqrt(variance)
	if sd == 0 {
		return 0
	}

	return mean / sd * math.Sqrt(barsPerYear(interval))
}

// barsPerYear assumes 252 sessions of 375 minutes.
func barsPerYear(interval string) float64 {
	d, err := time.ParseDuration(interval)
	if err != nil || d <= 0 {
		return 252
	}
	barsPerDay := (375 * time.Minute) / d
	if barsPerDay < 1 {
		barsPerDay = 1
	}
	return 252 * float64(barsPerDay)
}

// downsample thins the equity curve to at most max points so the response
// stays small enough to chart without pagination.
func downsample(points []EquityPoint, max int) []EquityPoint {
	if len(points) <= max || max <= 0 {
		return points
	}
	step := float64(len(points)) / float64(max)
	out := make([]EquityPoint, 0, max)
	for i := 0; i < max; i++ {
		idx := int(float64(i) * step)
		if idx >= len(points) {
			idx = len(points) - 1
		}
		out = append(out, points[idx])
	}
	// Always keep the true final value; the sampled one may miss it.
	if last := points[len(points)-1]; out[len(out)-1].Timestamp != last.Timestamp {
		out = append(out, last)
	}
	return out
}
