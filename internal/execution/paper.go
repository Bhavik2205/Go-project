package monitor

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/strategy"
)

// TradeTypePaper and TradeTypeLive populate the trade_type column shared by
// the orders, trades and positions tables.
const (
	TradeTypePaper = "PAPER"
	TradeTypeLive  = "LIVE"
)

// PaperConfig tunes the fill simulator.
//
// Slippage moves the price against the order and costs are the real Indian
// statutory schedule. Modelling either as a favour is the classic way to build
// a backtest that cannot be reproduced live.
type PaperConfig struct {
	// Slippage covers spread crossing and volume-dependent market impact.
	Slippage SlippageModel `yaml:"slippage"`

	// Costs is the brokerage and statutory charge schedule.
	Costs CostModel `yaml:"costs"`

	// RejectOutsideBand rejects a fill whose reference price is zero or
	// negative, which indicates a bad or missing mark rather than a trade.
	RejectOutsideBand bool `yaml:"reject_outside_band"`
}

// DefaultPaperConfig returns conservative defaults for Indian equity intraday.
func DefaultPaperConfig() PaperConfig {
	return PaperConfig{
		Slippage:          DefaultSlippageModel(),
		Costs:             DefaultCostModel(),
		RejectOutsideBand: true,
	}
}

// FrictionlessPaperConfig disables slippage and charges. Use it to isolate
// strategy logic in tests, never to judge whether a strategy makes money.
func FrictionlessPaperConfig() PaperConfig {
	return PaperConfig{
		Slippage:          SlippageModel{},
		Costs:             ZeroCostModel(),
		RejectOutsideBand: true,
	}
}

// PaperBroker simulates execution against an in-memory position book. It
// implements Broker, so the executor cannot tell it from a live venue.
//
// It is intentionally pessimistic: fills cross the spread and pay fees, so a
// strategy that is marginal on paper will not look better than it is.
type PaperBroker struct {
	cfg  PaperConfig
	book *Book

	// sink persists orders, trades and positions. Optional: nil means the
	// paper run is memory-only.
	sink TradeSink

	// userID owns the simulated account, needed for the DB foreign keys.
	userID uint

	seq atomic.Uint64

	// costMu guards the itemised cost totals. Fills are infrequent relative
	// to candles, so a mutex here is cheaper than atomics per component.
	costMu    sync.Mutex
	costs     CostBreakdown
	slipBpsWt float64 // turnover-weighted slippage, for a meaningful average
	turnover  float64
	partial   uint64 // fills cut short by the participation cap
	rejected  uint64 // orders refused outright for lack of liquidity
}

// NewPaperBroker builds a paper broker. Pass a nil sink for memory-only runs.
func NewPaperBroker(cfg PaperConfig, userID uint, sink TradeSink) *PaperBroker {
	// Negative rates would pay the trader to trade; clamp rather than trust.
	if cfg.Slippage.HalfSpreadBps < 0 {
		cfg.Slippage.HalfSpreadBps = 0
	}
	if cfg.Slippage.ImpactCoefficient < 0 {
		cfg.Slippage.ImpactCoefficient = 0
	}
	if cfg.Slippage.MaxParticipation < 0 {
		cfg.Slippage.MaxParticipation = 0
	}
	return &PaperBroker{
		cfg:    cfg,
		book:   NewBook(),
		sink:   sink,
		userID: userID,
	}
}

// Book exposes the position book for the API layer.
func (b *PaperBroker) Book() *Book { return b.book }

// Mode reports the trade type this broker records.
func (b *PaperBroker) Mode() string { return TradeTypePaper }

// Place simulates an execution and nets it into the book.
func (b *PaperBroker) Place(_ context.Context, o strategy.Order) (string, error) {
	ref := o.Price
	if b.cfg.RejectOutsideBand && ref <= 0 {
		return "", fmt.Errorf("paper: refusing fill at non-positive reference price %v", ref)
	}
	if o.Quantity <= 0 {
		return "", fmt.Errorf("paper: refusing fill with quantity %d", o.Quantity)
	}

	buy := isBuy(o)

	// Liquidity first: the bar may not support the requested size.
	plan := b.cfg.Slippage.Plan(ref, o.Quantity, o.BarVolume, buy)
	if plan.Quantity <= 0 {
		b.countRejected()
		return "", fmt.Errorf("paper: bar volume %.0f cannot support an order of %d at %.2f",
			o.BarVolume, o.Quantity, ref)
	}
	if plan.Capped {
		b.countPartial()
	}

	fillPrice := plan.Price
	fillQty := plan.Quantity

	signed := fillQty
	if !buy {
		signed = -signed
	}

	now := time.Now()
	realized, pos := b.book.Apply(Fill{
		InstrumentToken: o.Token,
		Symbol:          o.Symbol,
		SignedQuantity:  signed,
		Price:           fillPrice,
		Time:            now,
	})

	turnover := fillPrice * float64(fillQty)
	charges := b.cfg.Costs.Charges(turnover, buy)
	b.recordCosts(charges, turnover, plan.SlippageBps)
	fee := charges.Total

	id := b.nextID()

	if b.sink != nil {
		// Persistence is asynchronous; a slow database must never delay the
		// next fill.
		b.sink.Record(TradeRecord{
			BrokerOrderID:   id,
			UserID:          b.userID,
			InstrumentToken: o.Token,
			Symbol:          o.Symbol,
			StrategyName:    o.Strategy,
			TransactionType: sideWord(buy),
			Quantity:        fillQty,
			Price:           fillPrice,
			Fee:             fee,
			RealizedPnL:     realized,
			Position:        pos,
			TradeType:       TradeTypePaper,
			At:              now,
		})
	}

	return id, nil
}

// Mark updates the last price for an instrument so unrealised PnL tracks the
// live tape. Called from the market data path, so it must stay cheap.
func (b *PaperBroker) Mark(token uint32, price float64, at time.Time) {
	b.book.Mark(token, price, at)
}

// recordCosts accumulates the itemised charges and turnover-weighted slippage.
func (b *PaperBroker) recordCosts(c CostBreakdown, turnover, slipBps float64) {
	b.costMu.Lock()
	b.costs.Add(c)
	b.turnover += turnover
	b.slipBpsWt += slipBps * turnover
	b.costMu.Unlock()
}

func (b *PaperBroker) countPartial() {
	b.costMu.Lock()
	b.partial++
	b.costMu.Unlock()
}

func (b *PaperBroker) countRejected() {
	b.costMu.Lock()
	b.rejected++
	b.costMu.Unlock()
}

// FeesPaid returns the total of all charges.
func (b *PaperBroker) FeesPaid() float64 {
	b.costMu.Lock()
	defer b.costMu.Unlock()
	return b.costs.Total
}

// Costs returns the itemised charge breakdown.
func (b *PaperBroker) Costs() CostBreakdown {
	b.costMu.Lock()
	defer b.costMu.Unlock()
	return b.costs
}

// PaperStats summarises the simulated account.
type PaperStats struct {
	Mode          string          `json:"mode"`
	Fills         uint64          `json:"fills"`
	FeesPaid      float64         `json:"fees_paid"`
	RealizedPnL   float64         `json:"realized_pnl"`
	UnrealizedPnL float64         `json:"unrealized_pnl"`
	NetPnL        float64         `json:"net_pnl"`
	OpenPositions []PaperPosition `json:"open_positions"`

	// Costs itemises where the fees went, so a losing strategy can be told
	// apart from a strategy losing to its own transaction costs.
	Costs CostBreakdown `json:"costs"`
	// Turnover is the gross value transacted.
	Turnover float64 `json:"turnover"`
	// AvgSlippageBps is turnover-weighted, so one tiny fill cannot skew it.
	AvgSlippageBps float64 `json:"avg_slippage_bps"`
	// PartialFills were cut short by the volume participation cap;
	// RejectedFills found no liquidity at all. Both mean the strategy is
	// asking for more size than the market on those bars could give it.
	PartialFills  uint64 `json:"partial_fills"`
	RejectedFills uint64 `json:"rejected_fills"`
}

// Stats returns a snapshot for the monitoring API.
func (b *PaperBroker) Stats() PaperStats {
	realized, unrealized := b.book.Totals()

	b.costMu.Lock()
	costs := b.costs
	turnover := b.turnover
	slipWt := b.slipBpsWt
	partial := b.partial
	rejected := b.rejected
	b.costMu.Unlock()

	avgSlip := 0.0
	if turnover > 0 {
		avgSlip = slipWt / turnover
	}

	return PaperStats{
		Mode:           TradeTypePaper,
		Fills:          b.seq.Load(),
		FeesPaid:       costs.Total,
		RealizedPnL:    realized,
		UnrealizedPnL:  unrealized,
		NetPnL:         realized + unrealized - costs.Total,
		OpenPositions:  b.book.OpenPositions(),
		Costs:          costs,
		Turnover:       turnover,
		AvgSlippageBps: avgSlip,
		PartialFills:   partial,
		RejectedFills:  rejected,
	}
}

func (b *PaperBroker) nextID() string {
	n := b.seq.Add(1)
	return "PAPER-" + string(appendUint(make([]byte, 0, 20), n))
}

// isBuy maps a strategy order onto a market side.
//
// An entry goes long by buying and short by selling; an exit does the
// opposite, because it closes the exposure the entry opened.
func isBuy(o strategy.Order) bool {
	if o.Action == strategy.ActionExit {
		return o.Side == strategy.SideShort // closing a short means buying back
	}
	return o.Side == strategy.SideLong
}

func sideWord(buy bool) string {
	if buy {
		return "BUY"
	}
	return "SELL"
}

func appendUint(dst []byte, v uint64) []byte {
	if v == 0 {
		return append(dst, '0')
	}
	var tmp [20]byte
	i := len(tmp)
	for v > 0 {
		i--
		tmp[i] = byte('0' + v%10)
		v /= 10
	}
	return append(dst, tmp[i:]...)
}
