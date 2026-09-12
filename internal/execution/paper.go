package monitor

import (
	"context"
	"fmt"
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
type PaperConfig struct {
	// SlippageBps is applied against the order in the adverse direction:
	// buys fill above the reference price, sells below it. Modelling
	// slippage as a favour is the classic way to build a backtest that
	// cannot be reproduced live.
	SlippageBps float64
	// FeeBps is charged on notional per fill (brokerage plus statutory
	// charges rolled into one figure).
	FeeBps float64
	// RejectOutsideBand rejects a fill whose reference price is zero or
	// negative, which indicates a bad or missing mark rather than a trade.
	RejectOutsideBand bool
}

// DefaultPaperConfig returns conservative defaults for Indian equity intraday.
func DefaultPaperConfig() PaperConfig {
	return PaperConfig{
		SlippageBps:       2, // 0.02%
		FeeBps:            3, // 0.03% all-in, deliberately pessimistic
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

	seq  atomic.Uint64
	fees atomic.Uint64 // fees in paise, to keep the counter integral
}

// NewPaperBroker builds a paper broker. Pass a nil sink for memory-only runs.
func NewPaperBroker(cfg PaperConfig, userID uint, sink TradeSink) *PaperBroker {
	if cfg.SlippageBps < 0 {
		cfg.SlippageBps = 0
	}
	if cfg.FeeBps < 0 {
		cfg.FeeBps = 0
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
	fillPrice := applySlippage(ref, b.cfg.SlippageBps, buy)

	signed := o.Quantity
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

	notional := fillPrice * float64(o.Quantity)
	fee := notional * b.cfg.FeeBps / 10_000
	b.fees.Add(uint64(fee * 100))

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
			Quantity:        o.Quantity,
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

// FeesPaid returns cumulative simulated fees.
func (b *PaperBroker) FeesPaid() float64 { return float64(b.fees.Load()) / 100 }

// Stats summarises the simulated account.
type PaperStats struct {
	Mode          string          `json:"mode"`
	Fills         uint64          `json:"fills"`
	FeesPaid      float64         `json:"fees_paid"`
	RealizedPnL   float64         `json:"realized_pnl"`
	UnrealizedPnL float64         `json:"unrealized_pnl"`
	NetPnL        float64         `json:"net_pnl"`
	OpenPositions []PaperPosition `json:"open_positions"`
}

// Stats returns a snapshot for the monitoring API.
func (b *PaperBroker) Stats() PaperStats {
	realized, unrealized := b.book.Totals()
	fees := b.FeesPaid()
	return PaperStats{
		Mode:          TradeTypePaper,
		Fills:         b.seq.Load(),
		FeesPaid:      fees,
		RealizedPnL:   realized,
		UnrealizedPnL: unrealized,
		NetPnL:        realized + unrealized - fees,
		OpenPositions: b.book.OpenPositions(),
	}
}

func (b *PaperBroker) nextID() string {
	n := b.seq.Add(1)
	return "PAPER-" + string(appendUint(make([]byte, 0, 20), n))
}

// applySlippage moves the price against the taker.
func applySlippage(ref, bps float64, buy bool) float64 {
	adj := ref * bps / 10_000
	if buy {
		return ref + adj
	}
	return ref - adj
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
