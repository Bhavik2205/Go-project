package monitor

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync/atomic"

	"github.com/Bhavik2205/ML-Bot/internal/strategy"
	kiteconnect "github.com/zerodha/gokiteconnect/v4"
	"go.uber.org/zap"
)

// Env vars that gate live trading. Both must be set correctly; neither has a
// permissive default.
const (
	EnvTradingMode = "TRADING_MODE"         // "paper" (default) or "live"
	EnvLiveConfirm = "LIVE_TRADING_CONFIRM" // must equal LiveConfirmPhrase
)

// LiveConfirmPhrase is the exact value LIVE_TRADING_CONFIRM must hold before
// real orders are possible. A phrase rather than "true" so that live trading
// cannot be switched on by a stray boolean in a shared .env.
const LiveConfirmPhrase = "I_UNDERSTAND_THIS_PLACES_REAL_ORDERS"

// TradingMode reports the configured mode, defaulting to paper.
func TradingMode() string {
	if strings.EqualFold(os.Getenv(EnvTradingMode), "live") {
		return TradeTypeLive
	}
	return TradeTypePaper
}

// LiveEnabled reports whether live trading is both selected and confirmed.
//
// Two independent switches are required. Selecting live mode alone is not
// enough, so a copied config or a wrong environment cannot start trading real
// money by itself.
func LiveEnabled() (bool, string) {
	if TradingMode() != TradeTypeLive {
		return false, fmt.Sprintf("%s is not \"live\"", EnvTradingMode)
	}
	if os.Getenv(EnvLiveConfirm) != LiveConfirmPhrase {
		return false, fmt.Sprintf("%s is not set to the confirmation phrase", EnvLiveConfirm)
	}
	return true, ""
}

// KiteOrderPlacer is the slice of the Kite client this package needs, which
// keeps the live broker testable without a real session.
type KiteOrderPlacer interface {
	PlaceOrder(variety string, params kiteconnect.OrderParams) (kiteconnect.OrderResponse, error)
}

// SymbolResolver maps an instrument token to its exchange and trading symbol.
type SymbolResolver func(token uint32) (exchange, tradingSymbol string, ok bool)

// LiveBroker places real orders with Zerodha.
//
// It should always be constructed behind a Guard. Nothing in this type
// enforces daily caps; that is the Guard's job.
type LiveBroker struct {
	kite    KiteOrderPlacer
	resolve SymbolResolver
	product string
	userID  uint

	placed atomic.Uint64
	failed atomic.Uint64
}

// NewLiveBroker builds a live broker. It refuses to construct unless live
// trading is explicitly enabled, so a wiring mistake fails at startup rather
// than at the first signal.
func NewLiveBroker(kite KiteOrderPlacer, resolve SymbolResolver, product string, userID uint) (*LiveBroker, error) {
	if ok, why := LiveEnabled(); !ok {
		return nil, fmt.Errorf("live trading is not enabled: %s", why)
	}
	if kite == nil {
		return nil, fmt.Errorf("live broker requires a Kite client")
	}
	if resolve == nil {
		return nil, fmt.Errorf("live broker requires a symbol resolver")
	}
	if product == "" {
		product = kiteconnect.ProductMIS
	}
	zap.L().Warn("⚠️  LIVE TRADING ENABLED — real orders will be placed",
		zap.String("product", product), zap.Uint("user_id", userID))
	return &LiveBroker{kite: kite, resolve: resolve, product: product, userID: userID}, nil
}

// Mode reports the trade type this broker records.
func (b *LiveBroker) Mode() string { return TradeTypeLive }

// Place submits a market order to the exchange.
func (b *LiveBroker) Place(ctx context.Context, o strategy.Order) (string, error) {
	exchange, symbol, ok := b.resolve(o.Token)
	if !ok {
		b.failed.Add(1)
		return "", fmt.Errorf("live: no symbol mapping for instrument token %d", o.Token)
	}
	if o.Quantity <= 0 {
		b.failed.Add(1)
		return "", fmt.Errorf("live: refusing order with quantity %d", o.Quantity)
	}

	txn := kiteconnect.TransactionTypeBuy
	if !isBuy(o) {
		txn = kiteconnect.TransactionTypeSell
	}

	params := kiteconnect.OrderParams{
		Exchange:        exchange,
		Tradingsymbol:   symbol,
		TransactionType: txn,
		OrderType:       kiteconnect.OrderTypeMarket,
		Product:         b.product,
		Validity:        kiteconnect.ValidityDay,
		Quantity:        int(o.Quantity),
		Tag:             tagFor(o.Strategy),
	}

	// The Kite SDK call is synchronous and does not take a context, so honour
	// cancellation before spending the round trip.
	if err := ctx.Err(); err != nil {
		return "", err
	}

	resp, err := b.kite.PlaceOrder(kiteconnect.VarietyRegular, params)
	if err != nil {
		b.failed.Add(1)
		return "", fmt.Errorf("live: place order %s %s x%d: %w", txn, symbol, o.Quantity, err)
	}

	b.placed.Add(1)
	zap.L().Info("live order placed",
		zap.String("order_id", resp.OrderID),
		zap.String("symbol", symbol),
		zap.String("transaction", txn),
		zap.Int64("quantity", o.Quantity),
		zap.String("strategy", o.Strategy),
	)
	return resp.OrderID, nil
}

// Placed and Failed expose counters for the monitoring API.
func (b *LiveBroker) Placed() uint64 { return b.placed.Load() }
func (b *LiveBroker) Failed() uint64 { return b.failed.Load() }

// tagFor builds a Kite order tag. Kite caps tags at 20 characters.
func tagFor(strategyName string) string {
	const max = 20
	t := strings.ReplaceAll(strategyName, "_", "")
	if len(t) > max {
		t = t[:max]
	}
	return t
}

// NewBrokerForMode returns the broker matching the configured trading mode,
// already wrapped in a Guard.
//
// Paper is always the fallback: if live is selected but not confirmed, or the
// live broker cannot be built, the system runs on paper rather than silently
// doing nothing or, worse, trading unguarded.
func NewBrokerForMode(
	kite KiteOrderPlacer,
	resolve SymbolResolver,
	paper *PaperBroker,
	guardCfg GuardConfig,
	userID uint,
) (*Guard, string) {
	if ok, why := LiveEnabled(); !ok {
		zap.L().Info("Trading mode: PAPER", zap.String("reason", why))
		return NewGuard(paper, guardCfg), TradeTypePaper
	}

	live, err := NewLiveBroker(kite, resolve, kiteconnect.ProductMIS, userID)
	if err != nil {
		zap.L().Error("Live trading requested but unavailable, falling back to PAPER", zap.Error(err))
		return NewGuard(paper, guardCfg), TradeTypePaper
	}
	return NewGuard(live, guardCfg), TradeTypeLive
}
