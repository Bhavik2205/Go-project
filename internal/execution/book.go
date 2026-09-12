package monitor

import (
	"sync"
	"time"
)

// PaperPosition is the netted position for one instrument.
//
// Quantity is signed: positive is long, negative is short, zero is flat.
// AveragePrice is the volume-weighted entry of the currently open quantity;
// it is only recomputed when exposure is added, never when it is reduced,
// which is what makes realised PnL come out right on a partial close.
type PaperPosition struct {
	InstrumentToken uint32
	Symbol          string
	Quantity        int64
	AveragePrice    float64
	LastPrice       float64
	RealizedPnL     float64
	UpdatedAt       time.Time
}

// UnrealizedPnL marks the open quantity to LastPrice.
func (p *PaperPosition) UnrealizedPnL() float64 {
	if p.Quantity == 0 || p.LastPrice == 0 {
		return 0
	}
	return (p.LastPrice - p.AveragePrice) * float64(p.Quantity)
}

// IsOpen reports whether any exposure remains.
func (p *PaperPosition) IsOpen() bool { return p.Quantity != 0 }

// Fill is one simulated execution applied to the book.
type Fill struct {
	InstrumentToken uint32
	Symbol          string
	// SignedQuantity is positive for a buy, negative for a sell.
	SignedQuantity int64
	Price          float64
	Time           time.Time
}

// Book is a thread-safe collection of netted paper positions.
type Book struct {
	mu        sync.RWMutex
	positions map[uint32]*PaperPosition
	// realizedTotal is the account-level realised PnL across all instruments.
	realizedTotal float64
}

// NewBook builds an empty position book.
func NewBook() *Book {
	return &Book{positions: make(map[uint32]*PaperPosition)}
}

// Apply nets a fill into the book and returns the realised PnL produced by
// this fill (non-zero only when the fill reduces or flips exposure) together
// with the resulting position.
//
// The four cases are: opening from flat, adding to an existing side,
// reducing the same side, and flipping through zero to the other side.
func (b *Book) Apply(f Fill) (realized float64, pos PaperPosition) {
	b.mu.Lock()
	defer b.mu.Unlock()

	p, ok := b.positions[f.InstrumentToken]
	if !ok {
		p = &PaperPosition{InstrumentToken: f.InstrumentToken, Symbol: f.Symbol}
		b.positions[f.InstrumentToken] = p
	}
	if f.Symbol != "" {
		p.Symbol = f.Symbol
	}

	old := p.Quantity
	qty := f.SignedQuantity

	switch {
	case old == 0:
		// Opening a new position.
		p.Quantity = qty
		p.AveragePrice = f.Price

	case sameSign(old, qty):
		// Adding to the existing side: volume-weight the entry.
		total := float64(abs64(old))*p.AveragePrice + float64(abs64(qty))*f.Price
		p.Quantity = old + qty
		p.AveragePrice = total / float64(abs64(p.Quantity))

	case abs64(qty) <= abs64(old):
		// Reducing (or exactly closing) the existing side. The closed
		// quantity realises against the unchanged average entry.
		closed := abs64(qty)
		realized = pnlFor(old, p.AveragePrice, f.Price, closed)
		p.Quantity = old + qty
		if p.Quantity == 0 {
			p.AveragePrice = 0
		}

	default:
		// Flipping through zero: realise the whole old position, then open
		// the remainder on the other side at the fill price.
		closed := abs64(old)
		realized = pnlFor(old, p.AveragePrice, f.Price, closed)
		p.Quantity = old + qty
		p.AveragePrice = f.Price
	}

	p.RealizedPnL += realized
	b.realizedTotal += realized
	p.LastPrice = f.Price
	p.UpdatedAt = f.Time

	return realized, *p
}

// Mark updates the last traded price so unrealised PnL stays current.
func (b *Book) Mark(token uint32, price float64, at time.Time) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if p, ok := b.positions[token]; ok {
		p.LastPrice = price
		p.UpdatedAt = at
	}
}

// Position returns a copy of one position.
func (b *Book) Position(token uint32) (PaperPosition, bool) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	p, ok := b.positions[token]
	if !ok {
		return PaperPosition{}, false
	}
	return *p, true
}

// Positions returns a copy of every position, open or flat.
func (b *Book) Positions() []PaperPosition {
	b.mu.RLock()
	defer b.mu.RUnlock()
	out := make([]PaperPosition, 0, len(b.positions))
	for _, p := range b.positions {
		out = append(out, *p)
	}
	return out
}

// OpenPositions returns only positions with live exposure.
func (b *Book) OpenPositions() []PaperPosition {
	b.mu.RLock()
	defer b.mu.RUnlock()
	out := make([]PaperPosition, 0, len(b.positions))
	for _, p := range b.positions {
		if p.IsOpen() {
			out = append(out, *p)
		}
	}
	return out
}

// Totals returns account-level realised and unrealised PnL.
func (b *Book) Totals() (realized, unrealized float64) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	for _, p := range b.positions {
		unrealized += p.UnrealizedPnL()
	}
	return b.realizedTotal, unrealized
}

// pnlFor computes realised PnL for closing `closed` units of a position held
// in the direction of oldQty, entered at avg and closed at exit.
func pnlFor(oldQty int64, avg, exit float64, closed int64) float64 {
	if oldQty > 0 { // closing a long: sell high is profit
		return (exit - avg) * float64(closed)
	}
	return (avg - exit) * float64(closed) // closing a short
}

func sameSign(a, b int64) bool { return (a > 0 && b > 0) || (a < 0 && b < 0) }

func abs64(v int64) int64 {
	if v < 0 {
		return -v
	}
	return v
}
