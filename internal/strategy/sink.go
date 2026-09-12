package strategy

import "sync/atomic"

// Sink receives orders produced by the engine.
//
// Emit must never block. The engine calls it from the market data path, so a
// slow or stalled consumer must cost a dropped order, not a stalled pipeline.
// Emit reports whether the order was accepted.
type Sink interface {
	Emit(o Order) bool
}

// ChannelSink is the default Sink: a bounded channel with non-blocking sends.
type ChannelSink struct {
	ch      chan Order
	dropped atomic.Uint64
}

// NewChannelSink builds a sink with the given buffer depth.
func NewChannelSink(buffer int) *ChannelSink {
	if buffer <= 0 {
		buffer = 4096
	}
	return &ChannelSink{ch: make(chan Order, buffer)}
}

// Emit performs a non-blocking send, dropping on a full buffer.
func (s *ChannelSink) Emit(o Order) bool {
	select {
	case s.ch <- o:
		return true
	default:
		s.dropped.Add(1)
		return false
	}
}

// Orders is the receive end, consumed by the execution layer.
func (s *ChannelSink) Orders() <-chan Order { return s.ch }

// Dropped is the number of orders lost to a full buffer.
func (s *ChannelSink) Dropped() uint64 { return s.dropped.Load() }

// Len and Cap expose queue depth for the observability endpoints.
func (s *ChannelSink) Len() int { return len(s.ch) }
func (s *ChannelSink) Cap() int { return cap(s.ch) }

// Close shuts the channel. Call only after the engine has stopped.
func (s *ChannelSink) Close() { close(s.ch) }

// FuncSink adapts a plain function to the Sink interface, which keeps tests
// and one-off consumers from needing a goroutine.
type FuncSink func(o Order) bool

// Emit calls the underlying function.
func (f FuncSink) Emit(o Order) bool { return f(o) }

// DiscardSink drops everything. Useful for dry runs and benchmarks.
var DiscardSink Sink = FuncSink(func(Order) bool { return true })
