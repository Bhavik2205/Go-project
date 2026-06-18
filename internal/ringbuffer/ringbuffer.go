// Package ringbuffer provides a single-producer / single-consumer (SPSC) lock-free
// ring buffer designed for the tick worker pool hot path.
//
// Design constraints:
//   - Capacity must be a power of 2; NewSPSCRingBuffer enforces this.
//   - Head (consumer) and Tail (producer) are each padded to their own cache line
//     (64 bytes) to eliminate false-sharing between producer and consumer goroutines.
//   - Publish is non-blocking: returns false immediately when the buffer is full.
//   - TryConsume is non-blocking: returns (tick, true) when an item is available,
//     (zero, false) when empty. Intended for the worker hot loop.
//   - ConsumeBlocking parks via Gosched / exponential back-off until an item
//     arrives or the context is cancelled. Useful for low-rate consumers and tests.
//
// IMPORTANT: This ring buffer is designed for exactly ONE producer goroutine and
// ONE consumer goroutine. It is NOT safe for multiple producers. In this codebase,
// the only producer is the TickBus subscriber goroutine inside MarketDataIngestor.
// Do not call Publish from multiple goroutines concurrently.
package ringbuffer

import (
	"context"
	"fmt"
	"runtime"
	"sync/atomic"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/marketdata"
)

const cacheLinePad = 64

// paddedUint64 occupies exactly one cache line so that head and tail never
// share a cache line, eliminating false-sharing between producer and consumer.
type paddedUint64 struct {
	v uint64
	_ [cacheLinePad - 8]byte
}

// SPSCRingBuffer is a SPSC lock-free ring buffer of marketdata.NormalizedTick.
// Safe for exactly one producer goroutine and one consumer goroutine.
type SPSCRingBuffer struct {
	// Producer-owned: only Publish writes tail.
	tail paddedUint64
	// Consumer-owned: only TryConsume / ConsumeBlocking writes head.
	head paddedUint64

	mask uint64 // capacity - 1; replaces modulo on every access
	buf  []marketdata.NormalizedTick
}

// NewSPSCRingBuffer allocates a SPSCRingBuffer with the given capacity.
// capacity must be a positive power of 2.
func NewSPSCRingBuffer(capacity int) (*SPSCRingBuffer, error) {
	if capacity <= 0 || (capacity&(capacity-1)) != 0 {
		return nil, fmt.Errorf("ringbuffer: capacity %d is not a positive power of 2", capacity)
	}
	return &SPSCRingBuffer{
		mask: uint64(capacity - 1),
		buf:  make([]marketdata.NormalizedTick, capacity),
	}, nil
}

// Cap returns the buffer capacity.
func (rb *SPSCRingBuffer) Cap() int { return len(rb.buf) }

// Len returns an instantaneous snapshot of the number of items in the buffer.
// Accurate enough for monitoring; may be momentarily stale under concurrent access.
func (rb *SPSCRingBuffer) Len() int {
	tail := atomic.LoadUint64(&rb.tail.v)
	head := atomic.LoadUint64(&rb.head.v)
	// Unsigned subtraction wraps correctly on overflow.
	return int(tail - head)
}

// Publish writes tick into the ring buffer without blocking.
// Returns true on success, false if the buffer is full.
//
// This method is NOT safe for concurrent calls from multiple producers.
// Only one goroutine (the TickBus subscriber) should call it.
func (rb *SPSCRingBuffer) Publish(tick marketdata.NormalizedTick) bool {
	tail := atomic.LoadUint64(&rb.tail.v)
	head := atomic.LoadUint64(&rb.head.v)
	if tail-head >= uint64(len(rb.buf)) {
		return false
	}
	rb.buf[tail&rb.mask] = tick
	// Release store: slot write must be globally visible before tail advances.
	atomic.StoreUint64(&rb.tail.v, tail+1)
	return true
}

// TryConsume attempts to read one tick from the buffer without blocking.
// Returns (tick, true) if an item was available, (zero, false) if empty.
//
// This is the intended method for the worker hot loop:
//
//	for {
//	    if tick, ok := rb.TryConsume(); ok {
//	        process(tick)
//	        continue
//	    }
//	    runtime.Gosched()
//	}
func (rb *SPSCRingBuffer) TryConsume() (marketdata.NormalizedTick, bool) {
	head := atomic.LoadUint64(&rb.head.v)
	tail := atomic.LoadUint64(&rb.tail.v)
	if head == tail {
		var zero marketdata.NormalizedTick
		return zero, false
	}
	tick := rb.buf[head&rb.mask]
	// Advance head after reading so Publish cannot overwrite the slot we just read.
	atomic.StoreUint64(&rb.head.v, head+1)
	return tick, true
}

// ConsumeBlocking blocks until a tick is available or ctx is cancelled.
// It uses Gosched for the first spinYields iterations, then falls back to
// exponential sleep (1µs → 1ms) so idle workers don't burn a full core.
//
// Use this for low-rate consumers or in tests. For the high-throughput tick
// worker hot loop use TryConsume directly.
func (rb *SPSCRingBuffer) ConsumeBlocking(ctx context.Context) (marketdata.NormalizedTick, bool) {
	const (
		spinYields  = 16
		sleepMin    = time.Microsecond
		sleepMax    = time.Millisecond
		sleepFactor = 2
	)

	sleep := sleepMin
	spins := 0

	for {
		if tick, ok := rb.TryConsume(); ok {
			return tick, true
		}

		select {
		case <-ctx.Done():
			var zero marketdata.NormalizedTick
			return zero, false
		default:
		}

		if spins < spinYields {
			runtime.Gosched()
			spins++
		} else {
			time.Sleep(sleep)
			sleep *= sleepFactor
			if sleep > sleepMax {
				sleep = sleepMax
			}
		}
	}
}
