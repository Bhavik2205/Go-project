package ringbuffer_test

import (
	"context"
	"runtime"
	"testing"

	"github.com/Bhavik2205/ML-Bot/internal/marketdata"
	"github.com/Bhavik2205/ML-Bot/internal/ringbuffer"
)

var sink marketdata.NormalizedTick // prevents dead-code elimination

func makeTick(token uint32) marketdata.NormalizedTick {
	return marketdata.NormalizedTick{InstrumentToken: token, LastPrice: 1234.5}
}

// BenchmarkPublish measures how fast a single goroutine can publish to an
// otherwise-empty ring buffer (best-case producer throughput).
func BenchmarkPublish(b *testing.B) {
	rb, _ := ringbuffer.NewSPSCRingBuffer(65536)
	tick := makeTick(1)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if !rb.Publish(tick) {
			// Drain one slot so we never permanently stall.
			rb.TryConsume()
			rb.Publish(tick)
		}
	}
}

// BenchmarkTryConsume measures how fast a single goroutine can consume from a
// pre-filled ring buffer (best-case consumer throughput).
func BenchmarkTryConsume(b *testing.B) {
	rb, _ := ringbuffer.NewSPSCRingBuffer(65536)
	tick := makeTick(1)
	// Pre-fill.
	for i := 0; i < 65536; i++ {
		rb.Publish(tick)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		t, ok := rb.TryConsume()
		if ok {
			sink = t
		} else {
			// Refill one slot to keep the buffer non-empty.
			rb.Publish(tick)
		}
	}
}

// BenchmarkPublishTryConsume measures the full roundtrip: one publish
// immediately followed by one consume in the same goroutine.
func BenchmarkPublishTryConsume(b *testing.B) {
	rb, _ := ringbuffer.NewSPSCRingBuffer(65536)
	tick := makeTick(1)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		rb.Publish(tick)
		t, _ := rb.TryConsume()
		sink = t
	}
}

// BenchmarkThroughputSPSC is the headline benchmark: one producer goroutine
// and one consumer goroutine running concurrently, measuring end-to-end
// throughput.  b.N is the total number of ticks exchanged.
func BenchmarkThroughputSPSC(b *testing.B) {
	rb, _ := ringbuffer.NewSPSCRingBuffer(65536)
	tick := makeTick(42)

	done := make(chan struct{})
	b.ResetTimer()
	b.ReportAllocs()

	// Consumer goroutine.
	go func() {
		for i := 0; i < b.N; i++ {
			for {
				if t, ok := rb.TryConsume(); ok {
					sink = t
					break
				}
				runtime.Gosched()
			}
		}
		close(done)
	}()

	// Producer (this goroutine).
	for i := 0; i < b.N; i++ {
		for !rb.Publish(tick) {
			runtime.Gosched()
		}
	}

	<-done
}

// BenchmarkConsumeBlocking measures ConsumeBlocking on a pre-filled buffer
// (the blocking path is never hit, so this benchmarks the fast path only).
func BenchmarkConsumeBlocking(b *testing.B) {
	rb, _ := ringbuffer.NewSPSCRingBuffer(65536)
	tick := makeTick(1)
	ctx := context.Background()

	for i := 0; i < 65536; i++ {
		rb.Publish(tick)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		t, ok := rb.ConsumeBlocking(ctx)
		if ok {
			sink = t
		} else {
			rb.Publish(tick)
		}
	}
}

// BenchmarkLen measures the cost of reading the approximate depth.
func BenchmarkLen(b *testing.B) {
	rb, _ := ringbuffer.NewSPSCRingBuffer(65536)
	tick := makeTick(1)
	for i := 0; i < 100; i++ {
		rb.Publish(tick)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = rb.Len()
	}
}
