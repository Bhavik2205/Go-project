package wal

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"sync"

	"github.com/Bhavik2205/ML-Bot/internal/marketdata"
	"github.com/Bhavik2205/ML-Bot/internal/observability"
)

// Writer is the public interface callers use.
type Writer interface {
	Append(tick marketdata.NormalizedTick) error
	Flush() error
	Close() error
}

// walWriter is the concrete implementation.
type walWriter struct {
	mu      sync.Mutex
	dir     string
	current *segment
	closed  bool
}

// NewWriter creates a WAL writer that stores segments in dir.
// dir is created if it does not exist.
func NewWriter(dir string) (Writer, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("wal: create dir %s: %w", dir, err)
	}
	seg, err := newSegment(dir)
	if err != nil {
		return nil, err
	}
	observability.WALSegmentsTotal.Inc()
	return &walWriter{dir: dir, current: seg}, nil
}

// Append writes one tick to the WAL.
// The tick is persisted before the caller publishes it to TickBus.
func (w *walWriter) Append(tick marketdata.NormalizedTick) error {
	rec := TickRecord{
		Timestamp:         tick.IngestTime.UnixNano(),
		InstrumentToken:   tick.InstrumentToken,
		LastPrice:         tick.LastPrice,
		Volume:            uint32(tick.Volume),
		ExchangeTimestamp: tick.EventTime.UnixNano(),
	}
	payload := marshalPayload(&rec)
	rec.CRC32 = checksum(payload)

	// Wire frame: [4-byte length][payload][4-byte CRC32]
	// length == recordPayloadSize (constant, but stored for forward compatibility)
	frame := make([]byte, 4+recordPayloadSize+4)
	binary.LittleEndian.PutUint32(frame[0:], uint32(recordPayloadSize))
	copy(frame[4:], payload)
	binary.LittleEndian.PutUint32(frame[4+recordPayloadSize:], rec.CRC32)

	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return fmt.Errorf("wal: writer is closed")
	}

	if err := w.rotateIfNeeded(); err != nil {
		observability.WALAppendErrorsTotal.Inc()
		return err
	}

	n, err := w.current.write(frame)
	if err != nil {
		observability.WALAppendErrorsTotal.Inc()
		return fmt.Errorf("wal: write frame: %w", err)
	}

	observability.WALAppendsTotal.Inc()
	observability.WALBytesWrittenTotal.Add(float64(n))
	return nil
}

// Flush syncs the current segment to disk.
func (w *walWriter) Flush() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed || w.current == nil {
		return nil
	}
	return w.current.flush()
}

// Close flushes and closes the current segment.
func (w *walWriter) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil
	}
	w.closed = true
	if w.current == nil {
		return nil
	}
	if err := w.current.flush(); err != nil {
		return err
	}
	return w.current.close()
}

// rotateIfNeeded opens a new segment when the current one is full or old.
// Caller must hold w.mu.
func (w *walWriter) rotateIfNeeded() error {
	if !w.current.shouldRotate() {
		return nil
	}
	if err := w.current.flush(); err != nil {
		return fmt.Errorf("wal: flush before rotate: %w", err)
	}
	if err := w.current.close(); err != nil {
		return fmt.Errorf("wal: close before rotate: %w", err)
	}
	seg, err := newSegment(w.dir)
	if err != nil {
		return err
	}
	w.current = seg
	observability.WALSegmentsTotal.Inc()
	return nil
}

// NoopWriter is a Writer that discards all ticks — useful when WAL is disabled.
type NoopWriter struct{}

func (NoopWriter) Append(_ marketdata.NormalizedTick) error { return nil }
func (NoopWriter) Flush() error                             { return nil }
func (NoopWriter) Close() error                             { return nil }

// WithContext returns a context-aware Append wrapper so callers can cancel.
func WithContext(ctx context.Context, w Writer, tick marketdata.NormalizedTick) error {
	done := make(chan error, 1)
	go func() { done <- w.Append(tick) }()
	select {
	case err := <-done:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}
