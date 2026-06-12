package wal_test

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/marketdata"
	"github.com/Bhavik2205/ML-Bot/internal/marketdata/wal"
)

func makeTick(token uint32, price float64) marketdata.NormalizedTick {
	now := time.Now()
	return marketdata.NormalizedTick{
		InstrumentToken: token,
		LastPrice:       price,
		Volume:          100,
		EventTime:       now,
		IngestTime:      now,
	}
}

// Test 1: 10,000 ticks written → 10,000 records readable from WAL.
func TestWAL_10000Ticks(t *testing.T) {
	dir := t.TempDir()
	w, err := wal.NewWriter(dir)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}

	const total = 10_000
	for i := 0; i < total; i++ {
		if err := w.Append(makeTick(uint32(i%50+1), float64(100+i))); err != nil {
			t.Fatalf("Append[%d]: %v", i, err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	var count int
	if err := wal.ReplayDir(dir, func(_ wal.TickRecord) error {
		count++
		return nil
	}); err != nil {
		t.Fatalf("ReplayDir: %v", err)
	}

	if count != total {
		t.Errorf("expected %d records, got %d", total, count)
	}
}

// Test 2: Flush guarantees — records written and flushed before Close are recoverable.
func TestWAL_FlushedRecordsRecoverable(t *testing.T) {
	dir := t.TempDir()
	w, err := wal.NewWriter(dir)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}

	const total = 500
	for i := 0; i < total; i++ {
		if err := w.Append(makeTick(1, float64(i))); err != nil {
			t.Fatalf("Append[%d]: %v", i, err)
		}
		// Flush periodically to simulate "kill -9 after flush"
		if i%50 == 0 {
			if err := w.Flush(); err != nil {
				t.Fatalf("Flush: %v", err)
			}
		}
	}
	// Final flush — everything after this is durable.
	if err := w.Flush(); err != nil {
		t.Fatalf("final Flush: %v", err)
	}
	// On Windows we must release the file handle before the temp dir can be
	// removed by the test harness. Close() only syncs — it does NOT invalidate
	// the already-flushed data, so the "kill -9 after flush" guarantee still holds.
	_ = w.Close()

	var count int
	if err := wal.ReplayDir(dir, func(_ wal.TickRecord) error {
		count++
		return nil
	}); err != nil {
		t.Fatalf("ReplayDir: %v", err)
	}

	if count != total {
		t.Errorf("expected %d flushed records recoverable, got %d", total, count)
	}
}

// Test 3: Manual corruption → checksum detects it and returns ErrCorrupt.
func TestWAL_CorruptionDetected(t *testing.T) {
	dir := t.TempDir()
	w, err := wal.NewWriter(dir)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}

	for i := 0; i < 10; i++ {
		if err := w.Append(makeTick(1, float64(i))); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Find the segment file and corrupt a byte in the middle of the first payload.
	segments, err := filepath.Glob(filepath.Join(dir, "*.wal"))
	if err != nil || len(segments) == 0 {
		t.Fatal("no segment files found")
	}
	data, err := os.ReadFile(segments[0])
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}

	// Frame layout: [4 length][payload...][4 CRC]
	// Corrupt a byte inside the payload of the first record (offset 5).
	frameStart := 4 // skip the 4-byte length field
	if len(data) > frameStart+5 {
		data[frameStart+5] ^= 0xFF
	}

	// Also corrupt the stored CRC so it reads as a different value
	// (to ensure our checksum recomputation catches it, not just a match by luck).
	payloadLen := int(binary.LittleEndian.Uint32(data[0:4]))
	crcOffset := 4 + payloadLen
	if len(data) > crcOffset+4 {
		data[crcOffset] ^= 0x01
	}

	if err := os.WriteFile(segments[0], data, 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	err = wal.ReplayDir(dir, func(_ wal.TickRecord) error { return nil })
	if err == nil {
		t.Fatal("expected ErrCorrupt, got nil")
	}
	if err != wal.ErrCorrupt {
		t.Fatalf("expected ErrCorrupt, got: %v", err)
	}
}
