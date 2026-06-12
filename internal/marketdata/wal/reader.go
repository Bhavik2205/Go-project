package wal

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"

	"github.com/Bhavik2205/ML-Bot/internal/observability"
	"go.uber.org/zap"
)

// ErrCorrupt is returned when a record's checksum does not match.
var ErrCorrupt = errors.New("wal: corrupt record — checksum mismatch")

// ReplayFunc is called for each valid record during replay.
type ReplayFunc func(rec TickRecord) error

// ReplayDir replays every segment file in dir in chronological (filename) order.
// On checksum mismatch: logs, increments the alert metric, and returns ErrCorrupt.
func ReplayDir(dir string, fn ReplayFunc) error {
	entries, err := filepath.Glob(filepath.Join(dir, "*.wal"))
	if err != nil {
		return fmt.Errorf("wal: glob %s: %w", dir, err)
	}
	sort.Strings(entries) // filename sort = chronological order

	for _, path := range entries {
		if err := replayFile(path, fn); err != nil {
			return err
		}
	}
	return nil
}

// replayFile reads and validates every record in a single segment file.
func replayFile(path string, fn ReplayFunc) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("wal: open %s: %w", path, err)
	}
	defer f.Close()

	lenBuf := make([]byte, 4)
	for {
		// Read frame length
		if _, err := io.ReadFull(f, lenBuf); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return nil // clean end-of-file
			}
			return fmt.Errorf("wal: read length in %s: %w", path, err)
		}
		payloadLen := binary.LittleEndian.Uint32(lenBuf)

		// Read payload + CRC
		frame := make([]byte, payloadLen+4)
		if _, err := io.ReadFull(f, frame); err != nil {
			if errors.Is(err, io.ErrUnexpectedEOF) {
				// Partial write at tail — safe to stop
				zap.L().Warn("wal: partial record at end of segment, stopping replay",
					zap.String("file", path))
				return nil
			}
			return fmt.Errorf("wal: read frame in %s: %w", path, err)
		}

		payload := frame[:payloadLen]
		storedCRC := binary.LittleEndian.Uint32(frame[payloadLen:])
		computed := checksum(payload)

		if computed != storedCRC {
			zap.L().Error("wal: checksum mismatch — corrupt record",
				zap.String("file", path),
				zap.Uint32("stored_crc", storedCRC),
				zap.Uint32("computed_crc", computed),
			)
			// Pad payload to RecordSize for unmarshal (add zeroed CRC slot)
			full := make([]byte, RecordSize)
			copy(full, payload)
			return ErrCorrupt
		}

		// Unmarshal: copy payload into full-size buffer and set CRC
		full := make([]byte, RecordSize)
		copy(full, payload)
		binary.LittleEndian.PutUint32(full[recordPayloadSize:], storedCRC)
		rec := unmarshalRecord(full)

		observability.WALReplayRecordsTotal.Inc()

		if err := fn(rec); err != nil {
			return fmt.Errorf("wal: replay callback: %w", err)
		}
	}
}
