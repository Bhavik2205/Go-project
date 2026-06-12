package wal

import (
	"fmt"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"
)

const (
	maxSegmentBytes = 100 * 1024 * 1024 // 100 MB
	maxSegmentAge   = 1 * time.Hour
)

// segment represents a single WAL segment file.
type segment struct {
	file      *os.File
	path      string
	bytesWritten int64
	createdAt time.Time
}

// segmentCounter is a process-global monotonic segment index.
var segmentCounter uint64

// newSegment creates (or opens for append) a new segment file in dir.
func newSegment(dir string) (*segment, error) {
	idx := atomic.AddUint64(&segmentCounter, 1)
	name := fmt.Sprintf("ticks-%s-%04d.wal", time.Now().UTC().Format("20060102"), idx)
	path := filepath.Join(dir, name)

	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, fmt.Errorf("wal: open segment %s: %w", path, err)
	}

	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("wal: stat segment %s: %w", path, err)
	}

	return &segment{
		file:         f,
		path:         path,
		bytesWritten: info.Size(),
		createdAt:    time.Now(),
	}, nil
}

// shouldRotate returns true when this segment has hit the size or age limit.
func (s *segment) shouldRotate() bool {
	return s.bytesWritten >= maxSegmentBytes || time.Since(s.createdAt) >= maxSegmentAge
}

// write appends raw bytes to the segment.
func (s *segment) write(buf []byte) (int, error) {
	n, err := s.file.Write(buf)
	s.bytesWritten += int64(n)
	return n, err
}

// flush syncs the segment to disk.
func (s *segment) flush() error {
	return s.file.Sync()
}

// close closes the underlying file.
func (s *segment) close() error {
	return s.file.Close()
}
