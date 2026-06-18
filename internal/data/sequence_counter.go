package data

import (
	"sync"
	"sync/atomic"
	"time"
)

type sequenceEntry struct {
	lastSecond atomic.Int64
	counter    atomic.Uint64
}

type SequenceCounter struct {
	entries sync.Map
}

func NewSequenceCounter() *SequenceCounter {
	return &SequenceCounter{}
}

func (s *SequenceCounter) Next(
	token uint32,
	ts time.Time,
) uint64 {

	currentSecond := ts.Unix()

	value, _ := s.entries.LoadOrStore(
		token,
		&sequenceEntry{},
	)

	entry := value.(*sequenceEntry)

	prev := entry.lastSecond.Load()

	if prev != currentSecond {

		if entry.lastSecond.CompareAndSwap(
			prev,
			currentSecond,
		) {
			entry.counter.Store(0)
		}
	}

	return entry.counter.Add(1)
}
