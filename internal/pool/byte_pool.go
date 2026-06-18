package pool

import "sync"

const (
	// Default size for newly created buffers.
	DefaultBufferSize = 4 * 1024 // 4KB

	// Do not return huge buffers to the pool.
	// Prevents memory bloat after temporary spikes.
	MaxPooledBufferSize = 64 * 1024 // 64KB
)

var BytePool = sync.Pool{
	New: func() any {
		return make([]byte, 0, DefaultBufferSize)
	},
}

// GetBuffer returns a byte slice with at least the requested capacity.
//
// Example:
//
//	buf := pool.GetBuffer(1024)
//	defer pool.PutBuffer(buf)
//
//	buf = append(buf, data...)
func GetBuffer(size int) []byte {
	b := BytePool.Get().([]byte)

	if cap(b) < size {
		return make([]byte, 0, size)
	}

	return b[:0]
}

// GetBufferWithLen returns a slice with the requested length.
//
// Example:
//
//	buf := pool.GetBufferWithLen(1024)
//	copy(buf, src)
func GetBufferWithLen(size int) []byte {
	b := BytePool.Get().([]byte)

	if cap(b) < size {
		return make([]byte, size)
	}

	return b[:size]
}

// PutBuffer returns a buffer to the pool.
//
// Large buffers are discarded to avoid memory retention.
func PutBuffer(b []byte) {
	if b == nil {
		return
	}

	if cap(b) > MaxPooledBufferSize {
		return
	}

	BytePool.Put(b[:0])
}
