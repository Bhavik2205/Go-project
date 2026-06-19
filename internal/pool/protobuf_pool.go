// internal/pool/protobuf_pool.go
package pool

import (
	"sync"

	pb "github.com/Bhavik2205/ML-Bot/proto"
)

var tickMsgPool = sync.Pool{
	New: func() interface{} {
		return &pb.TickMessage{}
	},
}

// GetTick returns a TickMessage from the pool (reset to zero).
func GetTick() *pb.TickMessage {
	msg := tickMsgPool.Get().(*pb.TickMessage)
	// Reset fields to zero to avoid carrying old data.
	*msg = pb.TickMessage{}
	return msg
}

// PutTick returns a TickMessage to the pool.
func PutTick(msg *pb.TickMessage) {
	tickMsgPool.Put(msg)
}
