package pool

import (
	"sync"

	pb "github.com/Bhavik2205/ML-Bot/proto"
)

var TickMessagePool = sync.Pool{
	New: func() any {
		return &pb.TickMessage{}
	},
}

func GetTickMessage() *pb.TickMessage {
	return TickMessagePool.Get().(*pb.TickMessage)
}

func PutTickMessage(msg *pb.TickMessage) {
	msg.Reset()
	TickMessagePool.Put(msg)
}
