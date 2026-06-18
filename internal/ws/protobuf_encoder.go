package ws

import (
	pb "github.com/Bhavik2205/ML-Bot/proto"
	"google.golang.org/protobuf/proto"
)

type TickEvent struct {
	InstrumentToken uint32

	LastPrice int64
	BidPrice  int64
	AskPrice  int64

	BidQuantity uint32
	AskQuantity uint32

	Volume uint64

	TimestampNs int64

	SequenceID uint64
}

func MarshalTick(event TickEvent) ([]byte, error) {
	msg := &pb.TickMessage{
		InstrumentToken: event.InstrumentToken,
		LastPrice:       event.LastPrice,
		BidPrice:        event.BidPrice,
		AskPrice:        event.AskPrice,
		BidQuantity:     event.BidQuantity,
		AskQuantity:     event.AskQuantity,
		Volume:          event.Volume,
		TimestampNs:     event.TimestampNs,
		SequenceId:      event.SequenceID,
	}

	return proto.Marshal(msg)
}
