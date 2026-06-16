package wal

import (
	"encoding/binary"
	"math"
)

// TickRecord is the on-disk representation of a single tick.
// Only the fields needed for replay are stored — keep it small.
type TickRecord struct {
	Timestamp         int64   // IngestTime.UnixNano()
	InstrumentToken   uint32
	LastPrice         float64
	Volume            uint32
	ExchangeTimestamp int64 // EventTime.UnixNano()
	CRC32             uint32
}

const recordPayloadSize = 8 + 4 + 8 + 4 + 8 // = 32 bytes (without CRC)
const RecordSize = recordPayloadSize + 4       // + 4 bytes CRC32

// marshalPayload serialises the record fields (without CRC) into a fixed-size
// byte slice so checksumming and writing are allocation-free on the hot path.
func marshalPayload(r *TickRecord) []byte {
	buf := make([]byte, recordPayloadSize)
	binary.LittleEndian.PutUint64(buf[0:], uint64(r.Timestamp))
	binary.LittleEndian.PutUint32(buf[8:], r.InstrumentToken)
	binary.LittleEndian.PutUint64(buf[12:], math.Float64bits(r.LastPrice))
	binary.LittleEndian.PutUint32(buf[20:], r.Volume)
	binary.LittleEndian.PutUint64(buf[24:], uint64(r.ExchangeTimestamp))
	return buf
}

// MarshalRecord serialises the full record including CRC into buf.
// buf must be at least RecordSize bytes.
func MarshalRecord(r *TickRecord, buf []byte) {
	payload := marshalPayload(r)
	copy(buf[:recordPayloadSize], payload)
	binary.LittleEndian.PutUint32(buf[recordPayloadSize:], r.CRC32)
}

// unmarshalRecord deserialises a record from buf (must be RecordSize bytes).
func unmarshalRecord(buf []byte) TickRecord {
	var r TickRecord
	r.Timestamp = int64(binary.LittleEndian.Uint64(buf[0:]))
	r.InstrumentToken = binary.LittleEndian.Uint32(buf[8:])
	r.LastPrice = math.Float64frombits(binary.LittleEndian.Uint64(buf[12:]))
	r.Volume = binary.LittleEndian.Uint32(buf[20:])
	r.ExchangeTimestamp = int64(binary.LittleEndian.Uint64(buf[24:]))
	r.CRC32 = binary.LittleEndian.Uint32(buf[recordPayloadSize:])
	return r
}
