package wal

import (
	"encoding/binary"

	"github.com/Bhavik2205/ML-Bot/internal/marketdata/candles"
)

// TickRecord is the on-disk representation of a single tick.
// LastPrice is stored as a scaled integer (multiply by candles.PriceScale before write,
// divide on read) — same convention as market_data and ohlcv_candles.
type TickRecord struct {
	Timestamp         int64  // IngestTime.UnixNano()
	InstrumentToken   uint32
	LastPrice         int64  // scaled: raw_float * candles.PriceScale
	Volume            uint32
	ExchangeTimestamp int64  // EventTime.UnixNano()
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
	binary.LittleEndian.PutUint64(buf[12:], uint64(r.LastPrice))
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
	r.LastPrice = int64(binary.LittleEndian.Uint64(buf[12:]))
	r.Volume = binary.LittleEndian.Uint32(buf[20:])
	r.ExchangeTimestamp = int64(binary.LittleEndian.Uint64(buf[24:]))
	r.CRC32 = binary.LittleEndian.Uint32(buf[recordPayloadSize:])
	return r
}

// ScalePrice converts a float64 price to a WAL-scaled int64.
func ScalePrice(p float64) int64 { return int64(p * candles.PriceScale) }

// UnscalePrice converts a WAL-scaled int64 back to float64.
func UnscalePrice(p int64) float64 { return float64(p) / candles.PriceScale }
