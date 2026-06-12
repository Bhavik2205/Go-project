package wal

import "hash/crc32"

// checksum computes the CRC32-IEEE checksum of payload.
func checksum(payload []byte) uint32 {
	return crc32.ChecksumIEEE(payload)
}

// verifyRecord returns false when the stored CRC does not match the payload.
func verifyRecord(r *TickRecord) bool {
	payload := marshalPayload(r)
	return checksum(payload) == r.CRC32
}
