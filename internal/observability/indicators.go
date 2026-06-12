package observability

// Indicator metrics helpers are in marketdata.go
// This file is reserved for future indicator-specific functionality.

// RecordIndicatorComputationTime is a convenience wrapper for recording indicator latency.
func RecordIndicatorComputationTime(latencyMs float64) {
	RecordIndicatorLatency(latencyMs)
}
