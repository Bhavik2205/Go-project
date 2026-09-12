// Package httpx holds the shared HTTP plumbing for API handlers: one
// response envelope writer and one set of query helpers, so every handler
// package does not carry its own copy.
package httpx

import (
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/Bhavik2205/ML-Bot/internal/contracts"
	monitor "github.com/Bhavik2205/ML-Bot/internal/execution"
	"github.com/Bhavik2205/ML-Bot/internal/strategy"
)

// Error codes returned by the trading and monitoring endpoints.
const (
	CodeBadRequest  = "BAD_REQUEST"
	CodeNotFound    = "NOT_FOUND"
	CodeUnavailable = "SERVICE_UNAVAILABLE"
	CodeInternal    = "INTERNAL_ERROR"
	CodeForbidden   = "FORBIDDEN"
)

// requestID pulls the correlation id set by the request-id middleware.
func requestID(r *http.Request) string { return r.Header.Get("X-Request-ID") }

// WriteSuccess encodes a success envelope.
func WriteSuccess(w http.ResponseWriter, status int, r *http.Request, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(contracts.NewSuccess(requestID(r), data))
}

// WriteError encodes an error envelope.
func WriteError(w http.ResponseWriter, status int, r *http.Request, code, message string, details any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(contracts.NewError(requestID(r), code, message, details))
}

// Unavailable is the standard response when the strategy engine or execution
// stack is not running, which is the normal state when trading is disabled.
func Unavailable(w http.ResponseWriter, r *http.Request, what string) {
	WriteError(w, http.StatusServiceUnavailable, r, CodeUnavailable,
		what+" is not enabled; set engine.enabled: true in configs/strategy.yaml", nil)
}

// QueryInt reads an integer query parameter, clamped to [min, max].
func QueryInt(r *http.Request, key string, def, min, max int) int {
	raw := r.URL.Query().Get(key)
	if raw == "" {
		return def
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		return def
	}
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// QueryUint32 reads a uint32 query parameter; ok is false when absent or
// malformed, so callers can tell "not supplied" from "supplied as zero".
func QueryUint32(r *http.Request, key string) (uint32, bool) {
	raw := r.URL.Query().Get(key)
	if raw == "" {
		return 0, false
	}
	v, err := strconv.ParseUint(raw, 10, 32)
	if err != nil {
		return 0, false
	}
	return uint32(v), true
}

// Trading bundles the live trading components the monitoring endpoints read.
// Every field may be nil: the strategy engine is opt-in, so handlers must
// degrade to 503 rather than panic.
type Trading struct {
	Engine   *strategy.Engine
	Paper    *monitor.PaperBroker
	Guard    *monitor.Guard
	Executor *monitor.OrderExecutor
	Sink     *monitor.DBTradeSink
	// Mode is "PAPER" or "LIVE".
	Mode string
	// UserID owns the simulated or live account.
	UserID uint
}

// Enabled reports whether the trading stack is wired up.
func (t *Trading) Enabled() bool {
	return t != nil && t.Engine != nil
}
