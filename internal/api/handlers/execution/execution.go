// Package execution exposes trading mode, execution counters and the kill
// switch that halts order placement.
package execution

import (
	"net/http"

	"github.com/Bhavik2205/ML-Bot/internal/api/handlers/httpx"
	monitor "github.com/Bhavik2205/ML-Bot/internal/execution"
	"go.uber.org/zap"
)

type statusResponse struct {
	Mode        string                 `json:"mode"`
	Enabled     bool                   `json:"enabled"`
	Guard       *monitor.GuardStats    `json:"guard,omitempty"`
	Executor    *monitor.ExecutorStats `json:"executor,omitempty"`
	Persistence *monitor.SinkStats     `json:"persistence,omitempty"`
	Paper       *monitor.PaperStats    `json:"paper,omitempty"`
}

// HandleStatus reports trading mode, kill-switch state and counters.
func HandleStatus(t *httpx.Trading) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		resp := statusResponse{Mode: t.Mode, Enabled: t.Enabled()}

		if t.Guard != nil {
			g := t.Guard.Stats()
			resp.Guard = &g
		}
		if t.Executor != nil {
			e := t.Executor.Stats()
			resp.Executor = &e
		}
		if t.Sink != nil {
			s := t.Sink.Stats()
			resp.Persistence = &s
		}
		if t.Paper != nil {
			p := t.Paper.Stats()
			resp.Paper = &p
		}
		httpx.WriteSuccess(w, http.StatusOK, r, resp)
	}
}

// HandleKill trips the kill switch, halting all order placement immediately.
//
// This is intentionally a blunt instrument with no body and no options: in
// the situation where it gets used, the operator should not have to think.
func HandleKill(t *httpx.Trading) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if t == nil || t.Guard == nil {
			httpx.Unavailable(w, r, "execution")
			return
		}

		t.Guard.Trip(monitor.TripManual)
		zap.L().Error("kill switch tripped via API",
			zap.String("remote_addr", r.RemoteAddr),
			zap.String("request_id", r.Header.Get("X-Request-ID")),
		)

		stats := t.Guard.Stats()
		httpx.WriteSuccess(w, http.StatusOK, r, map[string]any{
			"tripped": true,
			"message": "order placement halted; call /execution/resume to re-arm",
			"guard":   stats,
		})
	}
}

// HandleResume re-arms order placement after a trip.
//
// Resetting also clears the daily counters, so it is a genuine operator
// decision rather than something to automate on a timer.
func HandleResume(t *httpx.Trading) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if t == nil || t.Guard == nil {
			httpx.Unavailable(w, r, "execution")
			return
		}

		t.Guard.Reset()
		zap.L().Warn("kill switch reset via API",
			zap.String("remote_addr", r.RemoteAddr),
			zap.String("request_id", r.Header.Get("X-Request-ID")),
		)

		stats := t.Guard.Stats()
		httpx.WriteSuccess(w, http.StatusOK, r, map[string]any{
			"tripped": false,
			"message": "order placement re-armed; daily counters cleared",
			"guard":   stats,
		})
	}
}
