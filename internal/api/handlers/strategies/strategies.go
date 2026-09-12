// Package strategies exposes the strategy engine's configuration and live
// counters for the monitoring UI.
package strategies

import (
	"net/http"

	"github.com/Bhavik2205/ML-Bot/internal/api/handlers/httpx"
	"github.com/Bhavik2205/ML-Bot/internal/strategy"
)

type strategyInfo struct {
	Name       string `json:"name"`
	Registered bool   `json:"registered"`
}

type listResponse struct {
	Registered []strategyInfo        `json:"registered"`
	Engine     *strategy.EngineStats `json:"engine,omitempty"`
	Enabled    bool                  `json:"enabled"`
	Mode       string                `json:"mode"`
}

// HandleList returns every registered strategy plus engine counters.
//
// It answers 200 even when the engine is disabled, because the UI still wants
// to show which strategies exist and that trading is currently off.
func HandleList(t *httpx.Trading) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		names := strategy.Names()
		infos := make([]strategyInfo, 0, len(names))
		for _, n := range names {
			infos = append(infos, strategyInfo{Name: n, Registered: true})
		}

		resp := listResponse{Registered: infos, Enabled: t.Enabled()}
		if t.Enabled() {
			stats := t.Engine.Stats()
			resp.Engine = &stats
			resp.Mode = t.Mode
		}
		httpx.WriteSuccess(w, http.StatusOK, r, resp)
	}
}

// HandleStats returns just the engine counters.
func HandleStats(t *httpx.Trading) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !t.Enabled() {
			httpx.Unavailable(w, r, "strategy engine")
			return
		}
		httpx.WriteSuccess(w, http.StatusOK, r, t.Engine.Stats())
	}
}
