// Package positions serves the live position book and PnL summary.
package positions

import (
	"net/http"

	"github.com/Bhavik2205/ML-Bot/internal/api/handlers/httpx"
	monitor "github.com/Bhavik2205/ML-Bot/internal/execution"
)

type positionsResponse struct {
	Mode          string                  `json:"mode"`
	Positions     []monitor.PaperPosition `json:"positions"`
	RealizedPnL   float64                 `json:"realized_pnl"`
	UnrealizedPnL float64                 `json:"unrealized_pnl"`
	FeesPaid      float64                 `json:"fees_paid"`
	NetPnL        float64                 `json:"net_pnl"`
}

// HandleList returns open positions and the account PnL.
//
// The book is the in-memory source of truth for the running session; the
// positions table is the durable copy written asynchronously behind it.
// Serving the book keeps the UI correct even while persistence lags.
func HandleList(t *httpx.Trading) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if t == nil || t.Paper == nil {
			httpx.Unavailable(w, r, "paper trading")
			return
		}

		includeFlat := r.URL.Query().Get("include_flat") == "true"
		book := t.Paper.Book()

		list := book.OpenPositions()
		if includeFlat {
			list = book.Positions()
		}

		stats := t.Paper.Stats()
		httpx.WriteSuccess(w, http.StatusOK, r, positionsResponse{
			Mode:          t.Mode,
			Positions:     list,
			RealizedPnL:   stats.RealizedPnL,
			UnrealizedPnL: stats.UnrealizedPnL,
			FeesPaid:      stats.FeesPaid,
			NetPnL:        stats.NetPnL,
		})
	}
}

// HandlePnL returns only the PnL summary, for a compact dashboard tile.
func HandlePnL(t *httpx.Trading) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if t == nil || t.Paper == nil {
			httpx.Unavailable(w, r, "paper trading")
			return
		}
		httpx.WriteSuccess(w, http.StatusOK, r, t.Paper.Stats())
	}
}
