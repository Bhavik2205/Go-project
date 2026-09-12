// Package backtest exposes the historical replay runner over HTTP.
package backtest

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api/handlers/httpx"
	bt "github.com/Bhavik2205/ML-Bot/internal/backtest"
	"github.com/Bhavik2205/ML-Bot/internal/db"
)

// runTimeout bounds a single replay. A backtest is a synchronous request, so
// it must not be able to hold a connection open indefinitely.
const runTimeout = 60 * time.Second

// HandleRun replays history and returns the result.
//
// The run is synchronous: results come back in the response rather than via a
// job id, which keeps the UI simple. The candle cap and timeout together bound
// how long that can take.
func HandleRun(runner *bt.Runner) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if runner == nil {
			httpx.WriteError(w, http.StatusServiceUnavailable, r, httpx.CodeUnavailable,
				"backtesting is unavailable because the database is not connected", nil)
			return
		}

		var req bt.Request
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			httpx.WriteError(w, http.StatusBadRequest, r, httpx.CodeBadRequest,
				"request body must be valid JSON", err.Error())
			return
		}

		ctx, cancel := contextWithTimeout(r, runTimeout)
		defer cancel()

		res, err := runner.Run(ctx, req)
		if err != nil {
			// Validation problems and "no data for this range" are both the
			// caller's to fix, so they are 400 rather than 500.
			httpx.WriteError(w, http.StatusBadRequest, r, httpx.CodeBadRequest, err.Error(), nil)
			return
		}

		httpx.WriteSuccess(w, http.StatusOK, r, res)
	}
}

type historyDTO struct {
	ID           uint            `json:"id"`
	StrategyName string          `json:"strategy_name"`
	Status       string          `json:"status"`
	FromTime     time.Time       `json:"from_time"`
	ToTime       time.Time       `json:"to_time"`
	CreatedAt    time.Time       `json:"created_at"`
	Result       json.RawMessage `json:"result,omitempty"`
	ErrorMessage string          `json:"error_message,omitempty"`
}

// HandleHistory lists previously stored backtest jobs.
//
// The synchronous runner above does not write rows here; this serves jobs
// created by other tooling against the backtest_jobs table, and returns an
// empty list when there are none.
func HandleHistory(dbc *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if dbc == nil {
			httpx.WriteError(w, http.StatusServiceUnavailable, r, httpx.CodeUnavailable, "database unavailable", nil)
			return
		}
		limit := httpx.QueryInt(r, "limit", 25, 1, 200)

		var rows []db.BacktestJob
		if err := dbc.DB.Order("created_at DESC").Limit(limit).Find(&rows).Error; err != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal,
				"failed to query backtest jobs", err.Error())
			return
		}

		out := make([]historyDTO, 0, len(rows))
		for _, j := range rows {
			out = append(out, historyDTO{
				ID: j.ID, StrategyName: j.StrategyName, Status: j.Status,
				FromTime: j.FromTime, ToTime: j.ToTime, CreatedAt: j.CreatedAt,
				Result: j.Result, ErrorMessage: j.ErrorMessage,
			})
		}
		httpx.WriteSuccess(w, http.StatusOK, r, out)
	}
}
