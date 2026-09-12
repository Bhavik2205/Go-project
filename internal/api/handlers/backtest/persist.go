package backtest

import (
	"encoding/json"
	"time"

	bt "github.com/Bhavik2205/ML-Bot/internal/backtest"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

// jobRecorder writes each run to backtest_jobs so history survives a restart.
//
// Persistence is best-effort: a database problem must not lose the result the
// caller is waiting for, so every failure is logged and the run still returns.
type jobRecorder struct {
	db *gorm.DB
}

func newJobRecorder(dbc *db.DBClient) *jobRecorder {
	if dbc == nil {
		return nil
	}
	return &jobRecorder{db: dbc.DB}
}

// start inserts a RUNNING row and returns its id. A zero id means the row
// could not be created and the later update should be skipped.
func (r *jobRecorder) start(userID uint, req bt.Request) uint {
	if r == nil {
		return 0
	}

	symbols, _ := json.Marshal(req.Instruments)
	params, _ := json.Marshal(req.Params)
	fees, _ := json.Marshal(map[string]any{
		"slippage_bps":      req.SlippageBps,
		"max_participation": req.MaxParticipation,
		"frictionless":      req.Frictionless,
	})

	now := time.Now()
	job := db.BacktestJob{
		UserID:         userID,
		StrategyName:   req.StrategyName,
		Symbols:        symbols,
		FromTime:       req.From,
		ToTime:         req.To,
		InitialCapital: req.InitialCapital,
		FeesConfig:     fees,
		Parameters:     params,
		Status:         "RUNNING",
		StartedAt:      &now,
	}
	if err := r.db.Create(&job).Error; err != nil {
		zap.L().Warn("could not record backtest job start", zap.Error(err))
		return 0
	}
	return job.ID
}

// complete stores the result and marks the row COMPLETED.
func (r *jobRecorder) complete(id uint, res *bt.Result) {
	if r == nil || id == 0 {
		return
	}

	// The trade log and equity curve can be large and are already returned to
	// the caller; store a summary so the history table stays queryable.
	summary := *res
	summary.Trades = nil
	summary.EquityCurve = nil

	payload, err := json.Marshal(summary)
	if err != nil {
		zap.L().Warn("could not encode backtest result", zap.Error(err))
		payload = []byte(`{}`)
	}

	now := time.Now()
	err = r.db.Model(&db.BacktestJob{}).Where("id = ?", id).Updates(map[string]any{
		"status":       "COMPLETED",
		"result":       payload,
		"completed_at": &now,
		"updated_at":   now,
	}).Error
	if err != nil {
		zap.L().Warn("could not record backtest completion", zap.Uint("job_id", id), zap.Error(err))
	}
}

// fail marks the row FAILED with the reason.
func (r *jobRecorder) fail(id uint, reason string) {
	if r == nil || id == 0 {
		return
	}
	now := time.Now()
	err := r.db.Model(&db.BacktestJob{}).Where("id = ?", id).Updates(map[string]any{
		"status":        "FAILED",
		"error_message": reason,
		"completed_at":  &now,
		"updated_at":    now,
	}).Error
	if err != nil {
		zap.L().Warn("could not record backtest failure", zap.Uint("job_id", id), zap.Error(err))
	}
}
