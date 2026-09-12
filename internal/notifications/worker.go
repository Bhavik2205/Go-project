package notifications

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/security"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

// Worker drains PENDING rows from notification_history and delivers them.
//
// Delivery is decoupled from the trading path on purpose: a slow or broken
// Telegram must never add latency to an order, so producers only insert a row
// and this worker does the network I/O.
type Worker struct {
	db        *gorm.DB
	registry  *Registry
	encryptor *security.Encryptor

	// pollInterval is how often the queue is swept. Alerts are low-volume, so
	// a short poll is cheaper than any pub/sub machinery here.
	pollInterval time.Duration
	batchSize    int
	maxAttempts  int

	// attempts tracks retries per row id. Kept in memory rather than adding a
	// schema column; a restart resets the count, which is the forgiving
	// direction to be wrong in.
	mu       sync.Mutex
	attempts map[uint]int

	sent    atomic.Uint64
	failed  atomic.Uint64
	retried atomic.Uint64
}

// WorkerConfig tunes the worker.
type WorkerConfig struct {
	PollInterval time.Duration
	BatchSize    int
	MaxAttempts  int
}

// DefaultWorkerConfig returns sensible defaults.
func DefaultWorkerConfig() WorkerConfig {
	return WorkerConfig{
		PollInterval: 5 * time.Second,
		BatchSize:    20,
		MaxAttempts:  4,
	}
}

// NewWorker builds a delivery worker. encryptor may be nil, in which case
// configs are read as plaintext.
func NewWorker(dbc *db.DBClient, encryptor *security.Encryptor, cfg WorkerConfig) *Worker {
	if cfg.PollInterval <= 0 {
		cfg.PollInterval = 5 * time.Second
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 20
	}
	if cfg.MaxAttempts <= 0 {
		cfg.MaxAttempts = 4
	}
	return &Worker{
		db:           dbc.DB,
		registry:     NewRegistry(),
		encryptor:    encryptor,
		pollInterval: cfg.PollInterval,
		batchSize:    cfg.BatchSize,
		maxAttempts:  cfg.MaxAttempts,
		attempts:     make(map[uint]int),
	}
}

// Run sweeps the queue until ctx is cancelled.
func (w *Worker) Run(ctx context.Context) {
	zap.L().Info("✅ Notification worker started",
		zap.Duration("poll", w.pollInterval),
		zap.Strings("channels", w.registry.Names()))

	ticker := time.NewTicker(w.pollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			w.sweep(ctx)
		case <-ctx.Done():
			zap.L().Info("Notification worker stopped",
				zap.Uint64("sent", w.sent.Load()),
				zap.Uint64("failed", w.failed.Load()))
			return
		}
	}
}

// sweep processes one batch of pending rows.
func (w *Worker) sweep(ctx context.Context) {
	var rows []db.NotificationHistory
	err := w.db.WithContext(ctx).
		Where("status = ?", StatusPending).
		Order("created_at ASC").
		Limit(w.batchSize).
		Find(&rows).Error
	if err != nil {
		zap.L().Warn("notification sweep query failed", zap.Error(err))
		return
	}

	for i := range rows {
		if ctx.Err() != nil {
			return
		}
		w.deliver(ctx, &rows[i])
	}
}

// deliver sends one row and records the outcome.
func (w *Worker) deliver(ctx context.Context, row *db.NotificationHistory) {
	provider, ok := w.registry.Get(row.ChannelType)
	if !ok {
		w.markFailed(row, fmt.Sprintf("no provider for channel %q", row.ChannelType))
		return
	}

	cfg, err := w.channelConfig(ctx, row.UserID, row.ChannelType)
	if err != nil {
		// A missing or disabled channel will not fix itself.
		w.markFailed(row, err.Error())
		return
	}

	id, err := provider.Send(ctx, cfg, row.Message)
	if err == nil {
		w.markSent(row, id)
		return
	}

	if IsPermanent(err) {
		w.markFailed(row, err.Error())
		return
	}

	// Transient: leave the row PENDING so the next sweep retries it, until
	// the attempt budget runs out.
	n := w.bumpAttempt(row.ID)
	if n >= w.maxAttempts {
		w.markFailed(row, fmt.Sprintf("giving up after %d attempts: %v", n, err))
		return
	}

	w.retried.Add(1)
	zap.L().Warn("notification delivery failed, will retry",
		zap.Uint("id", row.ID),
		zap.String("channel", row.ChannelType),
		zap.Int("attempt", n),
		zap.Error(err))
}

// channelConfig loads and decrypts the channel config for a user.
func (w *Worker) channelConfig(ctx context.Context, userID uint, channelType string) (json.RawMessage, error) {
	var ch db.NotificationChannel
	err := w.db.WithContext(ctx).
		Where("user_id = ? AND channel_type = ?", userID, channelType).
		First(&ch).Error
	if err == gorm.ErrRecordNotFound {
		return nil, fmt.Errorf("channel %s is not configured for this user", channelType)
	}
	if err != nil {
		return nil, fmt.Errorf("load channel: %w", err)
	}
	if !ch.IsEnabled {
		return nil, fmt.Errorf("channel %s is disabled", channelType)
	}

	return DecryptConfig(w.encryptor, ch.Config)
}

func (w *Worker) markSent(row *db.NotificationHistory, providerID string) {
	now := time.Now()
	err := w.db.Model(&db.NotificationHistory{}).Where("id = ?", row.ID).Updates(map[string]any{
		"status":              StatusSent,
		"provider_message_id": providerID,
		"sent_at":             &now,
		"error_message":       "",
	}).Error
	if err != nil {
		zap.L().Error("could not mark notification sent", zap.Uint("id", row.ID), zap.Error(err))
		return
	}

	w.clearAttempt(row.ID)
	w.sent.Add(1)
	zap.L().Info("notification sent",
		zap.Uint("id", row.ID),
		zap.String("channel", row.ChannelType),
		zap.String("event", row.EventType))
}

func (w *Worker) markFailed(row *db.NotificationHistory, reason string) {
	err := w.db.Model(&db.NotificationHistory{}).Where("id = ?", row.ID).Updates(map[string]any{
		"status":        StatusFailed,
		"error_message": reason,
	}).Error
	if err != nil {
		zap.L().Error("could not mark notification failed", zap.Uint("id", row.ID), zap.Error(err))
	}

	w.clearAttempt(row.ID)
	w.failed.Add(1)
	zap.L().Warn("notification failed",
		zap.Uint("id", row.ID),
		zap.String("channel", row.ChannelType),
		zap.String("reason", reason))
}

func (w *Worker) bumpAttempt(id uint) int {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.attempts[id]++
	return w.attempts[id]
}

func (w *Worker) clearAttempt(id uint) {
	w.mu.Lock()
	delete(w.attempts, id)
	w.mu.Unlock()
}

// WorkerStats reports delivery counters.
type WorkerStats struct {
	Sent    uint64 `json:"sent"`
	Failed  uint64 `json:"failed"`
	Retried uint64 `json:"retried"`
	Pending int64  `json:"pending"`
}

// Stats returns counters plus the current queue depth.
func (w *Worker) Stats() WorkerStats {
	var pending int64
	w.db.Model(&db.NotificationHistory{}).Where("status = ?", StatusPending).Count(&pending)

	return WorkerStats{
		Sent:    w.sent.Load(),
		Failed:  w.failed.Load(),
		Retried: w.retried.Load(),
		Pending: pending,
	}
}
