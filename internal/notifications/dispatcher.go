package notifications

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/db"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

// Dispatcher queues alerts for delivery.
//
// Notify only inserts a row; the Worker does the network I/O. That split is
// what lets the trading path raise an alert without ever waiting on Telegram.
type Dispatcher struct {
	db     *gorm.DB
	userID uint

	// enabled caches which channels are switched on, refreshed periodically.
	// Without it, every alert would hit the database for the channel list.
	mu          sync.RWMutex
	enabled     []string
	lastRefresh time.Time

	queued  atomic.Uint64
	skipped atomic.Uint64

	// dedupe suppresses repeats of the same alert inside a short window, so a
	// condition that stays true for an hour does not send an alert per candle.
	dedupeMu     sync.Mutex
	lastSent     map[string]time.Time
	dedupeWindow time.Duration
}

// NewDispatcher builds a dispatcher for one trading account.
func NewDispatcher(dbc *db.DBClient, userID uint) *Dispatcher {
	return &Dispatcher{
		db:           dbc.DB,
		userID:       userID,
		lastSent:     make(map[string]time.Time),
		dedupeWindow: time.Minute,
	}
}

// channelRefreshInterval bounds how stale the enabled-channel cache can be.
const channelRefreshInterval = 30 * time.Second

// enabledChannels returns the user's enabled channels, refreshing on demand.
func (d *Dispatcher) enabledChannels() []string {
	d.mu.RLock()
	fresh := time.Since(d.lastRefresh) < channelRefreshInterval
	cached := d.enabled
	d.mu.RUnlock()

	if fresh {
		return cached
	}

	var rows []db.NotificationChannel
	if err := d.db.Where("user_id = ? AND is_enabled = ?", d.userID, true).Find(&rows).Error; err != nil {
		zap.L().Warn("could not refresh notification channels", zap.Error(err))
		return cached
	}

	names := make([]string, 0, len(rows))
	for _, r := range rows {
		names = append(names, r.ChannelType)
	}

	d.mu.Lock()
	d.enabled = names
	d.lastRefresh = time.Now()
	d.mu.Unlock()

	return names
}

// Notify queues one alert on every enabled channel.
//
// dedupeKey suppresses repeats within the dedupe window; pass "" to always
// send. Errors are logged, never returned: an alert failing to queue must not
// change what the trading code does.
func (d *Dispatcher) Notify(eventType, message, dedupeKey string) {
	channels := d.enabledChannels()
	if len(channels) == 0 {
		d.skipped.Add(1)
		return
	}

	if dedupeKey != "" && d.suppressed(dedupeKey) {
		d.skipped.Add(1)
		return
	}

	for _, ch := range channels {
		row := db.NotificationHistory{
			UserID:      d.userID,
			ChannelType: ch,
			EventType:   eventType,
			Message:     message,
			Status:      StatusPending,
		}
		if err := d.db.Create(&row).Error; err != nil {
			zap.L().Warn("could not queue notification",
				zap.String("channel", ch),
				zap.String("event", eventType),
				zap.Error(err))
			continue
		}
		d.queued.Add(1)
	}
}

// suppressed reports whether this key was sent inside the dedupe window, and
// records it when it was not.
func (d *Dispatcher) suppressed(key string) bool {
	d.dedupeMu.Lock()
	defer d.dedupeMu.Unlock()

	now := time.Now()
	if last, ok := d.lastSent[key]; ok && now.Sub(last) < d.dedupeWindow {
		return true
	}
	d.lastSent[key] = now

	// Keep the map from growing without bound over a long session.
	if len(d.lastSent) > 1000 {
		for k, t := range d.lastSent {
			if now.Sub(t) > d.dedupeWindow*10 {
				delete(d.lastSent, k)
			}
		}
	}
	return false
}

// --- Convenience helpers for the events the system actually raises ---------

// TradeExecuted reports a fill.
func (d *Dispatcher) TradeExecuted(mode, strategy, symbol, action, side string, qty int64, price float64) {
	msg := fmt.Sprintf(
		"%s %s %s\n%s %s x%d @ %.2f\nStrategy: %s",
		emojiFor(side), action, symbol, mode, side, qty, price, strategy,
	)
	// Every fill is distinct, so no dedupe key.
	d.Notify(EventTradeExecution, msg, "")
}

// KillSwitchTripped reports that order placement has halted.
func (d *Dispatcher) KillSwitchTripped(reason string) {
	msg := fmt.Sprintf("🛑 TRADING HALTED\nReason: %s\n\nNo further orders will be placed until trading is resumed.", reason)
	d.Notify(EventErrorAlert, msg, "kill_switch:"+reason)
}

// PnLThreshold reports that realised PnL has crossed a configured level.
func (d *Dispatcher) PnLThreshold(pnl, threshold float64) {
	verb := "above"
	icon := "📈"
	if pnl < 0 {
		verb = "below"
		icon = "📉"
	}
	msg := fmt.Sprintf("%s P&L %s threshold\nRealised: %.2f (threshold %.2f)", icon, verb, pnl, threshold)
	// One alert per crossing direction, not per candle while it stays crossed.
	d.Notify(EventPnLThreshold, msg, fmt.Sprintf("pnl:%s", verb))
}

// ErrorAlert reports an operational problem.
func (d *Dispatcher) ErrorAlert(subject, detail string) {
	msg := fmt.Sprintf("⚠️ %s\n%s", subject, detail)
	d.Notify(EventErrorAlert, msg, "error:"+subject)
}

func emojiFor(side string) string {
	switch side {
	case "LONG":
		return "🟢"
	case "SHORT":
		return "🔴"
	default:
		return "•"
	}
}

// DispatcherStats reports queueing counters.
type DispatcherStats struct {
	Queued          uint64   `json:"queued"`
	Skipped         uint64   `json:"skipped"`
	EnabledChannels []string `json:"enabled_channels"`
}

// Stats returns dispatcher counters.
func (d *Dispatcher) Stats() DispatcherStats {
	d.mu.RLock()
	enabled := append([]string(nil), d.enabled...)
	d.mu.RUnlock()

	return DispatcherStats{
		Queued:          d.queued.Load(),
		Skipped:         d.skipped.Load(),
		EnabledChannels: enabled,
	}
}
