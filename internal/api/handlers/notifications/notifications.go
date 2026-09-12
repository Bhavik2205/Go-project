// Package notifications manages per-user alert channels and their delivery log.
package notifications

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api/handlers/httpx"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/middleware"
	notify "github.com/Bhavik2205/ML-Bot/internal/notifications"
	"github.com/Bhavik2205/ML-Bot/internal/security"
	"go.uber.org/zap"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

// providers is the source of truth for which channels exist and what a valid
// config looks like, so the API and the delivery worker cannot disagree.
var providers = notify.NewRegistry()

func validChannel(name string) bool {
	_, ok := providers.Get(name)
	return ok
}

type channelDTO struct {
	ID          uint   `json:"id"`
	ChannelType string `json:"channel_type"`
	IsEnabled   bool   `json:"is_enabled"`
	// Configured reports whether credentials are present, without returning
	// them. Bot tokens and phone numbers must not be echoed back to the UI.
	Configured bool `json:"configured"`
	// Encrypted reports whether the stored credentials are sealed at rest.
	Encrypted bool      `json:"encrypted"`
	UpdatedAt time.Time `json:"updated_at"`
}

type historyDTO struct {
	ID          uint       `json:"id"`
	ChannelType string     `json:"channel_type"`
	EventType   string     `json:"event_type"`
	Message     string     `json:"message"`
	Status      string     `json:"status"`
	ErrorMsg    string     `json:"error_message,omitempty"`
	CreatedAt   time.Time  `json:"created_at"`
	SentAt      *time.Time `json:"sent_at,omitempty"`
}

type upsertRequest struct {
	ChannelType string          `json:"channel_type"`
	IsEnabled   bool            `json:"is_enabled"`
	Config      json.RawMessage `json:"config"`
}

func userIDFrom(r *http.Request) (uint, bool) {
	id := middleware.UserIDFromContext(r.Context())
	return id, id != 0
}

// encryptor protects channel credentials at rest. Nil when no key is set, in
// which case configs are stored as plaintext and a warning is logged once.
var encryptor = loadEncryptor()

func loadEncryptor() *security.Encryptor {
	e, err := security.NewEncryptorFromEnv()
	if err != nil {
		// Not fatal: the feature still works, but bot tokens sit in the
		// database in the clear, which the caller should know about.
		zap.L().Warn("notification channel credentials will be stored UNENCRYPTED",
			zap.String("fix", "set "+security.EncryptionKeyEnv),
			zap.Error(err))
		return nil
	}
	return e
}

// HandleListChannels returns the user's notification channels.
func HandleListChannels(dbc *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if dbc == nil {
			httpx.WriteError(w, http.StatusServiceUnavailable, r, httpx.CodeUnavailable, "database unavailable", nil)
			return
		}
		userID, ok := userIDFrom(r)
		if !ok {
			httpx.WriteError(w, http.StatusUnauthorized, r, httpx.CodeForbidden, "not authenticated", nil)
			return
		}

		var rows []db.NotificationChannel
		if err := dbc.DB.Where("user_id = ?", userID).Order("channel_type ASC").Find(&rows).Error; err != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal, "failed to load channels", err.Error())
			return
		}

		out := make([]channelDTO, 0, len(rows))
		for _, c := range rows {
			out = append(out, channelDTO{
				ID: c.ID, ChannelType: c.ChannelType, IsEnabled: c.IsEnabled,
				Configured: notify.HasCredentials(encryptor, c.Config),
				Encrypted:  notify.IsEncrypted(c.Config), UpdatedAt: c.UpdatedAt,
			})
		}
		httpx.WriteSuccess(w, http.StatusOK, r, out)
	}
}

// HandleUpsertChannel creates or updates one channel.
func HandleUpsertChannel(dbc *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if dbc == nil {
			httpx.WriteError(w, http.StatusServiceUnavailable, r, httpx.CodeUnavailable, "database unavailable", nil)
			return
		}
		userID, ok := userIDFrom(r)
		if !ok {
			httpx.WriteError(w, http.StatusUnauthorized, r, httpx.CodeForbidden, "not authenticated", nil)
			return
		}

		var req upsertRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			httpx.WriteError(w, http.StatusBadRequest, r, httpx.CodeBadRequest, "request body must be valid JSON", err.Error())
			return
		}
		if !validChannel(req.ChannelType) {
			httpx.WriteError(w, http.StatusBadRequest, r, httpx.CodeBadRequest,
				"channel_type must be one of: telegram, whatsapp", nil)
			return
		}
		if len(req.Config) == 0 {
			req.Config = json.RawMessage(`{}`)
		}

		// Merge over whatever is stored, so the UI can send only the fields
		// the user actually retyped. Without this, toggling "enabled" would
		// wipe the bot token, since the API never returns it to be resent.
		merged, err := mergeWithStored(dbc, userID, req.ChannelType, req.Config)
		if err != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal,
				"failed to read existing channel config", err.Error())
			return
		}

		// Validate against the provider that will actually send, so bad
		// credentials are rejected now rather than at the first real alert.
		if req.IsEnabled {
			provider, _ := providers.Get(req.ChannelType)
			if err := provider.Validate(merged); err != nil {
				httpx.WriteError(w, http.StatusBadRequest, r, httpx.CodeBadRequest, err.Error(), nil)
				return
			}
		}

		sealed, err := notify.EncryptConfig(encryptor, merged)
		if err != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal,
				"failed to secure channel config", err.Error())
			return
		}

		row := db.NotificationChannel{
			UserID:      userID,
			ChannelType: req.ChannelType,
			IsEnabled:   req.IsEnabled,
			Config:      sealed,
		}
		err = dbc.DB.Clauses(clause.OnConflict{
			Columns:   []clause.Column{{Name: "user_id"}, {Name: "channel_type"}},
			DoUpdates: clause.AssignmentColumns([]string{"is_enabled", "config", "updated_at"}),
		}).Create(&row).Error
		if err != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal, "failed to save channel", err.Error())
			return
		}

		httpx.WriteSuccess(w, http.StatusOK, r, channelDTO{
			ID: row.ID, ChannelType: row.ChannelType, IsEnabled: row.IsEnabled,
			Configured: notify.HasCredentials(encryptor, row.Config),
			Encrypted:  notify.IsEncrypted(row.Config), UpdatedAt: row.UpdatedAt,
		})
	}
}

// HandleDeleteChannel removes a channel.
func HandleDeleteChannel(dbc *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if dbc == nil {
			httpx.WriteError(w, http.StatusServiceUnavailable, r, httpx.CodeUnavailable, "database unavailable", nil)
			return
		}
		userID, ok := userIDFrom(r)
		if !ok {
			httpx.WriteError(w, http.StatusUnauthorized, r, httpx.CodeForbidden, "not authenticated", nil)
			return
		}

		channelType := r.URL.Query().Get("channel_type")
		if channelType == "" {
			httpx.WriteError(w, http.StatusBadRequest, r, httpx.CodeBadRequest, "channel_type is required", nil)
			return
		}

		// Scoped by user_id so one user cannot delete another user's channel.
		res := dbc.DB.Where("user_id = ? AND channel_type = ?", userID, channelType).
			Delete(&db.NotificationChannel{})
		if res.Error != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal, "failed to delete channel", res.Error.Error())
			return
		}
		if res.RowsAffected == 0 {
			httpx.WriteError(w, http.StatusNotFound, r, httpx.CodeNotFound, "channel not found", nil)
			return
		}
		httpx.WriteSuccess(w, http.StatusOK, r, map[string]any{"deleted": true, "channel_type": channelType})
	}
}

// HandleHistory returns the delivery log, newest first.
func HandleHistory(dbc *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if dbc == nil {
			httpx.WriteError(w, http.StatusServiceUnavailable, r, httpx.CodeUnavailable, "database unavailable", nil)
			return
		}
		userID, ok := userIDFrom(r)
		if !ok {
			httpx.WriteError(w, http.StatusUnauthorized, r, httpx.CodeForbidden, "not authenticated", nil)
			return
		}
		limit := httpx.QueryInt(r, "limit", 50, 1, 200)

		q := dbc.DB.Where("user_id = ?", userID).Order("created_at DESC").Limit(limit)
		if s := r.URL.Query().Get("status"); s != "" {
			q = q.Where("status = ?", s)
		}

		var rows []db.NotificationHistory
		if err := q.Find(&rows).Error; err != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal, "failed to load history", err.Error())
			return
		}

		out := make([]historyDTO, 0, len(rows))
		for _, h := range rows {
			out = append(out, historyDTO{
				ID: h.ID, ChannelType: h.ChannelType, EventType: h.EventType,
				Message: h.Message, Status: h.Status, ErrorMsg: h.ErrorMessage,
				CreatedAt: h.CreatedAt, SentAt: h.SentAt,
			})
		}
		httpx.WriteSuccess(w, http.StatusOK, r, out)
	}
}

// HandleTest queues a test notification on an enabled channel.
//
// It records a PENDING row rather than claiming delivery, because no sender is
// wired up yet. Reporting success for a message nothing will send would make
// the feature look working when it is not.
func HandleTest(dbc *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if dbc == nil {
			httpx.WriteError(w, http.StatusServiceUnavailable, r, httpx.CodeUnavailable, "database unavailable", nil)
			return
		}
		userID, ok := userIDFrom(r)
		if !ok {
			httpx.WriteError(w, http.StatusUnauthorized, r, httpx.CodeForbidden, "not authenticated", nil)
			return
		}

		channelType := r.URL.Query().Get("channel_type")
		if !validChannel(channelType) {
			httpx.WriteError(w, http.StatusBadRequest, r, httpx.CodeBadRequest,
				"channel_type must be one of: telegram, whatsapp", nil)
			return
		}

		var ch db.NotificationChannel
		err := dbc.DB.Where("user_id = ? AND channel_type = ?", userID, channelType).First(&ch).Error
		if err == gorm.ErrRecordNotFound {
			httpx.WriteError(w, http.StatusNotFound, r, httpx.CodeNotFound, "channel is not configured", nil)
			return
		}
		if err != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal, "failed to load channel", err.Error())
			return
		}
		if !ch.IsEnabled {
			httpx.WriteError(w, http.StatusBadRequest, r, httpx.CodeBadRequest, "channel is disabled", nil)
			return
		}

		row := db.NotificationHistory{
			UserID:      userID,
			ChannelType: channelType,
			EventType:   "TEST",
			Message:     "Test notification from the trading dashboard.",
			Status:      "PENDING",
		}
		if err := dbc.DB.Create(&row).Error; err != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal, "failed to queue notification", err.Error())
			return
		}

		httpx.WriteSuccess(w, http.StatusAccepted, r, map[string]any{
			"queued": true,
			"id":     row.ID,
			"note":   "Queued as PENDING. No delivery worker is running yet, so it will not actually be sent.",
		})
	}
}

// mergeWithStored overlays incoming fields on the stored config.
//
// Credentials are write-only in the API: the UI never receives them, so it
// cannot send them back. An update that omits a field must therefore keep the
// stored value rather than clearing it. An empty string is treated as "not
// supplied" for the same reason.
func mergeWithStored(dbc *db.DBClient, userID uint, channelType string, incoming json.RawMessage) (json.RawMessage, error) {
	var existing db.NotificationChannel
	err := dbc.DB.Where("user_id = ? AND channel_type = ?", userID, channelType).First(&existing).Error
	if err == gorm.ErrRecordNotFound {
		return incoming, nil
	}
	if err != nil {
		return nil, err
	}

	plain, err := notify.DecryptConfig(encryptor, existing.Config)
	if err != nil {
		// Unreadable stored config: fall back to whatever was sent rather
		// than blocking the user from fixing it.
		return incoming, nil
	}

	base := map[string]any{}
	_ = json.Unmarshal(plain, &base)

	update := map[string]any{}
	if err := json.Unmarshal(incoming, &update); err != nil {
		return nil, fmt.Errorf("config must be a JSON object: %w", err)
	}

	for k, v := range update {
		if s, ok := v.(string); ok && s == "" {
			continue // blank means "leave what is stored"
		}
		base[k] = v
	}

	out, err := json.Marshal(base)
	if err != nil {
		return nil, fmt.Errorf("merge config: %w", err)
	}
	return out, nil
}
