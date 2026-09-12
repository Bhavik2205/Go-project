// Package notifications manages per-user alert channels and their delivery log.
package notifications

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api/handlers/httpx"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/middleware"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

// Supported channel types. Anything else is rejected rather than stored, so a
// typo does not create a channel that can never deliver.
var validChannels = map[string]bool{
	"telegram": true,
	"whatsapp": true,
}

type channelDTO struct {
	ID          uint   `json:"id"`
	ChannelType string `json:"channel_type"`
	IsEnabled   bool   `json:"is_enabled"`
	// Configured reports whether credentials are present, without returning
	// them. Bot tokens and phone numbers must not be echoed back to the UI.
	Configured bool      `json:"configured"`
	UpdatedAt  time.Time `json:"updated_at"`
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

// hasConfig reports whether a stored config holds anything beyond an empty
// object, which is how "credentials present" is decided without reading them.
func hasConfig(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return false
	}
	return len(m) > 0
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
				Configured: hasConfig(c.Config), UpdatedAt: c.UpdatedAt,
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
		if !validChannels[req.ChannelType] {
			httpx.WriteError(w, http.StatusBadRequest, r, httpx.CodeBadRequest,
				"channel_type must be one of: telegram, whatsapp", nil)
			return
		}
		if len(req.Config) == 0 {
			req.Config = json.RawMessage(`{}`)
		}
		// Enabling a channel with no credentials would queue notifications
		// that can never be delivered.
		if req.IsEnabled && !hasConfig(req.Config) {
			httpx.WriteError(w, http.StatusBadRequest, r, httpx.CodeBadRequest,
				"cannot enable a channel with an empty config", nil)
			return
		}

		row := db.NotificationChannel{
			UserID:      userID,
			ChannelType: req.ChannelType,
			IsEnabled:   req.IsEnabled,
			Config:      req.Config,
		}
		err := dbc.DB.Clauses(clause.OnConflict{
			Columns:   []clause.Column{{Name: "user_id"}, {Name: "channel_type"}},
			DoUpdates: clause.AssignmentColumns([]string{"is_enabled", "config", "updated_at"}),
		}).Create(&row).Error
		if err != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal, "failed to save channel", err.Error())
			return
		}

		httpx.WriteSuccess(w, http.StatusOK, r, channelDTO{
			ID: row.ID, ChannelType: row.ChannelType, IsEnabled: row.IsEnabled,
			Configured: hasConfig(row.Config), UpdatedAt: row.UpdatedAt,
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
		if !validChannels[channelType] {
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
