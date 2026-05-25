package settings

import (
	"encoding/json"
	"net/http"

	"github.com/Bhavik2205/ML-Bot/internal/audit"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/middleware"
	"github.com/Bhavik2205/ML-Bot/internal/validation"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

type settingsResponse struct {
	Section string          `json:"section"`
	Data    json.RawMessage `json:"data"`
}

// HandleGetSettings returns a single settings section
func HandleGetSettings(dbClient *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		userID := middleware.UserIDFromContext(r.Context())
		if userID == 0 {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		section := r.URL.Query().Get("section")
		if section == "" {
			http.Error(w, "missing section query parameter", http.StatusBadRequest)
			return
		}

		var setting db.UserSetting
		err := dbClient.DB.Where("user_id = ? AND section = ?", userID, section).First(&setting).Error
		if err == gorm.ErrRecordNotFound {
			// Log audit for empty response
			audit.LogEvent(r.Context(), "settings_get", "settings", "success",
				zap.Uint("user_id", userID),
				zap.String("section", section),
				zap.Bool("found", false),
			)
			writeJSON(w, http.StatusOK, settingsResponse{
				Section: section,
				Data:    json.RawMessage(`{}`),
			})
			return
		}
		if err != nil {
			zap.L().Error("Failed to fetch user setting", zap.Error(err))
			http.Error(w, "Internal server error", http.StatusInternalServerError)
			return
		}

		audit.LogEvent(r.Context(), "settings_get", "settings", "success",
			zap.Uint("user_id", userID),
			zap.String("section", section),
			zap.Bool("found", true),
		)

		writeJSON(w, http.StatusOK, settingsResponse{
			Section: section,
			Data:    setting.SettingsJSON,
		})
	}
}

// HandleUpdateSettings upserts a settings section
func HandleUpdateSettings(dbClient *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		userID := middleware.UserIDFromContext(r.Context())
		if userID == 0 {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		var req struct {
			Section string          `json:"section" validate:"required"`
			Data    json.RawMessage `json:"data" validate:"required"`
		}
		if !validation.BindAndValidate(w, r, &req) {
			return
		}

		// Validate allowed sections
		allowedSections := map[string]bool{
			"zerodha":       true,
			"notifications": true,
			"general":       true,
			"strategy":      true,
			"data":          true,
			"performance":   true,
		}
		if !allowedSections[req.Section] {
			http.Error(w, "invalid section name", http.StatusBadRequest)
			return
		}

		// Upsert
		var setting db.UserSetting
		result := dbClient.DB.Where("user_id = ? AND section = ?", userID, req.Section).First(&setting)
		if result.Error == gorm.ErrRecordNotFound {
			setting = db.UserSetting{
				UserID:       uint(userID),
				Section:      req.Section,
				SettingsJSON: req.Data,
			}
			if err := dbClient.DB.Create(&setting).Error; err != nil {
				zap.L().Error("Failed to create user setting", zap.Error(err))
				http.Error(w, "Failed to save settings", http.StatusInternalServerError)
				return
			}
		} else if result.Error != nil {
			zap.L().Error("Failed to fetch user setting", zap.Error(result.Error))
			http.Error(w, "Internal server error", http.StatusInternalServerError)
			return
		} else {
			setting.SettingsJSON = req.Data
			if err := dbClient.DB.Save(&setting).Error; err != nil {
				zap.L().Error("Failed to update user setting", zap.Error(err))
				http.Error(w, "Failed to save settings", http.StatusInternalServerError)
				return
			}
		}

		audit.LogEvent(r.Context(), "settings_update", "settings", "success",
			zap.Uint("user_id", userID),
			zap.String("section", req.Section),
		)

		writeJSON(w, http.StatusOK, map[string]string{"message": "settings updated"})
	}
}

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(data)
}
