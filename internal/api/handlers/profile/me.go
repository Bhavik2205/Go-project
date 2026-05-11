package profile

import (
	"encoding/json"
	"errors"
	"net/http"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/contracts"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/middleware"
	"github.com/Bhavik2205/ML-Bot/internal/validation"
	"gorm.io/gorm"
)

type meResponse struct {
	ID        uint      `json:"id"`
	Email     string    `json:"email"`
	UserName  string    `json:"userName"`
	IsActive  bool      `json:"isActive"`
	CreatedAt time.Time `json:"createdAt"`
}

type patchMeRequest struct {
	UserName string `json:"userName" validate:"required,min=1,max=100"`
}

// HandleGetMe handles GET /api/v1/me
func HandleGetMe(dbClient *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		userID := middleware.UserIDFromContext(r.Context())
		if userID == 0 {
			writeError(w, http.StatusUnauthorized, r, "UNAUTHORIZED", "not authenticated", nil)
			return
		}

		var user db.User
		if err := dbClient.DB.First(&user, userID).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				writeError(w, http.StatusNotFound, r, "NOT_FOUND", "user not found", nil)
				return
			}
			writeError(w, http.StatusInternalServerError, r, "INTERNAL_ERROR", "failed to fetch user", nil)
			return
		}

		writeSuccess(w, http.StatusOK, r, meResponse{
			ID:        user.ID,
			Email:     user.Email,
			UserName:  user.UserName,
			IsActive:  user.IsActive,
			CreatedAt: user.CreatedAt,
		})
	}
}

// HandlePatchMe handles PATCH /api/v1/me — updates UserName only
func HandlePatchMe(dbClient *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		userID := middleware.UserIDFromContext(r.Context())
		if userID == 0 {
			writeError(w, http.StatusUnauthorized, r, "UNAUTHORIZED", "not authenticated", nil)
			return
		}

		var req patchMeRequest
		if !validation.BindAndValidate(w, r, &req) {
			return
		}

		if err := dbClient.DB.Model(&db.User{}).Where("id = ?", userID).
			Update("user_name", req.UserName).Error; err != nil {
			writeError(w, http.StatusInternalServerError, r, "INTERNAL_ERROR", "failed to update user", nil)
			return
		}

		var user db.User
		if err := dbClient.DB.First(&user, userID).Error; err != nil {
			writeError(w, http.StatusInternalServerError, r, "INTERNAL_ERROR", "failed to fetch updated user", nil)
			return
		}

		writeSuccess(w, http.StatusOK, r, meResponse{
			ID:        user.ID,
			Email:     user.Email,
			UserName:  user.UserName,
			IsActive:  user.IsActive,
			CreatedAt: user.CreatedAt,
		})
	}
}

func writeSuccess(w http.ResponseWriter, status int, r *http.Request, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(contracts.NewSuccess(r.Header.Get("X-Request-ID"), data))
}

func writeError(w http.ResponseWriter, status int, r *http.Request, code, message string, details any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(contracts.NewError(r.Header.Get("X-Request-ID"), code, message, details))
}
