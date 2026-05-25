package auth

import (
	"errors"
	"net/http"

	"github.com/Bhavik2205/ML-Bot/internal/audit"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/validation"
	"go.uber.org/zap"
	"golang.org/x/crypto/bcrypt"
	"gorm.io/gorm"
)

type loginRequest struct {
	Email    string `json:"email"    validate:"required,email"`
	Password string `json:"password" validate:"required"`
}

func HandleLogin(dbClient *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req loginRequest
		if !validation.BindAndValidate(w, r, &req) {
			audit.LogEvent(r.Context(), "login", "user", "failure",
				zap.String("email", req.Email),
				zap.String("reason", "validation failed"),
			)
			return
		}

		var user db.User
		if err := dbClient.DB.Where("email = ?", req.Email).First(&user).Error; err != nil {
			audit.LogEvent(r.Context(), "login", "user", "failure",
				zap.String("email", req.Email),
				zap.String("reason", "user not found"),
			)
			if errors.Is(err, gorm.ErrRecordNotFound) {
				writeError(w, http.StatusUnauthorized, r, "UNAUTHORIZED", "Invalid email or password", nil)
				return
			}
			writeError(w, http.StatusInternalServerError, r, "INTERNAL_ERROR", "Failed to process request", nil)
			return
		}

		if err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(req.Password)); err != nil {
			audit.LogEvent(r.Context(), "login", "user", "failure",
				zap.String("email", req.Email),
				zap.String("reason", "invalid password"),
			)
			writeError(w, http.StatusUnauthorized, r, "UNAUTHORIZED", "Invalid email or password", nil)
			return
		}

		if !user.IsActive {
			audit.LogEvent(r.Context(), "login", "user", "failure",
				zap.String("email", req.Email),
				zap.String("reason", "account inactive"),
			)
			writeError(w, http.StatusForbidden, r, "FORBIDDEN", "Account is inactive", nil)
			return
		}

		accessToken, refreshToken, err := generateTokenPair(user.ID)
		if err != nil {
			audit.LogEvent(r.Context(), "login", "user", "failure",
				zap.String("email", req.Email),
				zap.String("reason", "token_generation_failed"),
			)
			writeError(w, http.StatusInternalServerError, r, "INTERNAL_ERROR", "Failed to generate tokens", nil)
			return
		}

		audit.LogEvent(r.Context(), "login", "user", "success",
			zap.String("email", req.Email),
			zap.Uint("user_id", user.ID),
		)
		writeSuccess(w, http.StatusOK, r, authResponse{
			User:         userPayload{ID: user.ID, Email: user.Email, UserName: user.UserName, IsActive: user.IsActive},
			AccessToken:  accessToken,
			RefreshToken: refreshToken,
			ExpiresIn:    900,
		})
	}
}
