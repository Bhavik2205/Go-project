package auth

import (
	"errors"
	"net/http"

	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/validation"
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
			return
		}

		var user db.User
		if err := dbClient.DB.Where("email = ?", req.Email).First(&user).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				writeError(w, http.StatusUnauthorized, r, "UNAUTHORIZED", "Invalid email or password", nil)
				return
			}
			writeError(w, http.StatusInternalServerError, r, "INTERNAL_ERROR", "Failed to process request", nil)
			return
		}

		if err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(req.Password)); err != nil {
			writeError(w, http.StatusUnauthorized, r, "UNAUTHORIZED", "Invalid email or password", nil)
			return
		}

		if !user.IsActive {
			writeError(w, http.StatusForbidden, r, "FORBIDDEN", "Account is inactive", nil)
			return
		}

		accessToken, refreshToken, err := generateTokenPair(user.ID)
		if err != nil {
			writeError(w, http.StatusInternalServerError, r, "INTERNAL_ERROR", "Failed to generate tokens", nil)
			return
		}

		writeSuccess(w, http.StatusOK, r, authResponse{
			User:         userPayload{ID: user.ID, Email: user.Email, UserName: user.UserName, IsActive: user.IsActive},
			AccessToken:  accessToken,
			RefreshToken: refreshToken,
			ExpiresIn:    900,
		})
	}
}
