package auth

import (
	"net/http"

	"github.com/Bhavik2205/ML-Bot/internal/audit"
	"github.com/Bhavik2205/ML-Bot/internal/auth"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/validation"
	"go.uber.org/zap"
	"golang.org/x/crypto/bcrypt"
)

type signupRequest struct {
	Email    string `json:"email"    validate:"required,email"`
	Password string `json:"password" validate:"required,min=8"`
	UserName string `json:"userName"`
}

type authResponse struct {
	User         userPayload `json:"user"`
	AccessToken  string      `json:"accessToken"`
	RefreshToken string      `json:"refreshToken"`
	ExpiresIn    int         `json:"expiresIn"`
}

type userPayload struct {
	ID       uint   `json:"id"`
	Email    string `json:"email"`
	UserName string `json:"userName"`
	IsActive bool   `json:"isActive"`
}

func HandleSignup(dbClient *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req signupRequest
		if !validation.BindAndValidate(w, r, &req) {
			audit.LogEvent(r.Context(), "signup", "user", "failure",
				zap.String("email", req.Email),
				zap.String("reason", "validation_failed"),
			)
			return
		}

		hash, err := bcrypt.GenerateFromPassword([]byte(req.Password), 12)
		if err != nil {
			audit.LogEvent(r.Context(), "signup", "user", "failure",
				zap.String("email", req.Email),
				zap.String("reason", "password_hashing_failed"),
			)
			writeError(w, http.StatusInternalServerError, r, "INTERNAL_ERROR", "Failed to process request", nil)
			return
		}

		user := db.User{
			Email:        req.Email,
			PasswordHash: string(hash),
			UserName:     req.UserName,
			IsActive:     true,
		}
		if err := dbClient.DB.Create(&user).Error; err != nil {
			audit.LogEvent(r.Context(), "signup", "user", "failure",
				zap.String("email", req.Email),
				zap.String("reason", "email_already_registered"),
			)
			writeError(w, http.StatusConflict, r, "CONFLICT", "Email already registered", nil)
			return
		}

		accessToken, refreshToken, err := generateTokenPair(user.ID)
		if err != nil {
			audit.LogEvent(r.Context(), "signup", "user", "failure",
				zap.String("email", req.Email),
				zap.String("reason", "token_generation_failed"),
			)
			writeError(w, http.StatusInternalServerError, r, "INTERNAL_ERROR", "Failed to generate tokens", nil)
			return
		}

		audit.LogEvent(r.Context(), "signup", "user", "success",
			zap.String("email", req.Email),
			zap.String("user_name", req.UserName),
			zap.Uint("user_id", user.ID),
		)
		writeSuccess(w, http.StatusCreated, r, authResponse{
			User:         userPayload{ID: user.ID, Email: user.Email, UserName: user.UserName, IsActive: user.IsActive},
			AccessToken:  accessToken,
			RefreshToken: refreshToken,
			ExpiresIn:    900,
		})
	}
}

func generateTokenPair(userID uint) (string, string, error) {
	access, err := auth.GenerateAccessToken(userID)
	if err != nil {
		return "", "", err
	}
	refresh, err := auth.GenerateRefreshToken(userID)
	if err != nil {
		return "", "", err
	}
	return access, refresh, nil
}
