package auth

import (
	"net/http"

	"github.com/Bhavik2205/ML-Bot/internal/auth"
	"github.com/Bhavik2205/ML-Bot/internal/cache"
	"github.com/Bhavik2205/ML-Bot/internal/validation"
)

type refreshRequest struct {
	RefreshToken string `json:"refreshToken" validate:"required"`
}

type refreshResponse struct {
	AccessToken  string `json:"accessToken"`
	RefreshToken string `json:"refreshToken"`
	ExpiresIn    int    `json:"expiresIn"`
}

// HandleRefresh now accepts redisClient to check blocklist
func HandleRefresh(redisClient *cache.RedisClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req refreshRequest
		if !validation.BindAndValidate(w, r, &req) {
			return
		}

		// 1. Parse and validate the refresh token
		claims, err := auth.ParseToken(req.RefreshToken, auth.TokenTypeRefresh)
		if err != nil {
			writeError(w, http.StatusUnauthorized, r, "UNAUTHORIZED", "Invalid or expired refresh token", nil)
			return
		}

		// 2. Check if the token is blocklisted in Redis (AUD-002)
		if redisClient != nil {
			val, err := redisClient.Get("blocklist:refresh:" + req.RefreshToken)
			if err == nil && val == "1" {
				writeError(w, http.StatusUnauthorized, r, "TOKEN_REVOKED", "Refresh token has been revoked", nil)
				return
			}
		}

		// 3. Generate new token pair (old refresh token is rotated)
		accessToken, refreshToken, err := generateTokenPair(claims.UserID)
		if err != nil {
			writeError(w, http.StatusInternalServerError, r, "INTERNAL_ERROR", "Failed to generate tokens", nil)
			return
		}

		writeSuccess(w, http.StatusOK, r, refreshResponse{
			AccessToken:  accessToken,
			RefreshToken: refreshToken,
			ExpiresIn:    900,
		})
	}
}
