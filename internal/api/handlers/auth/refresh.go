package auth

import (
	"net/http"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/audit"
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

func HandleRefresh(redisClient *cache.RedisClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req refreshRequest
		if !validation.BindAndValidate(w, r, &req) {
			audit.LogEvent(r.Context(),
				"REFRESH",
				"token",
				"",
				"UPDATE",
				"FAILURE",
				map[string]any{
					"reason": "validation failed",
				},
				"validation failed",
			)
			return
		}

		// Parse and validate the refresh token
		claims, err := auth.ParseToken(req.RefreshToken, auth.TokenTypeRefresh)
		if err != nil {
			audit.LogEvent(r.Context(),
				"REFRESH",
				"token",
				"",
				"UPDATE",
				"FAILURE",
				map[string]any{
					"reason": "invalid token",
				},
				"invalid token",
			)
			writeError(w, http.StatusUnauthorized, r, "UNAUTHORIZED", "Invalid or expired refresh token", nil)
			return
		}

		// Check if the token is already blocklisted (e.g., from previous use or logout)
		if redisClient != nil {
			if val, _ := redisClient.Get("blocklist:refresh:" + req.RefreshToken); val == "1" {
				audit.LogEvent(r.Context(),
					"REFRESH",
					"token",
					"",        // resourceID not applicable (could be the refresh token hash, but omit for security)
					"REFRESH", // action (or "READ")
					"FAILURE",
					map[string]any{
						"reason":  "token already used (reuse detected)",
						"user_id": claims.UserID,
					},
					"token already used (reuse detected)",
				)
				writeError(w, http.StatusUnauthorized, r, "TOKEN_REVOKED", "Refresh token has been revoked", nil)
				return
			}
		}

		// --- ONE‑TIME USE: Blocklist this refresh token immediately ---
		if redisClient != nil {
			ttl := time.Until(claims.ExpiresAt.Time)
			if ttl > 0 {
				_ = redisClient.Set("blocklist:refresh:"+req.RefreshToken, "1", ttl)
			}
		}

		// Generate new token pair
		accessToken, refreshToken, err := generateTokenPair(claims.UserID)
		if err != nil {
			audit.LogEvent(r.Context(),
				"REFRESH",
				"token",
				"",
				"UPDATE",
				"FAILURE",
				map[string]any{
					"reason":  "token generation failed",
					"user_id": claims.UserID,
				},
				"token generation failed",
			)
			writeError(w, http.StatusInternalServerError, r, "INTERNAL_ERROR", "Failed to generate tokens", nil)
			return
		}

		audit.LogEvent(r.Context(),
			"REFRESH",
			"token",
			"",
			"UPDATE",
			"SUCCESS",
			map[string]any{
				"user_id": claims.UserID,
			},
			"token generation successful",
		)
		writeSuccess(w, http.StatusOK, r, refreshResponse{
			AccessToken:  accessToken,
			RefreshToken: refreshToken,
			ExpiresIn:    900,
		})
	}
}
