package auth

import (
	"net/http"
	"strings"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/auth"
	"github.com/Bhavik2205/ML-Bot/internal/cache"
	"github.com/Bhavik2205/ML-Bot/internal/validation"
)

type logoutRequest struct {
	RefreshToken string `json:"refreshToken" validate:"required"`
}

func HandleLogout(redisClient *cache.RedisClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req logoutRequest
		if !validation.BindAndValidate(w, r, &req) {
			return
		}

		// Blocklist the refresh token
		if redisClient != nil {
			if claims, err := auth.ParseToken(req.RefreshToken, auth.TokenTypeRefresh); err == nil {
				if ttl := time.Until(claims.ExpiresAt.Time); ttl > 0 {
					_ = redisClient.Set("blocklist:refresh:"+req.RefreshToken, "1", ttl)
				}
			}
		}

		// Blocklist the access token from the Authorization header
		if redisClient != nil {
			if raw := r.Header.Get("Authorization"); strings.HasPrefix(raw, "Bearer ") {
				accessToken := strings.TrimPrefix(raw, "Bearer ")
				if claims, err := auth.ParseToken(accessToken, auth.TokenTypeAccess); err == nil {
					if ttl := time.Until(claims.ExpiresAt.Time); ttl > 0 {
						_ = redisClient.Set("blocklist:access:"+accessToken, "1", ttl)
					}
				}
			}
		}

		w.WriteHeader(http.StatusNoContent)
	}
}
