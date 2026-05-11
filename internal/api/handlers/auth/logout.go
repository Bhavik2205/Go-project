package auth

import (
	"net/http"
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

		claims, err := auth.ParseToken(req.RefreshToken, auth.TokenTypeRefresh)
		if err != nil {
			// Token is already invalid — treat as successful logout
			w.WriteHeader(http.StatusNoContent)
			return
		}

		ttl := time.Until(claims.ExpiresAt.Time)
		if ttl > 0 {
			key := "blocklist:refresh:" + req.RefreshToken
			_ = redisClient.Set(key, "1", ttl)
		}

		w.WriteHeader(http.StatusNoContent)
	}
}
