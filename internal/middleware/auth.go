package middleware

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/Bhavik2205/ML-Bot/internal/auth"
	"github.com/Bhavik2205/ML-Bot/internal/cache"
)

type contextKey string

const UserIDKey contextKey = "userID"

// Authenticate is an HTTP middleware that validates the Bearer JWT on every
// request. It rejects with 401 if the token is missing, invalid, expired, or
// present in the Redis refresh-token blocklist.
// On success it stores the userID in the request context under UserIDKey.
func Authenticate(redisClient *cache.RedisClient) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			raw := r.Header.Get("Authorization")
			if !strings.HasPrefix(raw, "Bearer ") {
				writeUnauthorized(w, r, "missing or malformed Authorization header")
				return
			}
			tokenStr := strings.TrimPrefix(raw, "Bearer ")

			claims, err := auth.ParseToken(tokenStr, auth.TokenTypeAccess)
			if err != nil {
				writeUnauthorized(w, r, "invalid or expired token")
				return
			}

			// Check Redis blocklist for revoked access tokens (written by logout handler)
			if redisClient != nil {
				key := "blocklist:access:" + tokenStr
				if val, _ := redisClient.Get(key); val != "" {
					writeUnauthorized(w, r, "token has been revoked")
					return
				}
			}

			ctx := context.WithValue(r.Context(), UserIDKey, claims.UserID)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// UserIDFromContext extracts the authenticated userID from the request context.
// Returns 0 if not present.
func UserIDFromContext(ctx context.Context) uint {
	v, _ := ctx.Value(UserIDKey).(uint)
	return v
}

func writeUnauthorized(w http.ResponseWriter, r *http.Request, message string) {
	requestID := r.Header.Get("X-Request-ID")
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusUnauthorized)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"code":    "UNAUTHORIZED",
			"message": message,
		},
		"meta": map[string]any{
			"requestId": requestID,
			"version":   "v1",
		},
	})
}
