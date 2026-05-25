// internal/middleware/audit.go
package middleware

import (
	"bufio"
	"bytes"
	"fmt"
	"net"
	"net/http"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/auth"
	"go.uber.org/zap"
)

// AuditMiddleware logs every HTTP request with details.
func AuditMiddleware(logger *zap.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()

			// Capture response status and size using a custom writer
			rec := &responseRecorder{ResponseWriter: w, statusCode: http.StatusOK, body: &bytes.Buffer{}}
			next.ServeHTTP(rec, r)

			duration := time.Since(start)

			// Get user ID from context (if authenticated)
			userID, _ := auth.GetUserIDFromContext(r.Context())

			// Build log fields
			fields := []zap.Field{
				zap.String("method", r.Method),
				zap.String("path", r.URL.Path),
				zap.Int("status", rec.statusCode),
				zap.Duration("duration_ms", duration),
				zap.String("ip", r.RemoteAddr),
				zap.String("user_agent", r.UserAgent()),
				zap.String("request_id", RequestIDFromContext(r.Context())),
			}
			if userID != 0 {
				fields = append(fields, zap.Uint64("user_id", userID))
			}
			if rec.body.Len() > 0 && rec.statusCode >= 400 {
				// Log response body only for errors to avoid clutter
				fields = append(fields, zap.String("response_body", rec.body.String()))
			}

			logger.Info("HTTP request", fields...)
		})
	}
}

// responseRecorder wraps http.ResponseWriter to capture status code and response body.
type responseRecorder struct {
	http.ResponseWriter
	statusCode int
	body       *bytes.Buffer
}

func (rec *responseRecorder) WriteHeader(code int) {
	rec.statusCode = code
	rec.ResponseWriter.WriteHeader(code)
}

func (rec *responseRecorder) Write(b []byte) (int, error) {
	rec.body.Write(b) // capture for logging
	return rec.ResponseWriter.Write(b)
}

// Hijack implements http.Hijacker to support WebSocket upgrades.
func (rec *responseRecorder) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	if hj, ok := rec.ResponseWriter.(http.Hijacker); ok {
		return hj.Hijack()
	}
	return nil, nil, fmt.Errorf("underlying ResponseWriter does not implement http.Hijacker")
}
