package middleware

import (
	"context"
	"fmt"
	"net/http"
	"sync/atomic"
	"time"
)

type requestIDKey struct{}

var requestIDCounter uint64

// RequestID injects a request ID into every request context and response header.
func RequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-ID")
		if id == "" {
			seq := atomic.AddUint64(&requestIDCounter, 1)
			id = fmt.Sprintf("req_%d_%d", time.Now().UnixNano(), seq)
		}
		w.Header().Set("X-Request-ID", id)
		ctx := context.WithValue(r.Context(), requestIDKey{}, id)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// RequestIDFromContext returns the request ID stored in ctx, or empty string.
func RequestIDFromContext(ctx context.Context) string {
	v, _ := ctx.Value(requestIDKey{}).(string)
	return v
}
