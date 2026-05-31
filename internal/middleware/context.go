package middleware

import (
	"context"
	"net"
	"net/http"
	"strings"
)

const (
	ipKey        contextKey = "ip"
	userAgentKey contextKey = "user_agent"
)

// AddRequestInfoToContext stores the client IP and User-Agent in the request context.
// It should be placed BEFORE the audit middleware.
func AddRequestInfoToContext(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ip := getClientIP(r)
		ua := r.UserAgent()
		ctx := context.WithValue(r.Context(), ipKey, ip)
		ctx = context.WithValue(ctx, userAgentKey, ua)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func getClientIP(r *http.Request) string {
	// Try X-Forwarded-For (for proxies)
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		ips := strings.Split(xff, ",")
		return strings.TrimSpace(ips[0])
	}
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return xri
	}
	// For RemoteAddr, use net.SplitHostPort to correctly handle IPv6 addresses.
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		// No port – return as is (e.g., when behind a unix socket)
		return r.RemoteAddr
	}
	return host
}

func IPFromContext(ctx context.Context) string {
	if v := ctx.Value(ipKey); v != nil {
		return v.(string)
	}
	return ""
}

func UserAgentFromContext(ctx context.Context) string {
	if v := ctx.Value(userAgentKey); v != nil {
		return v.(string)
	}
	return ""
}
