package middleware

import (
	"net/http"
)

// SecurityHeaders adds security-related HTTP headers to every response.
// This helps prevent XSS, MIME sniffing, and enforces HTTPS.
func SecurityHeaders() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Prevent MIME type sniffing
			w.Header().Set("X-Content-Type-Options", "nosniff")

			// Prevent clickjacking
			w.Header().Set("X-Frame-Options", "DENY")

			// Enforce HTTPS (HSTS) – only set if request is already HTTPS
			// In production, you would set this only when TLS is active.
			// For local dev, you can skip or set low max-age.
			if r.TLS != nil || r.Header.Get("X-Forwarded-Proto") == "https" {
				w.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload")
			}

			// Optional but recommended: X-XSS-Protection (though deprecated in modern browsers, still harmless)
			w.Header().Set("X-XSS-Protection", "1; mode=block")

			// Optional: Content-Security-Policy – could be added later
			// w.Header().Set("Content-Security-Policy", "default-src 'self'")

			next.ServeHTTP(w, r)
		})
	}
}
