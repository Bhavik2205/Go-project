package middleware

import (
	"bytes"
	"io"
	"log"
	"net/http"
)

func MaxBytesMiddleware(maxBytes int64) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			log.Printf("MaxBytesMiddleware: limiting body to %d bytes for %s %s", maxBytes, r.Method, r.URL.Path)

			// Read at most maxBytes+1 bytes to detect overflow
			limitedReader := io.LimitReader(r.Body, maxBytes+1)
			body, err := io.ReadAll(limitedReader)
			if err != nil {
				http.Error(w, "Error reading request body", http.StatusInternalServerError)
				return
			}

			// If we read more than maxBytes, reject
			if int64(len(body)) > maxBytes {
				http.Error(w, "Request body too large", http.StatusRequestEntityTooLarge)
				return
			}

			// Replace the body with a fresh reader for the handler
			r.Body = io.NopCloser(bytes.NewReader(body))

			next.ServeHTTP(w, r)
		})
	}
}
