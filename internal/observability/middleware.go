package observability

import (
	"fmt"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	httpRequestDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "trading_http_request_duration_ms",
			Help:    "HTTP request duration in milliseconds",
			Buckets: []float64{1, 5, 10, 25, 50, 100, 250, 500, 1000},
		},
		[]string{"method", "path", "status"},
	)

	httpRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "trading_http_requests_total",
			Help: "Total HTTP requests",
		},
		[]string{"method", "path", "status"},
	)
)

func init() {
	// Register HTTP metrics on import
	prometheus.Register(httpRequestDuration)
	prometheus.Register(httpRequestsTotal)
}

// HTTPMiddleware wraps an HTTP handler with metrics collection.
func HTTPMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// Wrap ResponseWriter to capture status code
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		next.ServeHTTP(wrapped, r)

		duration := time.Since(start)
		statusStr := fmt.Sprintf("%d", wrapped.statusCode)

		httpRequestDuration.WithLabelValues(r.Method, r.URL.Path, statusStr).Observe(duration.Seconds() * 1000)
		httpRequestsTotal.WithLabelValues(r.Method, r.URL.Path, statusStr).Inc()
	})
}

type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}
