// internal/httpapi/server.go
package httpapi

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/api/handlers/stockHandler"
	"github.com/Bhavik2205/ML-Bot/internal/middleware"
	"github.com/gorilla/mux"
	"go.uber.org/zap"
)

// Server represents the REST+WebSocket HTTP server (WebSockets are delegated to realtime.Hub).
type Server struct {
	deps       HTTPDeps
	httpServer *http.Server
}

// NewServer creates a new server instance.
func NewServer(deps HTTPDeps) *Server {
	return &Server{deps: deps}
}

// Start starts the HTTP server.
func (s *Server) Start(ctx context.Context, port int) error {
	router := mux.NewRouter()

	// Apply middlewares (same as before)
	maxBytes := int64(1 << 20)
	if s.deps.AppConfig != nil && s.deps.AppConfig.Server.MaxRequestBodyBytes > 0 {
		maxBytes = int64(s.deps.AppConfig.Server.MaxRequestBodyBytes)
	}
	// Rate limiting: 10 requests/sec, burst 20
	rateLimiter := middleware.NewRateLimiter(10, 20)

	router.Use(rateLimiter.Middleware)
	router.Use(middleware.MaxBytesMiddleware(maxBytes))
	router.Use(middleware.SecurityHeaders())
	router.Use(enableCORS)
	router.Use(recoverMiddleware)
	router.Use(middleware.AddRequestInfoToContext)
	router.Use(middleware.RequestID)
	router.Use(middleware.AuditMiddleware(zap.L()))
	router.Use(middleware.Logger)

	// Register REST routes
	RegisterRoutes(router, s.deps)

	// Register WebSocket routes via the hub
	if s.deps.Hub != nil {
		router.HandleFunc("/ws", s.deps.Hub.ServeTicks)
		router.HandleFunc("/ws/candles", s.deps.Hub.ServeCandles)
		router.HandleFunc("/ws/indicators", s.deps.Hub.ServeIndicators)
		router.HandleFunc("/ws/heatmap", s.deps.Hub.ServeHeatmap)
	}

	// Instrument lookup endpoint (if Zerodha client is available)
	if zc, ok := s.deps.ZerodhaClient.(interface{ GetKite() interface{} }); ok && zc != nil {
		// The actual endpoint expects a *api.ZerodhaClient; we can type-assert safely.
		// Simpler: we can pass the whole ZerodhaClient to the handler.
		// For now, we'll check the concrete type.
		if _, ok := s.deps.ZerodhaClient.(interface {
			FindInstrumentToken(string, []string) (*api.InstrumentInfo, error)
		}); ok {
			// This is a bit messy; the handler expects *api.ZerodhaClient.
			// We'll keep the old route only if the client is exactly *api.ZerodhaClient.
			if zc, ok2 := s.deps.ZerodhaClient.(*api.ZerodhaClient); ok2 && zc != nil {
				router.HandleFunc("/api/instrument", stockHandler.HandleInstrumentLookup(zc)).Methods("GET")
			}
		}
	}

	s.httpServer = &http.Server{
		Addr:         ":" + fmt.Sprint(port),
		Handler:      router,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	zap.L().Info("🌐 HTTP + WebSocket server starting...", zap.Int("port", port))

	serveErr := make(chan error, 1)
	go func() {
		if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			serveErr <- err
		}
		close(serveErr)
	}()

	select {
	case err := <-serveErr:
		return fmt.Errorf("HTTP server failed: %w", err)
	case <-ctx.Done():
		zap.L().Info("HTTP server shutting down gracefully...")
		shutCtx, shutCancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer shutCancel()
		return s.httpServer.Shutdown(shutCtx)
	}
}

// Stop gracefully shuts down the HTTP server.
func (s *Server) Stop(ctx context.Context) error {
	if s.httpServer != nil {
		return s.httpServer.Shutdown(ctx)
	}
	return nil
}

// ---- Middleware helpers (copied from internal/server/routes.go) ----

func enableCORS(h http.Handler) http.Handler {
	allowedOrigins := corsAllowedOrigins()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin != "" && allowedOrigins[origin] {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Vary", "Origin")
		}
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Request-ID")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		h.ServeHTTP(w, r)
	})
}

func corsAllowedOrigins() map[string]bool {
	raw := os.Getenv("ALLOWED_ORIGINS")
	if raw == "" {
		return map[string]bool{
			"http://localhost:3000": true,
			"http://localhost:5173": true,
			"http://localhost:8080": true,
		}
	}
	result := map[string]bool{}
	for _, o := range strings.Split(raw, ",") {
		if o = strings.TrimSpace(o); o != "" {
			result[o] = true
		}
	}
	return result
}

func recoverMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				zap.L().Error("Panic in HTTP handler", zap.Any("recover", rec))
				http.Error(w, "Internal server error", http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
}
