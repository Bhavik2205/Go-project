// internal/httpapi/routes.go
package httpapi

import (
	"net/http"
	_ "net/http/pprof"

	authhandler "github.com/Bhavik2205/ML-Bot/internal/api/handlers/auth"
	profilehandler "github.com/Bhavik2205/ML-Bot/internal/api/handlers/profile"
	runtimehandler "github.com/Bhavik2205/ML-Bot/internal/api/handlers/runtime"
	"github.com/Bhavik2205/ML-Bot/internal/api/handlers/settings"
	"github.com/Bhavik2205/ML-Bot/internal/middleware"
	"github.com/gorilla/mux"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// RegisterRoutes sets up all API v1 routes.
func RegisterRoutes(router *mux.Router, deps HTTPDeps) {
	apiV1 := router.PathPrefix("/api/v1").Subrouter()

	// Public routes
	apiV1.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		handleV1Health(w, r, deps)
	}).Methods("GET")
	apiV1.Handle("/metrics", promhttp.Handler()).Methods("GET")
	
	// pprof profiling (for memory/goroutine leak detection)
	// Available at: /debug/pprof
	router.PathPrefix("/debug/pprof").Handler(http.DefaultServeMux)
	
	apiV1.HandleFunc("/openapi.json", handleV1OpenAPISpec).Methods("GET")
	apiV1.HandleFunc("/auth/signup", authhandler.HandleSignup(deps.DBClient)).Methods("POST")
	apiV1.HandleFunc("/auth/login", authhandler.HandleLogin(deps.DBClient)).Methods("POST")
	apiV1.HandleFunc("/auth/refresh", authhandler.HandleRefresh(deps.RedisClient)).Methods("POST")
	apiV1.HandleFunc("/auth/logout", authhandler.HandleLogout(deps.RedisClient)).Methods("POST")

	// Test endpoint (does not parse body)
	apiV1.HandleFunc("/test/echo", func(w http.ResponseWriter, r *http.Request) {
		// _, _ = io.Copy(io.Discard, r.Body) // you can keep the body discarding if needed
		w.WriteHeader(http.StatusOK)
	}).Methods("POST")

	// Protected routes
	protected := apiV1.NewRoute().Subrouter()
	protected.Use(middleware.Authenticate(deps.RedisClient))

	protected.HandleFunc("/me", profilehandler.HandleGetMe(deps.DBClient)).Methods("GET")
	protected.HandleFunc("/me", profilehandler.HandlePatchMe(deps.DBClient)).Methods("PATCH")
	protected.HandleFunc("/settings", settings.HandleGetSettings(deps.DBClient)).Methods("GET")
	protected.HandleFunc("/settings", settings.HandleUpdateSettings(deps.DBClient)).Methods("PUT")
	protected.HandleFunc("/runtime/config", runtimehandler.HandleRuntimeConfig(deps.AppConfig)).Methods("GET")
	protected.HandleFunc("/runtime/metrics", runtimehandler.HandleRuntimeMetrics(deps.Hub, deps.DBClient, deps.StartupTime)).Methods("GET")
	protected.HandleFunc("/brokers/zerodha/status", func(w http.ResponseWriter, r *http.Request) {
		handleV1BrokerStatus(w, r, deps)
	}).Methods("GET")
	protected.HandleFunc("/quotes", func(w http.ResponseWriter, r *http.Request) {
		handleV1Quotes(w, r, deps)
	}).Methods("GET")
	protected.HandleFunc("/market/overview", func(w http.ResponseWriter, r *http.Request) {
		handleV1MarketOverview(w, r, deps)
	}).Methods("GET")
}
