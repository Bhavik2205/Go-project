package server

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/api/handlers/stockHandler"
	"github.com/Bhavik2205/ML-Bot/internal/cache"
	"github.com/Bhavik2205/ML-Bot/internal/data"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/middleware"
	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

// ZerodhaAPI interface to abstract Zerodha client methods used by handlers.
type ZerodhaAPI interface {
	FindInstrumentToken(symbol string, exchanges []string) (*api.InstrumentInfo, error)
}

var (
	zerodhaClient      ZerodhaAPI // Use the interface type
	dbClient           *db.DBClient
	redisClient        *cache.RedisClient
	ingestor           *data.MarketDataIngestor // New global variable for the ingestor
	wsClients          *sync.Map                // Shared sync.Map for WebSocket clients (for ticks)
	candleWsClients    *sync.Map                // Separate map for candle WebSocket clients
	indicatorWsClients *sync.Map                // Shared sync.Map for indicator/candle WebSocket clients

	candleGenerator *data.CandleGenerator // <-- ADDED: CandleGenerator for candle WebSocket streaming

	upgrader = websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool { return true }, // Allow all origins for simplicity, tighten in prod
	}
)

// SetZerodhaClient sets the Zerodha API client
func SetZerodhaClient(client ZerodhaAPI) { // Accept interface
	zerodhaClient = client
}

// SetDBClient sets the database client
func SetDBClient(client *db.DBClient) {
	dbClient = client
}

// SetRedisClient sets the Redis client
func SetRedisClient(client *cache.RedisClient) {
	redisClient = client
}

// SetIngestor sets the market data ingestor and shares the WebSocket clients map.
func SetIngestor(i *data.MarketDataIngestor, clients *sync.Map) {
	ingestor = i
	wsClients = clients // Assign the shared map for ticks
}

// SetCandleClients injects the shared WebSocket client map for candle data.
func SetCandleClients(clients *sync.Map) {
	candleWsClients = clients // Assign the shared map for candles
}

func SetIndicatorClients(clients *sync.Map) {
	indicatorWsClients = clients
}

// ADDED: Setter for CandleGenerator
func SetCandleGenerator(gen *data.CandleGenerator) {
	candleGenerator = gen
}

// StartHTTPServer starts the HTTP and WebSocket server
func StartHTTPServer(port int) {
	router := mux.NewRouter()

	router.Use(enableCORS)
	router.Use(recoverMiddleware)
	router.Use(middleware.RequestID)
	router.Use(middleware.Logger)
	// Handle OPTIONS preflight for all routes so CORS middleware fires correctly.
	router.Methods(http.MethodOptions).HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	registerVersionedRoutes(router)

	if zc, ok := zerodhaClient.(*api.ZerodhaClient); ok && zc != nil {
		router.HandleFunc("/api/instrument", stockHandler.HandleInstrumentLookup(zc)).Methods("GET")
	}



	// Existing WebSocket endpoint for tick data
	router.HandleFunc("/ws", handleConnections)
	// NEW: WebSocket endpoint for candle data
	router.HandleFunc("/ws/candles", handleCandleConnections)

	// NEW: WebSocket endpoint for real-time indicator updates
	router.HandleFunc("/ws/indicators", handleIndicatorConnections)

	//NEW: Heatmap websocket endpoint
	router.HandleFunc("/ws/heatmap", HeatmapWebSocketHandler(data.GetMarketHeatmap()))

	srv := &http.Server{
		Addr:         ":" + strconv.Itoa(port),
		Handler:      router,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	zap.L().Info("🌐 Unified HTTP + WebSocket server starting...", zap.Int("port", port))

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			zap.L().Fatal("HTTP server error", zap.Error(err))
		}
	}()

	<-quit
	zap.L().Info("HTTP server shutting down gracefully...")
	shutCtx, shutCancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer shutCancel()
	if err := srv.Shutdown(shutCtx); err != nil {
		zap.L().Error("HTTP server forced shutdown", zap.Error(err))
	}
}

// handleConnections upgrades HTTP connection to WebSocket and registers the client with the ingestor (for ticks).
func handleConnections(w http.ResponseWriter, r *http.Request) {
	defer func() {
		if r := recover(); r != nil {
			zap.L().Error("Panic in WebSocket handler (ticks)", zap.Any("recover", r))
		}
	}()

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		zap.L().Error("WebSocket upgrade error for ticks", zap.Error(err))
		return
	}
	// Important: Defer Unregister for proper cleanup when connection closes
	defer ingestor.UnregisterWebSocketClient(conn)

	ingestor.RegisterWebSocketClient(conn) // Register the new client for ticks

	// Keep the connection alive, listen for close messages from the client.
	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				zap.L().Debug("WebSocket unexpected close detected for tick client", zap.Error(err), zap.String("remote_addr", conn.RemoteAddr().String()))
			} else {
				zap.L().Info("WebSocket tick client disconnected", zap.String("remote_addr", conn.RemoteAddr().String()))
			}
			break // Exit the loop, triggering the defer
		}
	}
}
func handleIndicatorConnections(w http.ResponseWriter, r *http.Request) {
	defer func() {
		if r := recover(); r != nil {
			zap.L().Error("Panic in WebSocket handler (indicators)", zap.Any("recover", r))
		}
	}()

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		zap.L().Error("WebSocket upgrade error for indicators", zap.Error(err))
		return
	}
	// Use a unique key for the sync.Map (e.g., connection remote address)
	clientKey := conn.RemoteAddr().String()
	indicatorWsClients.Store(clientKey, conn) // Register the new client for indicators

	zap.L().Info("New WebSocket client connected for indicator data", zap.String("remote_addr", clientKey))

	// Keep the connection alive, listen for close messages from the client.
	for {
		// Read message to detect client disconnects or pings/pongs
		_, _, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				zap.L().Debug("WebSocket unexpected close for indicator client", zap.Error(err), zap.String("remote_addr", clientKey))
			} else {
				zap.L().Info("WebSocket indicator client disconnected", zap.String("remote_addr", clientKey))
			}
			indicatorWsClients.Delete(clientKey) // Unregister the client on disconnect
			break                                // Exit the loop
		}
	}
}

// UPDATED: handleCandleConnections now uses CandleGenerator for registration and streaming
func handleCandleConnections(w http.ResponseWriter, r *http.Request) {
	defer func() {
		if r := recover(); r != nil {
			zap.L().Error("Panic in WebSocket handler (candles)", zap.Any("recover", r))
		}
	}()

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		zap.L().Error("WebSocket upgrade error for candles", zap.Error(err))
		return
	}
	if candleGenerator == nil {
		zap.L().Error("CandleGenerator is not initialized")
		conn.Close()
		return
	}
	defer candleGenerator.UnregisterWebSocketClient(conn)
	candleGenerator.RegisterWebSocketClient(conn)

	zap.L().Info("New WebSocket client connected for candle data", zap.String("remote_addr", conn.RemoteAddr().String()))

	// Keep the connection alive, listen for close messages from the client.
	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				zap.L().Debug("WebSocket unexpected close for candle client", zap.Error(err), zap.String("remote_addr", conn.RemoteAddr().String()))
			} else {
				zap.L().Info("WebSocket candle client disconnected", zap.String("remote_addr", conn.RemoteAddr().String()))
			}
			break // Exit the loop, triggering the defer
		}
	}
}

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
		// Default: allow localhost dev origins only
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
			if r := recover(); r != nil {
				zap.L().Error("Panic in HTTP handler", zap.Any("recover", r))
				http.Error(w, "Internal server error", http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
}
