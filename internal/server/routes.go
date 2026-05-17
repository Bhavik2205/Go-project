package server

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
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

const (
	wsPongWait   = 60 * time.Second
	wsPingPeriod = 45 * time.Second // must be < pongWait
	wsWriteWait  = 10 * time.Second
	wsMaxMsgSize = 512 * 1024 // 512 KB
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

	candleGenerator  *data.CandleGenerator  // CandleGenerator for candle WebSocket streaming
	indicatorManager *data.IndicatorManager // IndicatorManager for indicator WebSocket streaming

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

// SetIndicatorManager injects the IndicatorManager for WebSocket client registration.
func SetIndicatorManager(im *data.IndicatorManager) {
	indicatorManager = im
}

// ADDED: Setter for CandleGenerator
func SetCandleGenerator(gen *data.CandleGenerator) {
	candleGenerator = gen
}

// StartHTTPServer starts the HTTP and WebSocket server. It blocks until ctx is
// cancelled or a fatal bind error occurs, then shuts down gracefully.
func StartHTTPServer(ctx context.Context, port int) error {
	router := mux.NewRouter()
	maxBytes := int64(1 << 20) // default

	if appConfig != nil && appConfig.Server.MaxRequestBodyBytes > 0 {
		maxBytes = int64(appConfig.Server.MaxRequestBodyBytes)
	}
	router.Use(middleware.MaxBytesMiddleware(maxBytes))
	router.Use(middleware.SecurityHeaders())
	router.Use(enableCORS)
	router.Use(recoverMiddleware)
	router.Use(middleware.RequestID)
	router.Use(middleware.Logger)
	router.Methods(http.MethodOptions).HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	registerVersionedRoutes(router)

	if zc, ok := zerodhaClient.(*api.ZerodhaClient); ok && zc != nil {
		router.HandleFunc("/api/instrument", stockHandler.HandleInstrumentLookup(zc)).Methods("GET")
	}

	router.HandleFunc("/ws", handleConnections)
	router.HandleFunc("/ws/candles", handleCandleConnections)
	router.HandleFunc("/ws/indicators", handleIndicatorConnections)
	router.HandleFunc("/ws/heatmap", HeatmapWebSocketHandler(data.GetMarketHeatmap()))

	srv := &http.Server{
		Addr:         ":" + strconv.Itoa(port),
		Handler:      router,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	zap.L().Info("🌐 HTTP + WebSocket server starting...", zap.Int("port", port))

	serveErr := make(chan error, 1)
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			serveErr <- err
		}
		close(serveErr)
	}()

	select {
	case err := <-serveErr:
		if err != nil {
			return fmt.Errorf("HTTP server failed: %w", err)
		}
		return nil
	case <-ctx.Done():
		zap.L().Info("HTTP server shutting down gracefully...")
		shutCtx, shutCancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer shutCancel()
		if err := srv.Shutdown(shutCtx); err != nil {
			zap.L().Error("HTTP server forced shutdown", zap.Error(err))
			return err
		}
		return nil
	}
}

// handleConnections upgrades HTTP connection to WebSocket and registers the client with the ingestor (for ticks).
func handleConnections(w http.ResponseWriter, r *http.Request) {
	defer func() {
		if rec := recover(); rec != nil {
			zap.L().Error("Panic in WebSocket handler (ticks)", zap.Any("recover", rec))
		}
	}()

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		zap.L().Error("WebSocket upgrade error for ticks", zap.Error(err))
		return
	}
	defer ingestor.UnregisterWebSocketClient(conn)

	conn.SetReadLimit(wsMaxMsgSize)
	_ = conn.SetReadDeadline(time.Now().Add(wsPongWait))
	conn.SetPongHandler(func(string) error {
		_ = conn.SetReadDeadline(time.Now().Add(wsPongWait))
		return nil
	})

	ingestor.RegisterWebSocketClient(conn)

	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				zap.L().Debug("WebSocket unexpected close (ticks)", zap.Error(err))
			}
			break
		}
	}
}
func handleIndicatorConnections(w http.ResponseWriter, r *http.Request) {
	defer func() {
		if rec := recover(); rec != nil {
			zap.L().Error("Panic in WebSocket handler (indicators)", zap.Any("recover", rec))
		}
	}()

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		zap.L().Error("WebSocket upgrade error for indicators", zap.Error(err))
		return
	}
	defer indicatorManager.UnregisterWebSocketClient(conn)

	conn.SetReadLimit(wsMaxMsgSize)
	_ = conn.SetReadDeadline(time.Now().Add(wsPongWait))
	conn.SetPongHandler(func(string) error {
		_ = conn.SetReadDeadline(time.Now().Add(wsPongWait))
		return nil
	})

	indicatorManager.RegisterWebSocketClient(conn)
	zap.L().Info("New WebSocket client connected for indicator data", zap.String("remote_addr", conn.RemoteAddr().String()))

	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				zap.L().Debug("WebSocket unexpected close (indicators)", zap.Error(err))
			}
			break
		}
	}
}

// handleCandleConnections uses CandleGenerator for registration and streaming.
func handleCandleConnections(w http.ResponseWriter, r *http.Request) {
	defer func() {
		if rec := recover(); rec != nil {
			zap.L().Error("Panic in WebSocket handler (candles)", zap.Any("recover", rec))
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

	conn.SetReadLimit(wsMaxMsgSize)
	_ = conn.SetReadDeadline(time.Now().Add(wsPongWait))
	conn.SetPongHandler(func(string) error {
		_ = conn.SetReadDeadline(time.Now().Add(wsPongWait))
		return nil
	})

	candleGenerator.RegisterWebSocketClient(conn)
	zap.L().Info("New WebSocket client connected for candle data", zap.String("remote_addr", conn.RemoteAddr().String()))

	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				zap.L().Debug("WebSocket unexpected close (candles)", zap.Error(err))
			}
			break
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
