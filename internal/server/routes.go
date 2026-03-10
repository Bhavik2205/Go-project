package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/api/handlers/stockHandler"
	"github.com/Bhavik2205/ML-Bot/internal/cache"
	"github.com/Bhavik2205/ML-Bot/internal/data"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

// ZerodhaAPI interface to abstract Zerodha client methods used by handlers.
type ZerodhaAPI interface {
	FindInstrumentToken(symbol string, exchanges []string) (*api.InstrumentInfo, error)
}

var (
	zerodhaClient      ZerodhaAPI
	dbClient           *db.DBClient
	redisClient        *cache.RedisClient
	ingestor           *data.MarketDataIngestor
	wsClients          *sync.Map
	candleWsClients    *sync.Map
	indicatorWsClients *sync.Map
	candleGenerator    *data.CandleGenerator

	upgrader = websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			return true
		},
		ReadBufferSize:  1024,
		WriteBufferSize: 1024,
	}

	// Mutex for thread-safe WebSocket operations
	wsMutex sync.RWMutex
)

// SetZerodhaClient sets the Zerodha API client
func SetZerodhaClient(client ZerodhaAPI) {
	zerodhaClient = client
	zap.L().Info("✅ Zerodha client set in server")
}

// SetDBClient sets the database client
func SetDBClient(client *db.DBClient) {
	dbClient = client
	zap.L().Info("✅ Database client set in server")
}

// SetRedisClient sets the Redis client
func SetRedisClient(client *cache.RedisClient) {
	redisClient = client
	zap.L().Info("✅ Redis client set in server")
}

// SetIngestor sets the market data ingestor and shares the WebSocket clients map.
func SetIngestor(i *data.MarketDataIngestor, clients *sync.Map) {
	ingestor = i
	wsClients = clients
	zap.L().Info("✅ Market data ingestor set in server")
}

// SetCandleClients injects the shared WebSocket client map for candle data.
func SetCandleClients(clients *sync.Map) {
	candleWsClients = clients
	zap.L().Info("✅ Candle WebSocket clients set in server")
}

func SetIndicatorClients(clients *sync.Map) {
	indicatorWsClients = clients
	zap.L().Info("✅ Indicator WebSocket clients set in server")
}

// SetCandleGenerator sets the candle generator
func SetCandleGenerator(gen *data.CandleGenerator) {
	candleGenerator = gen
	zap.L().Info("✅ Candle generator set in server")
}

// StartHTTPServer starts the HTTP and WebSocket server
func StartHTTPServer(port int) {
	router := mux.NewRouter()

	router.Use(enableCORS)
	router.Use(recoverMiddleware)
	router.Use(loggingMiddleware)

	// API Routes
	// router.HandleFunc("/api/instrument", stockHandler.HandleInstrumentLookup(zerodhaClient.(*api.ZerodhaClient))).Methods("GET")
	if zerodhaClient != nil {
		router.HandleFunc("/api/instrument",
			stockHandler.HandleInstrumentLookup(zerodhaClient.(*api.ZerodhaClient))).
			Methods("GET")
	} else {
		zap.L().Info("simulate=true – Zerodha routes disabled")
	}

	router.HandleFunc("/api/data/users", func(w http.ResponseWriter, r *http.Request) {
		var users []db.User
		if err := dbClient.Find(&users).Error; err != nil {
			zap.L().Error("Failed to fetch users from DB", zap.Error(err))
			http.Error(w, "Failed to fetch users", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(users)
	}).Methods("GET")

	router.HandleFunc("/api/cache/test", func(w http.ResponseWriter, r *http.Request) {
		err := redisClient.Set("test_web_key", "Value from Web!", 5*time.Minute)
		if err != nil {
			zap.L().Error("Failed to set cache key", zap.Error(err))
			http.Error(w, fmt.Sprintf("Failed to set cache: %v", err), http.StatusInternalServerError)
			return
		}
		val, err := redisClient.Get("test_web_key")
		if err != nil {
			zap.L().Error("Failed to get cache key", zap.Error(err))
			http.Error(w, fmt.Sprintf("Failed to get cache: %v", err), http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, "Cache test successful: %s", val)
	}).Methods("GET")

	// Health check
	router.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		health := map[string]interface{}{
			"status":    "healthy",
			"timestamp": time.Now().Unix(),
			"service":   "ML-Bot",
			"version":   "1.0.0",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(health)
	}).Methods("GET")

	// WebSocket endpoints
	router.HandleFunc("/ws", handleConnections).Methods("GET")
	router.HandleFunc("/ws/candles", handleCandleConnections).Methods("GET")
	router.HandleFunc("/ws/indicators", handleIndicatorConnections).Methods("GET")
	router.HandleFunc("/ws/heatmap", HeatmapWebSocketHandler(data.GetMarketHeatmap())).Methods("GET")

	// WebSocket test endpoint
	router.HandleFunc("/ws/test", handleTestConnection).Methods("GET")

	zap.L().Info("🌐 Unified HTTP + WebSocket server starting...", zap.Int("port", port))
	fmt.Printf("🌐 HTTP server listening on port %d\n", port)

	if err := http.ListenAndServe(":"+strconv.Itoa(port), router); err != nil {
		zap.L().Fatal("Failed to start HTTP server", zap.Error(err))
	}
}

// handleConnections upgrades HTTP connection to WebSocket and registers the client
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

	clientKey := conn.RemoteAddr().String()

	// Register client
	wsMutex.Lock()
	if wsClients != nil {
		wsClients.Store(clientKey, conn)
	} else {
		zap.L().Error("WebSocket clients map not initialized")
		conn.Close()
		return
	}
	wsMutex.Unlock()

	zap.L().Info("✅ New WebSocket client connected for tick data",
		zap.String("remote_addr", clientKey))
	fmt.Printf("✅ WebSocket tick client connected: %s\n", clientKey)

	// Setup connection
	conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	conn.SetPongHandler(func(string) error {
		conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	defer func() {
		wsMutex.Lock()
		wsClients.Delete(clientKey)
		wsMutex.Unlock()
		conn.Close()
		zap.L().Info("❌ WebSocket tick client disconnected",
			zap.String("remote_addr", clientKey))
		fmt.Printf("❌ WebSocket tick client disconnected: %s\n", clientKey)
	}()

	// Keep connection alive
	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				zap.L().Debug("WebSocket unexpected close for tick client",
					zap.Error(err), zap.String("remote_addr", clientKey))
			}
			break
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

	clientKey := conn.RemoteAddr().String()

	wsMutex.Lock()
	if indicatorWsClients != nil {
		indicatorWsClients.Store(clientKey, conn)
	} else {
		zap.L().Error("Indicator WebSocket clients map not initialized")
		conn.Close()
		return
	}
	wsMutex.Unlock()

	zap.L().Info("✅ New WebSocket client connected for indicator data",
		zap.String("remote_addr", clientKey))
	fmt.Printf("✅ WebSocket indicator client connected: %s\n", clientKey)

	defer func() {
		wsMutex.Lock()
		indicatorWsClients.Delete(clientKey)
		wsMutex.Unlock()
		conn.Close()
		zap.L().Info("❌ WebSocket indicator client disconnected",
			zap.String("remote_addr", clientKey))
		fmt.Printf("❌ WebSocket indicator client disconnected: %s\n", clientKey)
	}()

	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				zap.L().Debug("WebSocket unexpected close for indicator client",
					zap.Error(err), zap.String("remote_addr", clientKey))
			}
			break
		}
	}
}

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

	clientKey := conn.RemoteAddr().String()

	candleGenerator.RegisterWebSocketClient(conn)
	zap.L().Info("✅ New WebSocket client connected for candle data",
		zap.String("remote_addr", clientKey))
	fmt.Printf("✅ WebSocket candle client connected: %s\n", clientKey)

	defer func() {
		candleGenerator.UnregisterWebSocketClient(conn)
		conn.Close()
		zap.L().Info("❌ WebSocket candle client disconnected",
			zap.String("remote_addr", clientKey))
		fmt.Printf("❌ WebSocket candle client disconnected: %s\n", clientKey)
	}()

	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				zap.L().Debug("WebSocket unexpected close for candle client",
					zap.Error(err), zap.String("remote_addr", clientKey))
			}
			break
		}
	}
}

func handleTestConnection(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		zap.L().Error("Test WebSocket upgrade failed", zap.Error(err))
		return
	}
	defer conn.Close()

	clientKey := conn.RemoteAddr().String()
	zap.L().Info("✅ Test WebSocket client connected", zap.String("remote_addr", clientKey))
	fmt.Printf("✅ Test WebSocket client connected: %s\n", clientKey)

	// Send test message
	testMsg := map[string]interface{}{
		"type":      "test",
		"message":   "WebSocket connection successful",
		"timestamp": time.Now().Unix(),
		"service":   "ML-Bot",
	}

	if err := conn.WriteJSON(testMsg); err != nil {
		zap.L().Error("Failed to send test message", zap.Error(err))
		return
	}

	zap.L().Info("✅ Test message sent to client", zap.String("remote_addr", clientKey))

	// Keep connection alive for 10 seconds
	time.Sleep(10 * time.Second)

	zap.L().Info("✅ Test connection completed", zap.String("remote_addr", clientKey))
}

func enableCORS(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		h.ServeHTTP(w, r)
	})
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

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		zap.L().Info("HTTP request",
			zap.String("method", r.Method),
			zap.String("path", r.URL.Path),
			zap.String("remote_addr", r.RemoteAddr))

		next.ServeHTTP(w, r)

		zap.L().Info("HTTP response",
			zap.String("method", r.Method),
			zap.String("path", r.URL.Path),
			zap.Duration("duration", time.Since(start)))
	})
}
