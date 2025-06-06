package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"sync" // Required for sync.Map
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/api/handlers/stockHandler"
	"github.com/Bhavik2205/ML-Bot/internal/cache"
	"github.com/Bhavik2205/ML-Bot/internal/data" // New import for ingestion logic
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"go.uber.org/zap" // Use zap logger
)

// ZerodhaAPI interface to abstract Zerodha client methods used by handlers.
type ZerodhaAPI interface {
	FindInstrumentToken(symbol string, exchanges []string) (*api.InstrumentInfo, error)
}

var (
	zerodhaClient ZerodhaAPI // Use the interface type
	dbClient      *db.DBClient
	redisClient   *cache.RedisClient
	ingestor      *data.MarketDataIngestor // New global variable for the ingestor
	wsClients     = &sync.Map{}            // Shared sync.Map for WebSocket clients
	upgrader      = websocket.Upgrader{
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
	wsClients = clients // Assign the shared map
}

// StartHTTPServer starts the HTTP and WebSocket server
func StartHTTPServer(port int) {
	router := mux.NewRouter()

	router.Use(enableCORS)

	router.HandleFunc("/api/instrument", stockHandler.HandleInstrumentLookup(zerodhaClient.(*api.ZerodhaClient))).Methods("GET")

	router.HandleFunc("/api/data/users", func(w http.ResponseWriter, r *http.Request) {
		var users []db.User
		if err := dbClient.Find(&users).Error; err != nil {
			zap.L().Error("Failed to fetch users from DB", zap.Error(err))
			http.Error(w, "Failed to fetch users", http.StatusInternalServerError)
			return
		}
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

	router.HandleFunc("/ws", handleConnections)

	zap.L().Info("🌐 Unified HTTP + WebSocket server starting...", zap.Int("port", port))
	// Log.Fatal will exit the application if the server fails to start, which is acceptable
	// for the main HTTP listener in many production setups that rely on external process managers.
	if err := http.ListenAndServe(":"+strconv.Itoa(port), router); err != nil {
		zap.L().Fatal("Failed to start HTTP server", zap.Error(err))
	}
}

// handleConnections upgrades HTTP connection to WebSocket and registers the client with the ingestor.
func handleConnections(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		zap.L().Error("WebSocket upgrade error", zap.Error(err))
		return
	}
	// Important: Defer Unregister for proper cleanup when connection closes
	defer ingestor.UnregisterWebSocketClient(conn)

	ingestor.RegisterWebSocketClient(conn) // Register the new client

	// Keep the connection alive, listen for close messages from the client.
	// This loop prevents the handler from exiting, allowing the deferred unregister to run on close.
	for {
		// Read message to detect client disconnects or pings/pongs
		// We don't necessarily process incoming messages for this app, but need to read them to detect disconnects.
		_, _, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				zap.L().Debug("WebSocket unexpected close detected", zap.Error(err), zap.String("remote_addr", conn.RemoteAddr().String()))
			} else {
				// Normal close, or other non-critical errors (e.g., "use of closed network connection")
				zap.L().Info("WebSocket client disconnected", zap.String("remote_addr", conn.RemoteAddr().String()))
			}
			break // Exit the loop, triggering the defer
		}
	}
}

func enableCORS(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK) // Respond to preflight request
			return
		}

		h.ServeHTTP(w, r)
	})
}
