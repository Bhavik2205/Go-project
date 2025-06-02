// package server

// import (
// 	"encoding/json"
// 	"log"
// 	"net/http"

// 	"github.com/Bhavik2205/ML-Bot/internal/api"
// 	"github.com/Bhavik2205/ML-Bot/internal/api/handlers/stockHandler"
// 	"github.com/gorilla/mux"
// 	"github.com/gorilla/websocket"
// )

// type Zerodha interface {
// 	FindInstrumentToken(symbol string, exchanges []string) (*api.InstrumentInfo, error)
// }

// var zerodha *api.ZerodhaClient
// var clients = make(map[*websocket.Conn]bool)
// var broadcast = make(chan []byte)
// var livePrices = make(map[string][]byte)

// var upgrader = websocket.Upgrader{
// 	CheckOrigin: func(r *http.Request) bool { return true },
// }

// func SetZerodhaClient(client *api.ZerodhaClient) {
// 	zerodha = client
// }

// func StartHTTPServer() {
// 	router := mux.NewRouter()

// 	router.Use(enableCORS)

// 	router.HandleFunc("/api/instrument", stockHandler.HandleInstrumentLookup(zerodha)).Methods("GET")
// 	router.HandleFunc("/ws", handleConnections)

// 	go handleMessages()

// 	log.Println("🌐 Unified HTTP + WebSocket server started on :8000")
// 	log.Fatal(http.ListenAndServe(":8000", router))
// }

// func handleConnections(w http.ResponseWriter, r *http.Request) {
// 	ws, err := upgrader.Upgrade(w, r, nil)
// 	if err != nil {
// 		log.Println("WebSocket upgrade error:", err)
// 		return
// 	}
// 	clients[ws] = true
// 	log.Println("🧑‍💻 New WebSocket client connected.")
// }

// func handleMessages() {
// 	for {
// 		msg := <-broadcast
// 		for client := range clients {
// 			err := client.WriteMessage(websocket.TextMessage, msg)
// 			if err != nil {
// 				log.Printf("WebSocket write error: %v", err)
// 				client.Close()
// 				delete(clients, client)
// 			}
// 		}
// 	}
// }

// func PushToFrontend(msg []byte) {
// 	var tick map[string]interface{}
// 	if err := json.Unmarshal(msg, &tick); err == nil {
// 		if symbol, ok := tick["symbol"].(string); ok {
// 			livePrices[symbol] = msg
// 		}
// 	}

// 	broadcast <- msg
// }

// func enableCORS(h http.Handler) http.Handler {
// 	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
// 		w.Header().Set("Access-Control-Allow-Origin", "*")
// 		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
// 		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

// 		if r.Method == http.MethodOptions {
// 			return
// 		}

// 		h.ServeHTTP(w, r)
// 	})
// }

// internal/server/routes.go
package server

import (
	"encoding/json"
	"fmt" // Import fmt for cleaner log messages using port
	"log"
	"net/http"
	"strconv" // Import for converting port to string
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/api/handlers/stockHandler"
	"github.com/Bhavik2205/ML-Bot/internal/cache" // New import
	"github.com/Bhavik2205/ML-Bot/internal/db"    // New import
	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
)

type Zerodha interface {
	FindInstrumentToken(symbol string, exchanges []string) (*api.InstrumentInfo, error)
}

var (
	zerodhaClient *api.ZerodhaClient
	dbClient      *db.DBClient       // New global variable for DB client
	redisClient   *cache.RedisClient // New global variable for Redis client
	clients       = make(map[*websocket.Conn]bool)
	broadcast     = make(chan []byte)
	livePrices    = make(map[string][]byte)
	upgrader      = websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool { return true },
	}
)

// SetZerodhaClient sets the Zerodha API client
func SetZerodhaClient(client *api.ZerodhaClient) {
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

// StartHTTPServer starts the HTTP and WebSocket server
func StartHTTPServer(port int) {
	router := mux.NewRouter()

	router.Use(enableCORS)

	// Example handler using the DB client (you'd need to modify stockHandler accordingly)
	router.HandleFunc("/api/instrument", stockHandler.HandleInstrumentLookup(zerodhaClient)).Methods("GET")

	// Example of a new handler that uses DB and Redis (you would implement actual logic)
	router.HandleFunc("/api/data/users", func(w http.ResponseWriter, r *http.Request) {
		// Example: Fetch users from DB
		var users []db.User
		if err := dbClient.Find(&users).Error; err != nil {
			http.Error(w, "Failed to fetch users", http.StatusInternalServerError)
			return
		}
		json.NewEncoder(w).Encode(users)
	}).Methods("GET")

	router.HandleFunc("/api/cache/test", func(w http.ResponseWriter, r *http.Request) {
		// Example: Use Redis to set and get a value
		err := redisClient.Set("test_web_key", "Value from Web!", 5*time.Minute)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to set cache: %v", err), http.StatusInternalServerError)
			return
		}
		val, err := redisClient.Get("test_web_key")
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to get cache: %v", err), http.StatusInternalServerError)
			return
		}
		fmt.Fprintf(w, "Cache test successful: %s", val)
	}).Methods("GET")

	router.HandleFunc("/ws", handleConnections)

	go handleMessages()

	log.Printf("🌐 Unified HTTP + WebSocket server started on :%d", port)
	log.Fatal(http.ListenAndServe(":"+strconv.Itoa(port), router))
}

func handleConnections(w http.ResponseWriter, r *http.Request) {
	ws, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("WebSocket upgrade error:", err)
		return
	}
	clients[ws] = true
	log.Println("🧑‍💻 New WebSocket client connected.")
}

func handleMessages() {
	for {
		msg := <-broadcast
		for client := range clients {
			err := client.WriteMessage(websocket.TextMessage, msg)
			if err != nil {
				log.Printf("WebSocket write error: %v", err)
				client.Close()
				delete(clients, client)
			}
		}
	}
}

func PushToFrontend(msg []byte) {
	var tick map[string]interface{}
	if err := json.Unmarshal(msg, &tick); err == nil {
		if symbol, ok := tick["symbol"].(string); ok {
			livePrices[symbol] = msg
		}
	}

	broadcast <- msg
}

func enableCORS(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == http.MethodOptions {
			return
		}

		h.ServeHTTP(w, r)
	})
}
