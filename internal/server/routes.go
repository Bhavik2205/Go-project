package server

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/api/handlers/stockHandler"
	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
)

type Zerodha interface {
	FindInstrumentToken(symbol string, exchanges []string) (*api.InstrumentInfo, error)
}

var zerodha *api.ZerodhaClient
var clients = make(map[*websocket.Conn]bool)
var broadcast = make(chan []byte)
var livePrices = make(map[string][]byte)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

func SetZerodhaClient(client *api.ZerodhaClient) {
	zerodha = client
}

func StartHTTPServer() {
	router := mux.NewRouter()

	router.Use(enableCORS)

	router.HandleFunc("/api/instrument", stockHandler.HandleInstrumentLookup(zerodha)).Methods("GET")
	router.HandleFunc("/ws", handleConnections)

	go handleMessages()

	log.Println("🌐 Unified HTTP + WebSocket server started on :8000")
	log.Fatal(http.ListenAndServe(":8000", router))
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
