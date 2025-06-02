// package main

// import (
// 	"fmt"
// 	"log"
// 	"os"

// 	"github.com/Bhavik2205/ML-Bot/internal/api"
// 	"github.com/Bhavik2205/ML-Bot/internal/server"
// 	"github.com/joho/godotenv"
// )

// func main() {
// 	if err := godotenv.Load(); err != nil {
// 		fmt.Println("⚠️  .env file not found, using system env vars")
// 	}

// 	apiKey := os.Getenv("ZERODHA_API_KEY")
// 	apiSecret := os.Getenv("ZERODHA_API_SECRET")

// 	accessToken, err := api.LoadAccessTokenFromFile(".access_token")
// 	if err != nil {
// 		log.Fatal(err)
// 	}

// 	client := api.NewZerodhaClient(apiKey, apiSecret, accessToken)
// 	server.SetZerodhaClient(client)

// 	user, err := client.Kite.GetUserProfile()
// 	if err != nil {
// 		log.Fatalf("❌ Invalid session or token expired: %v", err)
// 	}
// 	fmt.Printf("✅ Logged in as: %s (%s)\n", user.UserName, user.UserID)

// 	// Start WebSocket server for frontend clients
// 	// go server.StartWebSocketServer()
// 	go server.StartHTTPServer()

// 	// Multiple symbols
// 	symbols := []string{"NIFTY 50", "NIFTY BANK", "RELIANCE", "TCS"}
// 	preferredExchanges := []string{"NSE"}

// 	var infos []*api.InstrumentInfo
// 	for _, symbol := range symbols {
// 		info, err := client.FindInstrumentToken(symbol, preferredExchanges)
// 		if err != nil {
// 			log.Printf("❌ Failed to find token for %s: %v", symbol, err)
// 			continue
// 		}
// 		fmt.Printf("🔍 Subscribing to %s on %s (Token: %d)\n", info.Symbol, info.Exchange, info.Token)
// 		infos = append(infos, info)
// 	}

// 	if len(infos) == 0 {
// 		log.Fatal("❌ No valid instruments to subscribe to.")
// 	}

// 	// Pass the handler callback to push ticks to frontend via server package
// 	err = client.SubscribeToTicks(infos, func(jsonData []byte) {
// 		server.PushToFrontend(jsonData)
// 	})
// 	if err != nil {
// 		log.Fatalf("❌ WebSocket error: %v", err)
// 	}

// 	select {} // block forever
// }

// cmd/main.go
package main

import (
	"fmt"
	"log"
	"os"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/cache" // New import
	"github.com/Bhavik2205/ML-Bot/internal/db"    // New import
	"github.com/Bhavik2205/ML-Bot/internal/server"
	"github.com/Bhavik2205/ML-Bot/internal/utils" // New import
	"github.com/joho/godotenv"
)

func main() {
	if err := godotenv.Load(); err != nil {
		fmt.Println("⚠️ .env file not found, using system env vars")
	} else {
		fmt.Println("✅ .env file loaded successfully.") // <--- Add this
	}

	// --- Load Configurations ---
	appCfg, err := utils.LoadAppConfig("configs/app.yaml")
	if err != nil {
		log.Fatalf("❌ Failed to load app config: %v", err)
	}
	dbCfg, err := utils.LoadDatabaseConfig("configs/database.yaml")
	if err != nil {
		log.Fatalf("❌ Failed to load database config: %v", err)
	}

	redisCfg, err := utils.LoadRedisConfig()
	if err != nil {
		log.Fatalf("❌ Failed to load Redis config: %v", err)
	}

	// --- Initialize Logger (using appCfg.Log) ---
	utils.InitLogger(appCfg.Log.Level, appCfg.Log.Output) // You'd implement InitLogger in utils/logger.go

	// --- Initialize Database ---
	dbClient, err := db.NewPostgresClient(dbCfg)
	if err != nil {
		log.Fatalf("❌ Failed to initialize database: %v", err)
	}
	// Auto-migrate database schemas
	if err := dbClient.AutoMigrate(&db.User{}, &db.Instrument{}); err != nil {
		log.Fatalf("❌ Database auto-migration failed: %v", err)
	}

	// --- Initialize Redis ---
	redisClient, err := cache.NewRedisClient(redisCfg)
	if err != nil {
		log.Fatalf("❌ Failed to initialize Redis: %v", err)
	}

	// Example: Set and get something from Redis
	err = redisClient.Set("test_key", "Hello from Redis!", 1*time.Minute)
	if err != nil {
		log.Printf("❌ Failed to set Redis key: %v", err)
	}
	val, err := redisClient.Get("test_key")
	if err == nil {
		log.Printf("✅ Retrieved from Redis: %s", val)
	}

	// --- Zerodha Client Initialization ---
	apiKey := os.Getenv("ZERODHA_API_KEY")
	apiSecret := os.Getenv("ZERODHA_API_SECRET")

	accessToken, err := api.LoadAccessTokenFromFile(".access_token")
	if err != nil {
		log.Fatal(err)
	}

	client := api.NewZerodhaClient(apiKey, apiSecret, accessToken)
	server.SetZerodhaClient(client) // Pass Zerodha client to server

	// --- Pass DB and Redis clients to handlers/services that need them ---
	// You will need to refactor your handlers to accept DB and Redis clients
	// For example, modify stockHandler.HandleInstrumentLookup to take dbClient or redisClient
	// For now, let's just illustrate by passing them where relevant:
	server.SetDBClient(dbClient)       // You'll need to add this setter in server/routes.go
	server.SetRedisClient(redisClient) // You'll need to add this setter in server/routes.go

	user, err := client.Kite.GetUserProfile()
	if err != nil {
		log.Fatalf("❌ Invalid session or token expired: %v", err)
	}
	fmt.Printf("✅ Logged in as: %s (%s)\n", user.UserName, user.UserID)

	// Start WebSocket server for frontend clients
	go server.StartHTTPServer(appCfg.Server.HTTPPort) // Pass HTTP port from config

	// Multiple symbols
	symbols := []string{"NIFTY 50", "NIFTY BANK", "RELIANCE", "TCS"}
	preferredExchanges := []string{"NSE"}

	var infos []*api.InstrumentInfo
	for _, symbol := range symbols {
		info, err := client.FindInstrumentToken(symbol, preferredExchanges)
		if err != nil {
			log.Printf("❌ Failed to find token for %s: %v", symbol, err)
			continue
		}
		fmt.Printf("🔍 Subscribing to %s on %s (Token: %d)\n", info.Symbol, info.Exchange, info.Token)
		infos = append(infos, info)
	}

	if len(infos) == 0 {
		log.Fatal("❌ No valid instruments to subscribe to.")
	}

	// Pass the handler callback to push ticks to frontend via server package
	err = client.SubscribeToTicks(infos, func(jsonData []byte) {
		server.PushToFrontend(jsonData)
	})
	if err != nil {
		log.Fatalf("❌ WebSocket error: %v", err)
	}

	select {} // block forever
}
