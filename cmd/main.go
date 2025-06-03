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

package main

import (
	"os"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/cache"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/server"
	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"github.com/joho/godotenv"
	"go.uber.org/zap"
)

func main() {
	// ─── Initialize logger as early as possible ────────────────────────────────
	utils.InitLogger("info", "app.log") // Default to info and app.log before loading config
	defer zap.L().Sync()

	// ─── Load Environment Variables ─────────────────────────────────────────────
	if err := godotenv.Load(); err != nil {
		zap.L().Warn("⚠️ .env file not found, using system environment variables", zap.Error(err))
	} else {
		zap.L().Info("✅ .env file loaded successfully")
	}

	// ─── Load Configurations ────────────────────────────────────────────────────
	appCfg, err := utils.LoadAppConfig("configs/app.yaml")
	if err != nil {
		wrappedErr := utils.WrapError(1001, "Failed to load app config", err)
		zap.L().Fatal(wrappedErr.Error())
	}
	dbCfg, err := utils.LoadDatabaseConfig("configs/database.yaml")
	if err != nil {
		wrappedErr := utils.WrapError(1002, "Failed to load database config", err)
		zap.L().Fatal(wrappedErr.Error())
	}
	redisCfg, err := utils.LoadRedisConfig()
	if err != nil {
		wrappedErr := utils.WrapError(1003, "Failed to load Redis config", err)
		zap.L().Fatal(wrappedErr.Error())
	}

	// ─── Re-init logger with config from file ───────────────────────────────────
	utils.InitLogger(appCfg.Log.Level, appCfg.Log.Output)
	zap.L().Info("📦 ML-Bot service starting up...")

	// ─── Initialize Database ────────────────────────────────────────────────────
	dbClient, err := db.NewPostgresClient(dbCfg)
	if err != nil {
		wrappedErr := utils.WrapError(2001, "Failed to connect to PostgreSQL", err)
		zap.L().Fatal(wrappedErr.Error())
	}

	// ─── Initialize Redis ───────────────────────────────────────────────────────
	redisClient, err := cache.NewRedisClient(redisCfg)
	if err != nil {
		wrappedErr := utils.WrapError(2002, "Failed to connect to Redis", err)
		zap.L().Fatal(wrappedErr.Error())
	}

	// Optional Redis test
	if err := redisClient.Set("test_key", "Hello from Redis!", time.Minute); err != nil {
		zap.L().Warn("⚠️ Redis SET test failed", zap.Error(err))
	}
	if val, err := redisClient.Get("test_key"); err == nil {
		zap.L().Info("✅ Redis GET success", zap.String("value", val))
	}

	// ─── Load Zerodha Credentials ───────────────────────────────────────────────
	apiKey := os.Getenv("ZERODHA_API_KEY")
	apiSecret := os.Getenv("ZERODHA_API_SECRET")
	if apiKey == "" || apiSecret == "" {
		err := utils.WrapError(3001, "ZERODHA_API_KEY or ZERODHA_API_SECRET not set in environment", nil)
		zap.L().Fatal(err.Error())
	}

	accessToken, err := api.LoadAccessTokenFromFile(".access_token")
	if err != nil {
		wrappedErr := utils.WrapError(3002, "Failed to load Zerodha access token", err)
		zap.L().Fatal(wrappedErr.Error())
	}

	client := api.NewZerodhaClient(apiKey, apiSecret, accessToken)

	// ─── Inject Dependencies ────────────────────────────────────────────────────
	server.SetZerodhaClient(client)
	server.SetDBClient(dbClient)
	server.SetRedisClient(redisClient)

	// ─── Validate Zerodha Session ───────────────────────────────────────────────
	user, err := client.Kite.GetUserProfile()
	if err != nil {
		wrappedErr := utils.WrapError(3003, "Invalid Zerodha session or token expired", err)
		zap.L().Fatal(wrappedErr.Error())
	}
	zap.L().Info("✅ Zerodha login success", zap.String("username", user.UserName), zap.String("userID", user.UserID))

	// ─── Start HTTP Server ──────────────────────────────────────────────────────
	go server.StartHTTPServer(appCfg.Server.HTTPPort)

	// ─── Subscribe to Market Symbols ────────────────────────────────────────────
	symbols := []string{"NIFTY 50", "NIFTY BANK", "RELIANCE", "TCS"}
	preferredExchanges := []string{"NSE"}
	var instruments []*api.InstrumentInfo

	for _, symbol := range symbols {
		info, err := client.FindInstrumentToken(symbol, preferredExchanges)
		if err != nil {
			zap.L().Warn("⚠️ Failed to subscribe to symbol", zap.String("symbol", symbol), zap.Error(err))
			continue
		}
		zap.L().Info("🔔 Subscribing", zap.String("symbol", info.Symbol), zap.String("exchange", info.Exchange), zap.Int("token", int(info.Token)))
		instruments = append(instruments, info)
	}

	if len(instruments) == 0 {
		err := utils.WrapError(4001, "No valid instruments to subscribe to", nil)
		zap.L().Fatal(err.Error())
	}

	// ─── Start Ticker Subscription ──────────────────────────────────────────────
	if err := client.SubscribeToTicks(instruments, func(data []byte) {
		server.PushToFrontend(data)
	}); err != nil {
		wrappedErr := utils.WrapError(4002, "WebSocket subscription error", err)
		zap.L().Fatal(wrappedErr.Error())
	}

	// ─── Block Forever ──────────────────────────────────────────────────────────
	select {}
}
