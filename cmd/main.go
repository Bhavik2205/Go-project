package main

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/cache"
	"github.com/Bhavik2205/ML-Bot/internal/data"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	monitor "github.com/Bhavik2205/ML-Bot/internal/execution"
	"github.com/Bhavik2205/ML-Bot/internal/indicators"
	"github.com/Bhavik2205/ML-Bot/internal/server"
	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"github.com/joho/godotenv"
	"go.uber.org/zap"
)

// --- NEW: Panic recovery helper for goroutines ---
func recoverGoroutine(where string) {
	if r := recover(); r != nil {
		zap.L().Error("Panic recovered in goroutine", zap.String("where", where), zap.Any("recover", r))
	}
}

func main() {
	// ─── Immediate console output ────────────────────────────────────────────────
	fmt.Println("🚀 ML-Bot starting up...")
	fmt.Printf("PID: %d\n", os.Getpid())
	fmt.Println("Initializing components...")

	// ─── Initialize logger as early as possible ────────────────────────────────
	utils.InitLogger("info", "app.log")
	defer func() {
		err := zap.L().Sync()
		if err != nil && err.Error() != "sync /dev/stderr: invalid argument" {
			fmt.Printf("Warning: failed to sync zap logger: %v\n", err)
		}
	}()

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
	fmt.Println("✅ App config loaded")

	dbCfg, err := utils.LoadDatabaseConfig("configs/database.yaml")
	if err != nil {
		wrappedErr := utils.WrapError(1002, "Failed to load database config", err)
		zap.L().Fatal(wrappedErr.Error())
	}
	fmt.Println("✅ Database config loaded")

	redisCfg, err := utils.LoadRedisConfig()
	if err != nil {
		wrappedErr := utils.WrapError(1003, "Failed to load Redis config", err)
		zap.L().Fatal(wrappedErr.Error())
	}
	fmt.Println("✅ Redis config loaded")

	indicatorsCfg, err := utils.LoadIndicatorsConfig("configs/indicators.yaml")
	if err != nil {
		wrappedErr := utils.WrapError(1005, "Failed to load indicators config", err)
		zap.L().Fatal(wrappedErr.Error())
	}
	fmt.Println("✅ Indicators config loaded")

	// ─── Re-init logger with config from file ───────────────────────────────────
	utils.InitLogger(appCfg.Log.Level, appCfg.Log.Output)
	zap.L().Info("📦 ML-Bot service starting up...")

	// ─── Initialize Database ────────────────────────────────────────────────────
	dbClient, err := db.NewPostgresClient(dbCfg)
	if err != nil {
		wrappedErr := utils.WrapError(2001, "Failed to connect to PostgreSQL", err)
		zap.L().Fatal(wrappedErr.Error())
	}
	fmt.Println("✅ Database connected")

	sqlDB, err := dbClient.DB.DB()
	if err != nil {
		zap.L().Fatal("Failed to get underlying DB connection for close", zap.Error(err))
	}
	defer func() {
		if err := sqlDB.Close(); err != nil {
			zap.L().Error("Failed to close DB connection", zap.Error(err))
		} else {
			zap.L().Info("Database connection closed gracefully.")
		}
	}()

	// ─── Initialize Redis ───────────────────────────────────────────────────────
	redisClient, err := cache.NewRedisClient(redisCfg)
	if err != nil {
		wrappedErr := utils.WrapError(2002, "Failed to connect to Redis", err)
		zap.L().Fatal(wrappedErr.Error())
	}
	fmt.Println("✅ Redis connected")

	defer func() {
		if err := redisClient.Client.Close(); err != nil {
			zap.L().Error("Failed to close Redis client", zap.Error(err))
		} else {
			zap.L().Info("Redis client closed gracefully.")
		}
	}()

	// Test Redis connection
	if err := redisClient.Set("test_key", "Hello from Redis!", time.Minute); err != nil {
		zap.L().Warn("⚠️ Redis SET test failed", zap.Error(err))
	}
	if val, err := redisClient.Get("test_key"); err == nil {
		zap.L().Info("✅ Redis GET success", zap.String("value", val))
		fmt.Println("✅ Redis connection verified")
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

	// var client *api.ZerodhaClient
	// if !appCfg.Market.Simulate {
	// 	client = api.NewZerodhaClient(apiKey, apiSecret, accessToken)
	// 	server.SetZerodhaClient(client)
	// 	fmt.Println("✅ Zerodha client initialized")
	// }
	var client *api.ZerodhaClient
	if !appCfg.Market.Simulate {
		client = api.NewZerodhaClient(apiKey, apiSecret, accessToken)
		fmt.Println("✅ Zerodha client initialized")

		// ─── Validate Zerodha Session (live mode only) ────────────────────────────
		user, err := client.Kite.GetUserProfile()
		if err != nil {
			wrappedErr := utils.WrapError(3003, "Invalid Zerodha session or token expired", err)
			zap.L().Fatal(wrappedErr.Error())
		}
		zap.L().Info("✅ Zerodha login success", zap.String("username", user.UserName), zap.String("userID", user.UserID))
		fmt.Println("✅ Zerodha session validated")
	}
	server.SetZerodhaClient(client) // ✅ now works for both simulate and live

	server.SetDBClient(dbClient)
	server.SetRedisClient(redisClient)

	// ─── Validate Zerodha Session ───────────────────────────────────────────────
	// if !appCfg.Market.Simulate {
	// 	user, err := client.Kite.GetUserProfile()
	// 	if err != nil {
	// 		wrappedErr := utils.WrapError(3003, "Invalid Zerodha session or token expired", err)
	// 		zap.L().Fatal(wrappedErr.Error())
	// 	}
	// 	zap.L().Info("✅ Zerodha login success", zap.String("username", user.UserName), zap.String("userID", user.UserID))
	// 	fmt.Println("✅ Zerodha session validated")
	// }

	// ─── Setup graceful shutdown context ────────────────────────────────────────
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		defer recoverGoroutine("SignalHandler")
		sig := <-sigChan
		zap.L().Info("Received shutdown signal", zap.String("signal", sig.String()))
		fmt.Printf("\n🛑 Shutdown signal received: %s\n", sig.String())
		cancel()
	}()

	// ─── Initialize WebSocket client maps ───────────────────────────────────────
	wsClients := &sync.Map{}
	candleWsClients := &sync.Map{}
	indicatorWsClients := &sync.Map{}
	fmt.Println("✅ WebSocket client maps initialized")

	// ─── Initialize channels ───────────────────────────────────────────────────
	indicatorManagerInputCh := make(chan indicators.Candle, 5000)

	// ─── Initialize core components ────────────────────────────────────────────
	dataIngestor := data.NewMarketDataIngestor(dbClient, redisClient, wsClients, appCfg, indicatorsCfg)
	candleGenerator := data.NewCandleGenerator(dbClient, redisClient, appCfg, candleWsClients, indicatorManagerInputCh)
	indicatorManager := data.NewIndicatorManager(dbClient, appCfg, indicatorsCfg, indicatorManagerInputCh, indicatorWsClients)

	// Set server dependencies
	server.SetIngestor(dataIngestor, wsClients)
	server.SetCandleClients(candleWsClients)
	server.SetIndicatorClients(indicatorWsClients)
	server.SetCandleGenerator(candleGenerator)

	fmt.Println("✅ Core components initialized")

	// ─── Verify all initializations ────────────────────────────────────────────
	verifyInitialization(dbClient, redisClient, wsClients, candleWsClients, indicatorWsClients)
	// Use it before starting the server:
	if err := checkPortAvailable(appCfg.Server.HTTPPort); err != nil {
		zap.L().Fatal("Port is already in use", zap.Int("port", appCfg.Server.HTTPPort), zap.Error(err))
	}
	// ─── Start HTTP Server ─────────────────────────────────────────────────────
	go func() {
		defer recoverGoroutine("HTTPServer")
		zap.L().Info("🌐 Starting HTTP server...", zap.Int("port", appCfg.Server.HTTPPort))
		fmt.Printf("🌐 HTTP server starting on port %d\n", appCfg.Server.HTTPPort)
		server.StartHTTPServer(appCfg.Server.HTTPPort)
	}()

	// Wait a bit for HTTP server to start
	time.Sleep(2 * time.Second)

	// ─── Start Market Data Services ────────────────────────────────────────────
	go func() {
		defer recoverGoroutine("MarketDataIngestor")
		zap.L().Info("📊 Starting market data ingestion...")
		fmt.Println("📊 Market data ingestion started")
		dataIngestor.StartIngestionAndBroadcast(ctx)
	}()

	go func() {
		defer recoverGoroutine("CandleDBWriter")
		candleGenerator.StartCandleDBWriter(ctx)
	}()

	go func() {
		defer recoverGoroutine("CandleGenerator")
		zap.L().Info("🕯️ Starting candle generation...")
		fmt.Println("🕯️ Candle generation started")
		candleGenerator.StartCandleGeneration(ctx)
	}()

	go func() {
		defer recoverGoroutine("IndicatorManager")
		zap.L().Info("📈 Starting indicator calculations...")
		fmt.Println("📈 Indicator calculations started")
		indicatorManager.StartIndicatorCalculations(ctx)
	}()

	go func() {
		defer recoverGoroutine("SystemMonitor")
		monitor.StartSystemMonitor(5*time.Second, func(msg string) {
			zap.L().Warn(msg)
		})
	}()

	// ─── Subscribe to Market Symbols ───────────────────────────────────────────
	symbols := []string{
		"ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV",
		"BPCL", "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT",
		"GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
		"ITC", "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT", "LTIM", "M&M", "MARUTI", "NESTLEIND",
		"NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA", "TATACONSUM", "TATAMOTORS",
		"TATASTEEL", "TCS", "TECHM", "TITAN", "ULTRACEMCO", "UPL", "WIPRO", "DMART", "IRCTC", "PIDILITIND",
		"PAGEIND", "MUTHOOTFIN", "JUBLFOOD", "ETERNAL", "NYKAA", "POLYCAB", "BOSCHLTD", "GUJGASLTD", "DEEPAKNTR",
		"TATAELXSI", "AARTIIND", "LTTS", "SRF", "ABB", "ADANIGREEN", "ADANIENSOL", "ALKEM", "AMBUJACEM",
		"AUROPHARMA", "BALKRISIND", "BANDHANBNK", "BANKBARODA", "BERGEPAINT", "BIOCON", "CANBK", "CHOLAFIN",
		"COLPAL", "CONCOR", "CROMPTON", "DABUR", "DALBHARAT", "DIXON", "ESCORTS", "EXIDEIND", "FEDERALBNK",
		"GAIL", "GLENMARK", "GODREJCP", "GODREJPROP", "HAVELLS", "HDFCAMC", "HINDPETRO", "ICICIGI", "ICICIPRULI",
		"IDFCFIRSTB", "IGL", "INDIGO", "INDUSTOWER", "IOC", "IPCALAB", "LTF", "LALPATHLAB", "LICHSGFIN",
	}

	preferredExchanges := []string{"NSE"}
	var instruments []*api.InstrumentInfo

	fmt.Printf("🔍 Looking up %d symbols...\n", len(symbols))

	for _, symbol := range symbols {
		info, err := client.FindInstrumentToken(symbol, preferredExchanges)
		if err != nil {
			zap.L().Warn("⚠️ Failed to find instrument token, skipping subscription",
				zap.String("symbol", symbol), zap.Error(err))
			continue
		}
		zap.L().Info("🔔 Attempting to subscribe",
			zap.String("symbol", info.Symbol),
			zap.String("exchange", info.Exchange),
			zap.Int("token", int(info.Token)))
		instruments = append(instruments, info)

		// Ensure instruments are in the DB
		var existingInstrument db.Instrument
		res := dbClient.DB.Where("instrument_token = ?", info.Token).First(&existingInstrument)
		if res.Error != nil && res.Error.Error() == "record not found" {
			newInstrument := db.Instrument{
				InstrumentToken: uint(info.Token),
				Exchange:        info.Exchange,
				Tradingsymbol:   info.Symbol,
				InstrumentType:  info.InstrumentType,
				Name:            info.Name,
				Segment:         info.Segment,
				TickSize:        float64(info.TickSize),
				LotSize:         int(info.LotSize),
				Expiry:          nil,
				Strike:          nil,
				OptionType:      "",
				LastUpdated:     time.Now(),
			}
			if createErr := dbClient.DB.Create(&newInstrument).Error; createErr != nil {
				zap.L().Error("Failed to save new instrument to DB",
					zap.Error(createErr), zap.String("symbol", info.Symbol))
			} else {
				zap.L().Info("Saved new instrument to DB", zap.String("symbol", info.Symbol))
			}
		} else if res.Error != nil {
			zap.L().Error("Error checking for existing instrument in DB",
				zap.Error(res.Error), zap.String("symbol", info.Symbol))
		}
	}

	if len(instruments) == 0 {
		err := utils.WrapError(4001, "No valid instruments found to subscribe to. Check symbol configuration or Zerodha API response.", nil)
		zap.L().Fatal(err.Error())
	}

	fmt.Printf("✅ Found %d valid instruments\n", len(instruments))

	// ─── Start Market Data Feed ────────────────────────────────────────────────
	if appCfg.Market.Simulate {
		zap.L().Info("Starting **SIMULATED** market data feed based on app configuration.")
		fmt.Println("🎮 Starting SIMULATED market data feed")
		startSimulatedFeed(ctx, appCfg, instruments, redisClient, dbClient)
	} else {
		zap.L().Info("Starting **REAL** market data feed (via Zerodha Ticker) based on app configuration.")
		fmt.Println("📈 Starting REAL market data feed")
		if err := client.SubscribeToTicks(instruments, redisClient); err != nil {
			wrappedErr := utils.WrapError(4002, "Zerodha WebSocket subscription error", err)
			zap.L().Fatal(wrappedErr.Error())
		}
	}

	// ─── Block until context is cancelled ──────────────────────────────────────
	fmt.Println("\n✅ All systems ready! Press Ctrl+C to shutdown")
	zap.L().Info("✅ All systems ready! Waiting for shutdown signal...")

	<-ctx.Done()

	fmt.Println("\n🛑 Shutting down ML-Bot service gracefully...")
	zap.L().Info("Shutting down ML-Bot service gracefully...")

	// Give services time to shutdown gracefully
	time.Sleep(2 * time.Second)

	fmt.Println("✅ ML-Bot service stopped.")
	zap.L().Info("ML-Bot service stopped.")
}

func verifyInitialization(dbClient *db.DBClient, redisClient *cache.RedisClient,
	wsClients, candleWsClients, indicatorWsClients *sync.Map) {

	zap.L().Info("🔍 Verifying component initialization...")

	if dbClient == nil {
		zap.L().Fatal("❌ Database client not initialized")
	}
	if redisClient == nil {
		zap.L().Fatal("❌ Redis client not initialized")
	}
	if wsClients == nil {
		zap.L().Fatal("❌ WebSocket clients map not initialized")
	}
	if candleWsClients == nil {
		zap.L().Fatal("❌ Candle WebSocket clients map not initialized")
	}
	if indicatorWsClients == nil {
		zap.L().Fatal("❌ Indicator WebSocket clients map not initialized")
	}

	zap.L().Info("✅ All components initialized successfully")
	fmt.Println("✅ All components verified")
}

func startSimulatedFeed(ctx context.Context, appCfg *utils.AppConfig,
	instruments []*api.InstrumentInfo, redisClient *cache.RedisClient,
	dbClient *db.DBClient) {

	simGenerator := api.NewSimulatedZerodhaClient()

	// Ensure simulated instruments exist in database
	for _, info := range instruments {
		var existingInstrument db.Instrument
		res := dbClient.DB.Where("tradingsymbol = ? AND exchange = ?",
			info.Symbol, info.Exchange).First(&existingInstrument)

		if res.Error != nil {
			if res.Error.Error() == "record not found" {
				newInstrument := db.Instrument{
					InstrumentToken: uint(info.Token),
					Exchange:        info.Exchange,
					Tradingsymbol:   info.Symbol,
					InstrumentType:  info.InstrumentType,
					Name:            info.Name,
					Segment:         info.Segment,
					TickSize:        info.TickSize,
					LotSize:         int(info.LotSize),
					Expiry:          nil,
					Strike:          nil,
					OptionType:      "",
					LastUpdated:     time.Now(),
				}
				if createErr := dbClient.DB.Create(&newInstrument).Error; createErr != nil {
					zap.L().Error("Failed to save new instrument to DB for simulation",
						zap.Error(createErr), zap.String("symbol", info.Symbol))
				} else {
					zap.L().Info("✅ Saved new instrument to DB for simulation",
						zap.String("symbol", info.Symbol))
				}
			}
		}
	}

	// Start simulated ticker
	go func() {
		defer recoverGoroutine("SimulatedTicker")
		if err := simGenerator.SimulateTicks(ctx, instruments, redisClient,
			appCfg.Market.SimulationSpeedMultiplier); err != nil {
			wrappedErr := utils.WrapError(4003, "Simulated market data feed error", err)
			zap.L().Fatal(wrappedErr.Error())
		}
	}()
}

// Add this function to main.go
func checkPortAvailable(port int) error {
	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return err
	}
	listener.Close()
	return nil
}
