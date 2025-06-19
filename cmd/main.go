package main

import (
	"context"
	"fmt"
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
	// ─── Initialize logger as early as possible ────────────────────────────────
	utils.InitLogger("info", "app.log") // Default to info and app.log before loading config
	defer func() {
		// Ensure all buffered logs are written before exiting
		err := zap.L().Sync()
		// Ignore specific error on stderr when closing, common in some environments
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

	// NEW: Load Strategy Config (if uncommented in the future)
	// strategyCfg, err := utils.LoadStrategyConfig("configs/strategy.yaml")
	// if err != nil {
	// 	wrappedErr := utils.WrapError(1004, "Failed to load strategy config", err)
	// 	zap.L().Fatal(wrappedErr.Error())
	// }

	// NEW: Load Indicators Config
	indicatorsCfg, err := utils.LoadIndicatorsConfig("configs/indicators.yaml")
	if err != nil {
		wrappedErr := utils.WrapError(1005, "Failed to load indicators config", err)
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

	// Defer closing DB connection
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
	defer func() {
		if err := redisClient.Client.Close(); err != nil { // Use Client.Close()
			zap.L().Error("Failed to close Redis client", zap.Error(err))
		} else {
			zap.L().Info("Redis client closed gracefully.")
		}
	}()

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

	// ─── Setup graceful shutdown context ────────────────────────────────────────
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel() // Ensure cancel is called on exit

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		defer recoverGoroutine("SignalHandler") // --- NEW: Panic recovery
		sig := <-sigChan
		zap.L().Info("Received shutdown signal", zap.String("signal", sig.String()))
		cancel() // Trigger context cancellation
	}()

	// ─── Initialize and inject Ingestor and other Dependencies ──────────────────
	// wsClients will be shared between server (for accepting connections) and ingestor (for broadcasting)
	wsClients := &sync.Map{}          // Initialize the map here for TICK data
	candleWsClients := &sync.Map{}    // NEW: Initialize the map here for CANDLE data
	indicatorWsClients := &sync.Map{} // NEW: Initialize the map here for INDICATOR data

	// 7. Create channels for inter-service communication
	// This channel transports completed candles from CandleGenerator to IndicatorsManager.
	// A buffered channel is used to avoid blocking the CandleGenerator.
	indicatorManagerInputCh := make(chan indicators.Candle, 100) // NEW: Channel for candles -> IndicatorsManager

	dataIngestor := data.NewMarketDataIngestor(dbClient, redisClient, wsClients, appCfg, indicatorsCfg)
	// NEW: Pass candleWsClients to CandleGenerator
	candleGenerator := data.NewCandleGenerator(dbClient, redisClient, appCfg, candleWsClients, indicatorManagerInputCh)
	server.SetCandleGenerator(candleGenerator) // <-- ADD THIS LINE
	go func() {
		defer recoverGoroutine("CandleDBWriter") // --- NEW: Panic recovery
		candleGenerator.StartCandleDBWriter(ctx)
	}()
	indicatorManager := data.NewIndicatorManager(dbClient, appCfg, indicatorsCfg, indicatorManagerInputCh, indicatorWsClients)

	server.SetZerodhaClient(client)
	server.SetDBClient(dbClient)
	server.SetRedisClient(redisClient)
	server.SetIngestor(dataIngestor, wsClients)    // Inject the ingestor and shared WS map for ticks
	server.SetCandleClients(candleWsClients)       // NEW: Inject the shared WS map for candles
	server.SetIndicatorClients(indicatorWsClients) // NEW: Inject the shared WS map for indicators

	// ─── Start HTTP Server (for WebSocket connections and API endpoints) ────────
	// Run HTTP server in a goroutine so it doesn't block main.
	go func() {
		defer recoverGoroutine("HTTPServer")
		server.StartHTTPServer(appCfg.Server.HTTPPort)
	}()

	// ─── Start Market Data Ingestion & Broadcasting ─────────────────────────────
	go func() {
		defer recoverGoroutine("MarketDataIngestor")
		dataIngestor.StartIngestionAndBroadcast(ctx)
	}()

	// ─── Start Candle Generation ────────────────────────────────────────────────
	// This starts listening to Redis ticks and aggregating them into candles.
	go func() {
		defer recoverGoroutine("CandleGenerator")
		candleGenerator.StartCandleGeneration(ctx)
	}()
	// NEW: Start Indicator Calculation ───────────────────────────────────────────
	go func() {
		defer recoverGoroutine("IndicatorManager")
		indicatorManager.StartIndicatorCalculations(ctx)
	}()

	// NEW: Start System Monitoring ----------------------------------------------
	go func() {
		defer recoverGoroutine("SystemMonitor")
		monitor.StartSystemMonitor(5*time.Second, func(msg string) {
			zap.L().Warn(msg)
		})
	}()

	// ─── Subscribe to Market Symbols (Zerodha ticker) ───────────────────────────
	symbols := []string{"ITCHOTELS", "HDFCBANK", "RELIANCE", "TCS"}
	preferredExchanges := []string{"NSE"}
	var instruments []*api.InstrumentInfo

	for _, symbol := range symbols {
		info, err := client.FindInstrumentToken(symbol, preferredExchanges)
		if err != nil {
			zap.L().Warn("⚠️ Failed to find instrument token, skipping subscription", zap.String("symbol", symbol), zap.Error(err))
			continue
		}
		zap.L().Info("🔔 Attempting to subscribe", zap.String("symbol", info.Symbol), zap.String("exchange", info.Exchange), zap.Int("token", int(info.Token)))
		instruments = append(instruments, info)

		// Ensure instruments are in the DB. This is crucial as MarketData uses InstrumentToken as FK.
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
				Expiry:          nil, // Default, update if F&O
				Strike:          nil, // Default, update if F&O
				OptionType:      "",  // Default, update if F&O
				LastUpdated:     time.Now(),
			}
			if createErr := dbClient.DB.Create(&newInstrument).Error; createErr != nil {
				zap.L().Error("Failed to save new instrument to DB", zap.Error(createErr), zap.String("symbol", info.Symbol))
			} else {
				zap.L().Info("Saved new instrument to DB", zap.String("symbol", info.Symbol))
			}
		} else if res.Error != nil {
			zap.L().Error("Error checking for existing instrument in DB", zap.Error(res.Error), zap.String("symbol", info.Symbol))
		}
	}

	if len(instruments) == 0 {
		err := utils.WrapError(4001, "No valid instruments found to subscribe to. Check symbol configuration or Zerodha API response.", nil)
		zap.L().Fatal(err.Error())
	}

	// ─── Start Zerodha Ticker Subscription (publishes to Redis) ─────────────────

	if appCfg.Market.Simulate {
		zap.L().Info("Starting **SIMULATED** market data feed based on app configuration.")
		// Create a simulated client instance.
		simGenerator := api.NewSimulatedZerodhaClient()

		// Define instruments for simulation. These are hardcoded for simplicity.
		// In a production simulation environment, you might load these from a dedicated config file or a database.
		instruments := []*api.InstrumentInfo{
			{Token: 12345, Symbol: "ITCHOTELS", Exchange: "NSE", InstrumentType: "EQ", Name: "ITC Hotels", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 67890, Symbol: "HDFCBANK", Exchange: "NSE", InstrumentType: "EQ", Name: "HDFC Bank", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 11223, Symbol: "RELIANCE", Exchange: "NSE", InstrumentType: "EQ", Name: "Reliance Industries", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 44556, Symbol: "TCS", Exchange: "NSE", InstrumentType: "EQ", Name: "Tata Consultancy Services", Segment: "EQ", TickSize: 0.05, LotSize: 1},
		}

		// Ensure these simulated instruments exist in the database.
		// This is critical because other parts of the application (e.g., market data storage)
		// rely on instrument tokens being foreign keys.
		// For each simulated instrument, ensure it exists in the DB
		for _, info := range instruments {
			var existingInstrument db.Instrument
			// Fix 1: Change existence check to use tradingsymbol and exchange
			// This matches the unique constraint "instruments_tradingsymbol_exchange_key"
			res := dbClient.DB.Where("tradingsymbol = ? AND exchange = ?", info.Symbol, info.Exchange).First(&existingInstrument)

			if res.Error != nil {
				if res.Error.Error() == "record not found" {
					// Instrument not found by tradingsymbol and exchange, so create a new one.
					newInstrument := db.Instrument{
						InstrumentToken: uint(info.Token), // Use the simulated token
						Exchange:        info.Exchange,
						Tradingsymbol:   info.Symbol,
						InstrumentType:  info.InstrumentType,
						Name:            info.Name,
						Segment:         info.Segment,
						TickSize:        info.TickSize,
						LotSize:         int(info.LotSize),
						Expiry:          nil, // Assuming these are not set for simulated EQ instruments
						Strike:          nil,
						OptionType:      "",
						LastUpdated:     time.Now(),
					}
					if createErr := dbClient.DB.Create(&newInstrument).Error; createErr != nil {
						zap.L().Error("Failed to save new instrument to DB for simulation",
							zap.Error(createErr),
							zap.String("symbol", info.Symbol),
						)
					} else {
						zap.L().Info("✅ Saved new instrument to DB for simulation", zap.String("symbol", info.Symbol))
					}
				} else {
					// Handle other database errors during the check
					zap.L().Error("Error checking for existing instrument in DB for simulation",
						zap.Error(res.Error),
						zap.String("symbol", info.Symbol),
					)
				}
			} else {
				if existingInstrument.InstrumentToken != uint(info.Token) {
					existingInstrument.InstrumentToken = uint(info.Token)
					if updateErr := dbClient.DB.Save(&existingInstrument).Error; updateErr != nil {
						zap.L().Error("Failed to update existing instrument token for simulation", zap.Error(updateErr), zap.String("symbol", info.Symbol))
					} else {
						zap.L().Info("Updated existing instrument token for simulation", zap.String("symbol", info.Symbol))
					}
				}
			}
		}

		// Start the simulated ticker, passing the graceful shutdown context,
		// the instruments, Redis client, and the configured simulation speed.
		go func() { // Run in a goroutine so it doesn't block main
			defer recoverGoroutine("SimulatedTicker")
			if err := simGenerator.SimulateTicks(ctx, instruments, redisClient, appCfg.Market.SimulationSpeedMultiplier); err != nil {
				wrappedErr := utils.WrapError(4003, "Simulated market data feed error", err)
				zap.L().Fatal(wrappedErr.Error())
			}
		}()
	} else {
		zap.L().Info("Starting **REAL** market data feed (via Zerodha Ticker) based on app configuration.")
		if err := client.SubscribeToTicks(instruments, redisClient); err != nil {
			wrappedErr := utils.WrapError(4002, "Zerodha WebSocket subscription error", err)
			zap.L().Fatal(wrappedErr.Error())
		}
	}

	// ─── Block until context is cancelled (e.g., via SIGINT/SIGTERM) ────────────
	<-ctx.Done()
	zap.L().Info("Shutting down ML-Bot service gracefully...")

	zap.L().Info("ML-Bot service stopped.")
}
