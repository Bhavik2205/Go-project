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
	indicatorManagerInputCh := make(chan indicators.Candle, 5000) // NEW: Channel for candles -> IndicatorsManager

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
			{Token: 10001, Symbol: "ADANIENT", Exchange: "NSE", InstrumentType: "EQ", Name: "Adani Enterprises", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10002, Symbol: "ADANIPORTS", Exchange: "NSE", InstrumentType: "EQ", Name: "Adani Ports", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10003, Symbol: "APOLLOHOSP", Exchange: "NSE", InstrumentType: "EQ", Name: "Apollo Hospitals", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10004, Symbol: "ASIANPAINT", Exchange: "NSE", InstrumentType: "EQ", Name: "Asian Paints", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10005, Symbol: "AXISBANK", Exchange: "NSE", InstrumentType: "EQ", Name: "Axis Bank", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10006, Symbol: "BAJAJ-AUTO", Exchange: "NSE", InstrumentType: "EQ", Name: "Bajaj Auto", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10007, Symbol: "BAJFINANCE", Exchange: "NSE", InstrumentType: "EQ", Name: "Bajaj Finance", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10008, Symbol: "BAJAJFINSV", Exchange: "NSE", InstrumentType: "EQ", Name: "Bajaj Finserv", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10009, Symbol: "BPCL", Exchange: "NSE", InstrumentType: "EQ", Name: "BPCL", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10010, Symbol: "BHARTIARTL", Exchange: "NSE", InstrumentType: "EQ", Name: "Bharti Airtel", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10011, Symbol: "BRITANNIA", Exchange: "NSE", InstrumentType: "EQ", Name: "Britannia", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10012, Symbol: "CIPLA", Exchange: "NSE", InstrumentType: "EQ", Name: "Cipla", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10013, Symbol: "COALINDIA", Exchange: "NSE", InstrumentType: "EQ", Name: "Coal India", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10014, Symbol: "DIVISLAB", Exchange: "NSE", InstrumentType: "EQ", Name: "Divi's Lab", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10015, Symbol: "DRREDDY", Exchange: "NSE", InstrumentType: "EQ", Name: "Dr Reddy's", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10016, Symbol: "EICHERMOT", Exchange: "NSE", InstrumentType: "EQ", Name: "Eicher Motors", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10017, Symbol: "GRASIM", Exchange: "NSE", InstrumentType: "EQ", Name: "Grasim", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10018, Symbol: "HCLTECH", Exchange: "NSE", InstrumentType: "EQ", Name: "HCL Tech", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10019, Symbol: "HDFCBANK", Exchange: "NSE", InstrumentType: "EQ", Name: "HDFC Bank", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10020, Symbol: "HDFCLIFE", Exchange: "NSE", InstrumentType: "EQ", Name: "HDFC Life", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10021, Symbol: "HEROMOTOCO", Exchange: "NSE", InstrumentType: "EQ", Name: "Hero MotoCorp", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10022, Symbol: "HINDALCO", Exchange: "NSE", InstrumentType: "EQ", Name: "Hindalco", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10023, Symbol: "HINDUNILVR", Exchange: "NSE", InstrumentType: "EQ", Name: "Hindustan Unilever", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10024, Symbol: "ICICIBANK", Exchange: "NSE", InstrumentType: "EQ", Name: "ICICI Bank", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10025, Symbol: "ITC", Exchange: "NSE", InstrumentType: "EQ", Name: "ITC", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10026, Symbol: "INDUSINDBK", Exchange: "NSE", InstrumentType: "EQ", Name: "IndusInd Bank", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10027, Symbol: "INFY", Exchange: "NSE", InstrumentType: "EQ", Name: "Infosys", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10028, Symbol: "JSWSTEEL", Exchange: "NSE", InstrumentType: "EQ", Name: "JSW Steel", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10029, Symbol: "KOTAKBANK", Exchange: "NSE", InstrumentType: "EQ", Name: "Kotak Bank", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10030, Symbol: "LT", Exchange: "NSE", InstrumentType: "EQ", Name: "Larsen & Toubro", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10031, Symbol: "LTIM", Exchange: "NSE", InstrumentType: "EQ", Name: "LTIMindtree", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10032, Symbol: "M&M", Exchange: "NSE", InstrumentType: "EQ", Name: "Mahindra & Mahindra", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10033, Symbol: "MARUTI", Exchange: "NSE", InstrumentType: "EQ", Name: "Maruti Suzuki", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10034, Symbol: "NESTLEIND", Exchange: "NSE", InstrumentType: "EQ", Name: "Nestle India", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10035, Symbol: "NTPC", Exchange: "NSE", InstrumentType: "EQ", Name: "NTPC", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10036, Symbol: "ONGC", Exchange: "NSE", InstrumentType: "EQ", Name: "ONGC", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10037, Symbol: "POWERGRID", Exchange: "NSE", InstrumentType: "EQ", Name: "Power Grid", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10038, Symbol: "RELIANCE", Exchange: "NSE", InstrumentType: "EQ", Name: "Reliance Industries", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10039, Symbol: "SBILIFE", Exchange: "NSE", InstrumentType: "EQ", Name: "SBI Life", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10040, Symbol: "SBIN", Exchange: "NSE", InstrumentType: "EQ", Name: "State Bank of India", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10041, Symbol: "SUNPHARMA", Exchange: "NSE", InstrumentType: "EQ", Name: "Sun Pharma", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10042, Symbol: "TATACONSUM", Exchange: "NSE", InstrumentType: "EQ", Name: "Tata Consumer", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10043, Symbol: "TATAMOTORS", Exchange: "NSE", InstrumentType: "EQ", Name: "Tata Motors", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10044, Symbol: "TATASTEEL", Exchange: "NSE", InstrumentType: "EQ", Name: "Tata Steel", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10045, Symbol: "TCS", Exchange: "NSE", InstrumentType: "EQ", Name: "TCS", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10046, Symbol: "TECHM", Exchange: "NSE", InstrumentType: "EQ", Name: "Tech Mahindra", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10047, Symbol: "TITAN", Exchange: "NSE", InstrumentType: "EQ", Name: "Titan Company", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10048, Symbol: "ULTRACEMCO", Exchange: "NSE", InstrumentType: "EQ", Name: "UltraTech Cement", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10049, Symbol: "UPL", Exchange: "NSE", InstrumentType: "EQ", Name: "UPL", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10050, Symbol: "WIPRO", Exchange: "NSE", InstrumentType: "EQ", Name: "Wipro", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10051, Symbol: "DMART", Exchange: "NSE", InstrumentType: "EQ", Name: "Avenue Supermarts", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10052, Symbol: "IRCTC", Exchange: "NSE", InstrumentType: "EQ", Name: "IRCTC", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10053, Symbol: "PIDILITIND", Exchange: "NSE", InstrumentType: "EQ", Name: "Pidilite Industries", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10054, Symbol: "PAGEIND", Exchange: "NSE", InstrumentType: "EQ", Name: "Page Industries", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10055, Symbol: "MUTHOOTFIN", Exchange: "NSE", InstrumentType: "EQ", Name: "Muthoot Finance", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10056, Symbol: "JUBLFOOD", Exchange: "NSE", InstrumentType: "EQ", Name: "Jubilant FoodWorks", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10057, Symbol: "ETERNAL", Exchange: "NSE", InstrumentType: "EQ", Name: "Zomato", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10058, Symbol: "NYKAA", Exchange: "NSE", InstrumentType: "EQ", Name: "Nykaa", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10059, Symbol: "POLYCAB", Exchange: "NSE", InstrumentType: "EQ", Name: "Polycab India", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10060, Symbol: "BOSCHLTD", Exchange: "NSE", InstrumentType: "EQ", Name: "Bosch", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10061, Symbol: "GUJGASLTD", Exchange: "NSE", InstrumentType: "EQ", Name: "Gujarat Gas", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10062, Symbol: "DEEPAKNTR", Exchange: "NSE", InstrumentType: "EQ", Name: "Deepak Nitrite", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10063, Symbol: "TATAELXSI", Exchange: "NSE", InstrumentType: "EQ", Name: "Tata Elxsi", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10064, Symbol: "AARTIIND", Exchange: "NSE", InstrumentType: "EQ", Name: "Aarti Industries", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10065, Symbol: "LTTS", Exchange: "NSE", InstrumentType: "EQ", Name: "L&T Tech Services", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10066, Symbol: "SRF", Exchange: "NSE", InstrumentType: "EQ", Name: "SRF", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10067, Symbol: "ABB", Exchange: "NSE", InstrumentType: "EQ", Name: "ABB India", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10068, Symbol: "ADANIGREEN", Exchange: "NSE", InstrumentType: "EQ", Name: "Adani Green", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10069, Symbol: "ADANITRANS", Exchange: "NSE", InstrumentType: "EQ", Name: "Adani Transmission", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10070, Symbol: "ALKEM", Exchange: "NSE", InstrumentType: "EQ", Name: "Alkem Labs", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10071, Symbol: "AMBUJACEM", Exchange: "NSE", InstrumentType: "EQ", Name: "Ambuja Cements", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10072, Symbol: "AUROPHARMA", Exchange: "NSE", InstrumentType: "EQ", Name: "Aurobindo Pharma", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10073, Symbol: "BALKRISIND", Exchange: "NSE", InstrumentType: "EQ", Name: "Balkrishna Ind", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10074, Symbol: "BANDHANBNK", Exchange: "NSE", InstrumentType: "EQ", Name: "Bandhan Bank", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10075, Symbol: "BANKBARODA", Exchange: "NSE", InstrumentType: "EQ", Name: "Bank of Baroda", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10076, Symbol: "BERGEPAINT", Exchange: "NSE", InstrumentType: "EQ", Name: "Berger Paints", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10077, Symbol: "BIOCON", Exchange: "NSE", InstrumentType: "EQ", Name: "Biocon", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10078, Symbol: "CANBK", Exchange: "NSE", InstrumentType: "EQ", Name: "Canara Bank", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10079, Symbol: "CHOLAFIN", Exchange: "NSE", InstrumentType: "EQ", Name: "Cholamandalam", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10080, Symbol: "COLPAL", Exchange: "NSE", InstrumentType: "EQ", Name: "Colgate Palmolive", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10081, Symbol: "CONCOR", Exchange: "NSE", InstrumentType: "EQ", Name: "Container Corp", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10082, Symbol: "CROMPTON", Exchange: "NSE", InstrumentType: "EQ", Name: "Crompton Greaves", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10083, Symbol: "DABUR", Exchange: "NSE", InstrumentType: "EQ", Name: "Dabur India", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10084, Symbol: "DALBHARAT", Exchange: "NSE", InstrumentType: "EQ", Name: "Dalmia Bharat", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10085, Symbol: "DIXON", Exchange: "NSE", InstrumentType: "EQ", Name: "Dixon Tech", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10086, Symbol: "ESCORTS", Exchange: "NSE", InstrumentType: "EQ", Name: "Escorts Kubota", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10087, Symbol: "EXIDEIND", Exchange: "NSE", InstrumentType: "EQ", Name: "Exide Industries", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10088, Symbol: "FEDERALBNK", Exchange: "NSE", InstrumentType: "EQ", Name: "Federal Bank", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10089, Symbol: "GAIL", Exchange: "NSE", InstrumentType: "EQ", Name: "GAIL India", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10090, Symbol: "GLENMARK", Exchange: "NSE", InstrumentType: "EQ", Name: "Glenmark Pharma", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10091, Symbol: "GODREJCP", Exchange: "NSE", InstrumentType: "EQ", Name: "Godrej Consumer", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10092, Symbol: "GODREJPROP", Exchange: "NSE", InstrumentType: "EQ", Name: "Godrej Properties", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10093, Symbol: "HAVELLS", Exchange: "NSE", InstrumentType: "EQ", Name: "Havells India", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10094, Symbol: "HDFCAMC", Exchange: "NSE", InstrumentType: "EQ", Name: "HDFC AMC", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10095, Symbol: "HINDPETRO", Exchange: "NSE", InstrumentType: "EQ", Name: "HPCL", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10096, Symbol: "ICICIGI", Exchange: "NSE", InstrumentType: "EQ", Name: "ICICI Lombard", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10097, Symbol: "ICICIPRULI", Exchange: "NSE", InstrumentType: "EQ", Name: "ICICI Prudential", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10098, Symbol: "IDFCFIRSTB", Exchange: "NSE", InstrumentType: "EQ", Name: "IDFC First Bank", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10099, Symbol: "IGL", Exchange: "NSE", InstrumentType: "EQ", Name: "Indraprastha Gas", Segment: "EQ", TickSize: 0.05, LotSize: 1},
			{Token: 10100, Symbol: "INDIGO", Exchange: "NSE", InstrumentType: "EQ", Name: "InterGlobe Aviation", Segment: "EQ", TickSize: 0.05, LotSize: 1},
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
