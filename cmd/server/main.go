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
	"github.com/Bhavik2205/ML-Bot/internal/auth"
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

// recoverGoroutine is a deferred panic handler for goroutines.
// Usage: defer recoverGoroutine("YourGoroutineName")
// It logs the panic but does NOT call os.Exit — the main shutdown
// path handles termination via context cancellation.
func recoverGoroutine(where string) {
	if r := recover(); r != nil {
		zap.L().Error("Panic recovered in goroutine",
			zap.String("where", where),
			zap.Any("recover", r),
		)
	}
}

func main() {
	// ─── Initialize logger as early as possible ────────────────────────────────
	// Use defaults until the config file is loaded below.
	utils.InitLogger("info", "app.log") // Default to info and app.log before loading config
	defer func() {
		if err := zap.L().Sync(); err != nil && err.Error() != "sync /dev/stderr: invalid argument" {
			fmt.Printf("Warning: failed to sync zap logger: %v\n", err)
		}
	}()

	// ─── Load Environment Variables ─────────────────────────────────────────────
	if err := godotenv.Load(); err != nil {
		zap.L().Warn("⚠️ .env file not found, using system environment variables", zap.Error(err))
	} else {
		zap.L().Info("✅ .env file loaded successfully")
	}

	// after godotenv.Load() and before auth.MustLoadJWTSecret()
	if err := utils.ValidateRequiredEnv(); err != nil {
		zap.L().Fatal("Environment validation failed", zap.Error(err))
	}

	// ─── SECURITY: Validate JWT secret immediately after env is loaded ──────────
	// This call panics if JWT_SECRET is missing or too short.
	// We want the server to refuse to start in that case — not discover the
	// problem later when someone tries to log in.
	// See internal/auth/jwt.go → MustLoadJWTSecret for details.
	auth.MustLoadJWTSecret()
	zap.L().Info("✅ JWT secret validated")

	// ─── Load Configurations ────────────────────────────────────────────────────
	appCfg, err := utils.LoadAppConfig("configs/app.yaml")
	if err != nil {
		zap.L().Fatal("Failed to load app config",
			zap.Error(utils.WrapError(1001, "Failed to load app config", err)))
	}
	dbCfg, err := utils.LoadDatabaseConfig("configs/database.yaml")
	if err != nil {
		zap.L().Fatal("Failed to load database config",
			zap.Error(utils.WrapError(1002, "Failed to load database config", err)))
	}
	redisCfg, err := utils.LoadRedisConfig()
	if err != nil {
		zap.L().Fatal("Failed to load Redis config",
			zap.Error(utils.WrapError(1003, "Failed to load Redis config", err)))
	}
	indicatorsCfg, err := utils.LoadIndicatorsConfig("configs/indicators.yaml")
	if err != nil {
		zap.L().Fatal("Failed to load indicators config",
			zap.Error(utils.WrapError(1005, "Failed to load indicators config", err)))
	}

	// ─── Re-init logger with config from file ───────────────────────────────────
	utils.InitLogger(appCfg.Log.Level, appCfg.Log.Output)
	zap.L().Info("📦 ML-Bot service starting up...")

	// ─── Initialize Database ────────────────────────────────────────────────────
	dbClient, err := db.NewPostgresClient(dbCfg)
	if err != nil {
		zap.L().Fatal("Failed to connect to database",
			zap.Error(utils.WrapError(2001, "Failed to connect to PostgreSQL", err)))
	}
	sqlDB, err := dbClient.DB.DB()
	if err != nil {
		zap.L().Fatal("Failed to get underlying DB connection", zap.Error(err))
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
		zap.L().Fatal("Failed to connect to Redis",
			zap.Error(utils.WrapError(2002, "Failed to connect to Redis", err)))
	}
	defer func() {
		if err := redisClient.Client.Close(); err != nil {
			zap.L().Error("Failed to close Redis client", zap.Error(err))
		} else {
			zap.L().Info("Redis client closed gracefully.")
		}
	}()

	// Optional Redis connectivity check
	if err := redisClient.Set("test_key", "Hello from Redis!", time.Minute); err != nil {
		zap.L().Warn("⚠️ Redis SET test failed", zap.Error(err))
	} else if val, err := redisClient.Get("test_key"); err == nil {
		zap.L().Info("✅ Redis GET success", zap.String("value", val))
	}

	// ─── Initialize Zerodha client (live mode only) ─────────────────────────────
	var client *api.ZerodhaClient
	if !appCfg.Market.Simulate {
		apiKey := os.Getenv("ZERODHA_API_KEY")
		apiSecret := os.Getenv("ZERODHA_API_SECRET")
		if apiKey == "" || apiSecret == "" {
			zap.L().Fatal("ZERODHA_API_KEY or ZERODHA_API_SECRET not set in environment")
		}

		// SECURITY FIX (AUD-003): Read the Zerodha access token from the
		// ZERODHA_ACCESS_TOKEN environment variable instead of a plaintext file.
		//
		// Previously this read from `.access_token` on disk, which is:
		//   - a credential leak risk if the file ends up in git
		//   - unsafe on any shared or cloud machine
		//
		// To migrate:
		//   1. Remove the `.access_token` file from your project directory
		//   2. Add .access_token to your .gitignore if it isn't already
		//   3. Add ZERODHA_ACCESS_TOKEN=<your_token> to your .env file
		//
		// Note: This is a temporary fix. The proper long-term solution (AUD-003)
		// is an OAuth callback flow that stores encrypted tokens in the database
		// per user. We will implement that in a later sprint.
		accessToken := os.Getenv("ZERODHA_ACCESS_TOKEN")
		if accessToken == "" {
			zap.L().Fatal(
				"ZERODHA_ACCESS_TOKEN environment variable is not set. " +
					"Add it to your .env file. " +
					"Do not store it in the .access_token file on disk.",
			)
		}

		client = api.NewZerodhaClient(apiKey, apiSecret, accessToken)
		user, err := client.Kite.GetUserProfile()
		if err != nil {
			zap.L().Fatal("Invalid Zerodha session or token expired",
				zap.Error(utils.WrapError(3001, "Invalid Zerodha session or token expired", err)))
		}
		zap.L().Info("✅ Zerodha login success",
			zap.String("username", user.UserName),
			zap.String("userID", user.UserID),
		)
	} else {
		zap.L().Info("🧪 Simulation mode enabled — skipping Zerodha credential validation")
	}

	// ─── Setup graceful shutdown context ────────────────────────────────────────
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel() // Ensure cancel is called on exit

	// ─── Initialize shared WebSocket client maps ────────────────────────────────
	// These sync.Maps hold active WebSocket connections for broadcasting.
	// Three separate maps because tick data, candle data, and indicator data
	// are broadcast to different subscriber sets.
	wsClients := &sync.Map{}          // tick data subscribers
	candleWsClients := &sync.Map{}    // candle data subscribers
	indicatorWsClients := &sync.Map{} // indicator data subscribers

	// ─── Channel: candles → indicator manager ───────────────────────────────────
	// Buffer of 5000 so CandleGenerator is never blocked waiting for
	// IndicatorManager to consume. If this fills up, ticks are NOT dropped here
	// (that's a separate known issue tracked as AUD-010).
	indicatorManagerInputCh := make(chan indicators.Candle, 5000)

	// ─── Initialize core services ───────────────────────────────────────────────
	dataIngestor := data.NewMarketDataIngestor(dbClient, redisClient, wsClients, appCfg, indicatorsCfg)
	candleGenerator := data.NewCandleGenerator(dbClient, redisClient, appCfg, candleWsClients, indicatorManagerInputCh)
	server.SetCandleGenerator(candleGenerator)

	indicatorManager := data.NewIndicatorManager(dbClient, appCfg, indicatorsCfg, indicatorManagerInputCh, indicatorWsClients)

	// Inject dependencies into server package via setters.
	// NOTE (AUD-007): these global setters are a known design debt.
	// They will be replaced with constructor-based injection in Sprint 2.
	if client != nil {
		server.SetZerodhaClient(client)
	}
	server.SetDBClient(dbClient)
	server.SetRedisClient(redisClient)
	server.SetAppConfig(appCfg)
	server.SetStartupTime(time.Now())
	server.SetIngestor(dataIngestor, wsClients)
	server.SetCandleClients(candleWsClients)
	server.SetIndicatorClients(indicatorWsClients)
	server.SetIndicatorManager(indicatorManager)

	// ─── Start background goroutines ────────────────────────────────────────────
	//
	// IMPORTANT: None of these goroutines call zap.L().Fatal().
	// Fatal calls os.Exit(1) which bypasses all defer statements — the DB and
	// Redis connections would not be closed cleanly. Instead, errors inside
	// goroutines are logged and the goroutine exits. The main shutdown path
	// (signal handler below) calls cancel() to stop everything gracefully.

	go func() {
		defer recoverGoroutine("CandleDBWriter")
		candleGenerator.StartCandleDBWriter(ctx)
	}()

	go func() {
		defer recoverGoroutine("HTTPServer")
		if err := server.StartHTTPServer(ctx, appCfg.Server.HTTPPort); err != nil {
			zap.L().Error("HTTP server failed, stopping bot", zap.Error(err))
			cancel() // triggers graceful shutdown of all components
		}
	}()

	go func() {
		defer recoverGoroutine("MarketDataIngestor")
		dataIngestor.StartIngestionAndBroadcast(ctx)
	}()

	go func() {
		defer recoverGoroutine("CandleGenerator")
		candleGenerator.StartCandleGeneration(ctx)
	}()

	go func() {
		defer recoverGoroutine("IndicatorManager")
		indicatorManager.StartIndicatorCalculations(ctx)
	}()

	go func() {
		defer recoverGoroutine("SystemMonitor")
		monitor.StartSystemMonitor(5*time.Second, func(msg string) {
			zap.L().Warn(msg)
		})
	}()

	// ─── Subscribe to market data ───────────────────────────
	var instruments []*api.InstrumentInfo

	if !appCfg.Market.Simulate {
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
		for _, symbol := range symbols {
			info, err := client.FindInstrumentToken(symbol, preferredExchanges)
			if err != nil {
				zap.L().Warn("⚠️ Failed to find instrument token, skipping",
					zap.String("symbol", symbol), zap.Error(err))
				continue
			}
			instruments = append(instruments, info)

			var existingInstrument db.Instrument
			res := dbClient.DB.Where("instrument_token = ?", info.Token).First(&existingInstrument)
			if res.Error != nil && res.Error.Error() == "record not found" {
				newInstrument := db.Instrument{
					InstrumentToken: uint32(info.Token),
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
					zap.L().Error("Failed to save instrument to DB",
						zap.Error(createErr), zap.String("symbol", info.Symbol))
				}
			} else if res.Error != nil {
				zap.L().Error("Error checking instrument in DB",
					zap.Error(res.Error), zap.String("symbol", info.Symbol))
			}
		}
		if len(instruments) == 0 {
			zap.L().Fatal("No valid instruments found. Check symbol config or Zerodha API response.")
		}
	}

	// ─── Start market data feed ──────────────────────────────────────────────────
	if appCfg.Market.Simulate {
		zap.L().Info("Starting SIMULATED market data feed based on app configuration.")

		// In simulation mode, drop FK constraints so we can freely upsert
		// instrument tokens without orphaned time-series data.
		fkConstraints := []string{
			"ALTER TABLE market_data DROP CONSTRAINT IF EXISTS fk_market_data_instrument_token CASCADE",
			"ALTER TABLE ohlcv_candles DROP CONSTRAINT IF EXISTS fk_ohlcv_candles_instrument_token CASCADE",
		}
		for _, sql := range fkConstraints {
			if err := dbClient.DB.Exec(sql).Error; err != nil {
				zap.L().Warn("Could not drop FK constraint (may not exist)", zap.Error(err))
			}
		}

		simGenerator := api.NewSimulatedZerodhaClient()
		simInstruments := []*api.InstrumentInfo{
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

		for _, info := range simInstruments {
			if delErr := dbClient.DB.Unscoped().
				Where("tradingsymbol = ? AND exchange = ?", info.Symbol, info.Exchange).
				Delete(&db.Instrument{}).Error; delErr != nil {
				zap.L().Warn("Could not delete stale instrument for simulation (may not exist)",
					zap.String("symbol", info.Symbol), zap.Error(delErr))
			}
			newInstrument := db.Instrument{
				InstrumentToken: uint32(info.Token),
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
				zap.L().Error("Failed to upsert instrument for simulation",
					zap.Error(createErr), zap.String("symbol", info.Symbol))
			} else {
				zap.L().Info("Upserted instrument for simulation",
					zap.String("symbol", info.Symbol), zap.Uint("token", uint(info.Token)))
			}
		}

		// NOTE: This goroutine logs errors but does NOT call Fatal.
		// If the simulated feed fails, the error is logged and the goroutine
		// exits cleanly. The main process continues running (other goroutines
		// are still live) until a shutdown signal is received.
		go func() {
			defer recoverGoroutine("SimulatedTicker")
			if err := simGenerator.SimulateTicks(ctx, simInstruments, redisClient, appCfg.Market.SimulationSpeedMultiplier); err != nil {
				zap.L().Error("Simulated market data feed stopped with error",
					zap.Error(utils.WrapError(4003, "Simulated market data feed error", err)))
				// Signal shutdown so the rest of the app doesn't keep running without data.
				cancel()
			}
		}()
	} else {
		zap.L().Info("Starting REAL market data feed via Zerodha Ticker.")
		if err := client.SubscribeToTicks(instruments, redisClient); err != nil {
			zap.L().Fatal("Zerodha WebSocket subscription error",
				zap.Error(utils.WrapError(4002, "Zerodha WebSocket subscription error", err)))
		}
	}

	// ─── Wait for shutdown signal ────────────────────────────────────────────────────────────
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		defer recoverGoroutine("SignalHandler")
		sig := <-sigChan
		zap.L().Info("Received shutdown signal", zap.String("signal", sig.String()))
		cancel()
	}()

	<-ctx.Done()
	zap.L().Info("Shutting down ML-Bot service gracefully...")
	// Deferred DB and Redis close calls run here.
	zap.L().Info("ML-Bot service stopped.")
}
