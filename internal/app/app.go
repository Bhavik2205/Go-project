package app

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/cache"
	"github.com/Bhavik2205/ML-Bot/internal/data"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	monitor "github.com/Bhavik2205/ML-Bot/internal/execution"
	"github.com/Bhavik2205/ML-Bot/internal/httpapi"
	"github.com/Bhavik2205/ML-Bot/internal/indicators"
	"github.com/Bhavik2205/ML-Bot/internal/marketdata/tickbus"
	"github.com/Bhavik2205/ML-Bot/internal/realtime"
	"github.com/Bhavik2205/ML-Bot/internal/runtime"
	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"go.uber.org/zap"
)

// App is the central application container.
type App struct {
	Config  *utils.AppConfig
	DB      *db.DBClient
	Redis   *cache.RedisClient
	Zerodha *api.ZerodhaClient // nil in simulation mode

	DataIngestor     *data.MarketDataIngestor
	CandleGenerator  *data.CandleGenerator
	IndicatorManager *data.IndicatorManager

	WsClients          *sync.Map
	CandleWsClients    *sync.Map
	IndicatorWsClients *sync.Map
	IndicatorInputCh   chan indicators.Candle

	rm            *runtime.RuntimeManager
	tickBus       tickbus.TickBus
	symbolsConfig *utils.SymbolsConfig
}

// New creates and initialises a new App.
func New(appCfg *utils.AppConfig, dbCfg *utils.DatabaseConfig, redisCfg *utils.RedisConfig, indicatorsCfg *utils.IndicatorsConfig, symbolsCfg *utils.SymbolsConfig) (*App, error) {
	app := &App{
		Config:             appCfg,
		symbolsConfig:      symbolsCfg,
		WsClients:          &sync.Map{},
		CandleWsClients:    &sync.Map{},
		IndicatorWsClients: &sync.Map{},
		IndicatorInputCh:   make(chan indicators.Candle, 5000),
		rm:                 runtime.NewRuntimeManager(),
	}

	// --- Database ---
	dbClient, err := db.NewPostgresClient(dbCfg)
	if err != nil {
		return nil, fmt.Errorf("db init: %w", err)
	}
	app.DB = dbClient

	// --- Redis ---
	redisClient, err := cache.NewRedisClient(redisCfg)
	if err != nil {
		return nil, fmt.Errorf("redis init: %w", err)
	}
	app.Redis = redisClient

	// --- Tick Bus (in‑process or Redis) ---
	var tb tickbus.TickBus
	if appCfg.Market.TickBus == "inprocess" {
		tb = tickbus.NewInProcess()
		zap.L().Info("TickBus: using in‑process channels (zero overhead)")
	} else if appCfg.Market.TickBus == "dual" {
		local := tickbus.NewInProcess()
		redis := tickbus.NewRedis(redisClient, api.RedisMarketDataChannel)
		tb = tickbus.NewDual(local, redis)
		zap.L().Info("TickBus: using dual mode (local + Redis Pub/Sub)")
	} else {
		tb = tickbus.NewRedis(redisClient, api.RedisMarketDataChannel)
		zap.L().Info("TickBus: using Redis Pub/Sub")
	}
	app.tickBus = tb

	// --- Core components ---
	app.DataIngestor = data.NewMarketDataIngestor(app.DB, app.tickBus, app.WsClients, appCfg, indicatorsCfg)
	app.CandleGenerator = data.NewCandleGenerator(app.DB, app.tickBus, appCfg, app.CandleWsClients, app.IndicatorInputCh)
	app.IndicatorManager = data.NewIndicatorManager(app.DB, appCfg, indicatorsCfg, app.IndicatorInputCh, app.IndicatorWsClients)

	// --- Zerodha (live mode only) ---
	if !appCfg.Market.Simulate {
		apiKey := os.Getenv("ZERODHA_API_KEY")
		apiSecret := os.Getenv("ZERODHA_API_SECRET")
		accessToken := os.Getenv("ZERODHA_ACCESS_TOKEN")
		if apiKey == "" || apiSecret == "" || accessToken == "" {
			return nil, fmt.Errorf("Zerodha credentials missing in environment")
		}
		app.Zerodha = api.NewZerodhaClient(apiKey, apiSecret, accessToken)
		// validate session
		if _, err := app.Zerodha.Kite.GetUserProfile(); err != nil {
			return nil, fmt.Errorf("Zerodha session invalid: %w", err)
		}
	}

	// --- Create WebSocket hub and HTTP server ---
	hub := realtime.NewHub(app.DataIngestor, app.CandleGenerator, app.IndicatorManager, app.Redis)

	// Build dependencies for HTTP server
	httpDeps := httpapi.HTTPDeps{
		ZerodhaClient: app.Zerodha,
		DBClient:      app.DB,
		RedisClient:   app.Redis,
		AppConfig:     appCfg,
		StartupTime:   time.Now(),
		Hub:           hub,
	}

	// Create HTTP server instance
	httpServer := httpapi.NewServer(httpDeps)

	// --- Register services with RuntimeManager (in order of dependencies) ---
	app.rm.Register(&serviceAdapter{
		name: "data_ingestor",
		start: func(ctx context.Context) error {
			go app.DataIngestor.StartIngestionAndBroadcast(ctx)
			return nil
		},
		stop: nil,
	})
	app.rm.Register(&serviceAdapter{
		name: "candle_generator",
		start: func(ctx context.Context) error {
			go app.CandleGenerator.StartCandleGeneration(ctx)
			return nil
		},
		// stop: func(ctx context.Context) error { // <-- add this
		// 	return app.CandleGenerator.Stop(ctx)
		// },
	})
	app.rm.Register(&serviceAdapter{
		name: "indicator_manager",
		start: func(ctx context.Context) error {
			go app.IndicatorManager.StartIndicatorCalculations(ctx)
			return nil
		},
	})
	app.rm.Register(&serviceAdapter{
		name: "system_monitor",
		start: func(ctx context.Context) error {
			var statsProvider interface{ DroppedTicks() uint64 }
			if tb, ok := app.tickBus.(*tickbus.InProcessTickBus); ok {
				statsProvider = tb
			}
			go monitor.StartSystemMonitor(5*time.Second, func(msg string) {
				zap.L().Warn(msg)
			}, statsProvider)
			return nil
		},
	})
	app.rm.Register(&serviceAdapter{
		name: "http_server",
		start: func(ctx context.Context) error {
			go func() {
				if err := httpServer.Start(ctx, appCfg.Server.HTTPPort); err != nil {
					zap.L().Error("HTTP server failed", zap.Error(err))
				}
			}()
			return nil
		},
	})
	if appCfg.Market.Simulate {
		app.rm.Register(&serviceAdapter{
			name: "simulated_feed",
			start: func(ctx context.Context) error {
				return app.startSimulatedFeed(ctx)
			},
		})
	} else {
		app.rm.Register(&serviceAdapter{
			name: "real_feed",
			start: func(ctx context.Context) error {
				return app.startRealFeed(ctx)
			},
		})
	}

	return app, nil
}

// Start runs all services via RuntimeManager.
func (a *App) Start(ctx context.Context) error {
	zap.L().Info("Starting App services...")
	return a.rm.StartAll(ctx)
}

// Stop stops all services via RuntimeManager.
func (a *App) Stop(ctx context.Context) error {
	zap.L().Info("Stopping App services...")
	return a.rm.StopAll(ctx)
}

// startSimulatedFeed builds the instrument list and launches the simulated ticker.
func (a *App) startSimulatedFeed(ctx context.Context) error {
	zap.L().Info("Starting SIMULATED market data feed")

	// In simulation mode we need to drop FK constraints so we can freely upsert instruments
	fkConstraints := []string{
		"ALTER TABLE market_data DROP CONSTRAINT IF EXISTS fk_market_data_instrument_token CASCADE",
		"ALTER TABLE ohlcv_candles DROP CONSTRAINT IF EXISTS fk_ohlcv_candles_instrument_token CASCADE",
	}
	for _, sql := range fkConstraints {
		if err := a.DB.Exec(sql).Error; err != nil {
			zap.L().Warn("Could not drop FK constraint (may not exist)", zap.Error(err))
		}
	}

	simGenerator := api.NewSimulatedZerodhaClient()
	simInstruments := make([]*api.InstrumentInfo, len(a.symbolsConfig.SimulationInstruments))
	// simInstruments := buildSimulatedInstrumentList() // function defined below

	for i, cfg := range a.symbolsConfig.SimulationInstruments {
		simInstruments[i] = &api.InstrumentInfo{
			Token:          cfg.Token,
			Exchange:       cfg.Exchange,
			Symbol:         cfg.Symbol,
			InstrumentType: cfg.InstrumentType,
			Name:           cfg.Name,
			Segment:        cfg.Segment,
			TickSize:       cfg.TickSize,
			LotSize:        cfg.LotSize,
		}
	}
	// Upsert instruments into DB
	for _, info := range simInstruments {
		// Delete stale entry (if any)
		_ = a.DB.Unscoped().
			Where("tradingsymbol = ? AND exchange = ?", info.Symbol, info.Exchange).
			Delete(&db.Instrument{})

		newInstrument := db.Instrument{
			InstrumentToken: uint32(info.Token),
			Exchange:        info.Exchange,
			Tradingsymbol:   info.Symbol,
			InstrumentType:  info.InstrumentType,
			Name:            info.Name,
			Segment:         info.Segment,
			TickSize:        info.TickSize,
			LotSize:         int(info.LotSize),
			LastUpdated:     time.Now(),
		}
		if err := a.DB.Create(&newInstrument).Error; err != nil {
			zap.L().Error("Failed to upsert instrument for simulation", zap.String("symbol", info.Symbol), zap.Error(err))
		}
	}

	go func() {
		if err := simGenerator.SimulateTicks(ctx, simInstruments, a.tickBus, a.Config.Market.SimulationSpeedMultiplier); err != nil {
			zap.L().Error("Simulated market data feed stopped", zap.Error(err))
		}
	}()
	return nil
}

// startRealFeed subscribes to real Zerodha ticks.
func (a *App) startRealFeed(ctx context.Context) error {
	zap.L().Info("Starting REAL market data feed via Zerodha Ticker")

	symbols := a.symbolsConfig.LiveSymbols
	preferredExchanges := []string{"NSE"}
	var instruments []*api.InstrumentInfo

	for _, symbol := range symbols {
		info, err := a.Zerodha.FindInstrumentToken(symbol, preferredExchanges)
		if err != nil {
			zap.L().Warn("Failed to find instrument token, skipping", zap.String("symbol", symbol), zap.Error(err))
			continue
		}
		instruments = append(instruments, info)

		// Insert into DB if not exists
		var existing db.Instrument
		res := a.DB.Where("instrument_token = ?", info.Token).First(&existing)
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
				LastUpdated:     time.Now(),
			}
			if err := a.DB.Create(&newInstrument).Error; err != nil {
				zap.L().Error("Failed to save instrument", zap.String("symbol", info.Symbol), zap.Error(err))
			}
		} else if res.Error != nil {
			zap.L().Error("Error checking instrument", zap.Error(res.Error))
		}
	}
	if len(instruments) == 0 {
		return fmt.Errorf("no valid instruments found")
	}

	// Subscribe (this blocks until ctx is done)
	return a.Zerodha.SubscribeToTicks(instruments, a.tickBus)
}

// serviceAdapter implements runtime.Service using functions.
type serviceAdapter struct {
	name   string
	start  func(ctx context.Context) error
	stop   func(ctx context.Context) error
	health func() map[string]interface{}
}

func (s *serviceAdapter) Name() string { return s.name }

func (s *serviceAdapter) Start(ctx context.Context) error {
	if s.start != nil {
		return s.start(ctx)
	}
	return nil
}

func (s *serviceAdapter) Stop(ctx context.Context) error {
	if s.stop != nil {
		return s.stop(ctx)
	}
	return nil
}

func (s *serviceAdapter) Health() map[string]interface{} {
	if s.health != nil {
		return s.health()
	}
	return nil
}
