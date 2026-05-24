// cmd/server/main.go
package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"github.com/Bhavik2205/ML-Bot/internal/app"
	"github.com/Bhavik2205/ML-Bot/internal/audit"
	"github.com/Bhavik2205/ML-Bot/internal/auth"
	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"github.com/joho/godotenv"
	"go.uber.org/zap"
)

func main() {
	// --- Early logger (temporary, will be replaced after config loads) ---
	utils.InitLogger("info", "app.log")
	defer func() {
		_ = zap.L().Sync()
	}()

	// --- Load environment ---
	if err := godotenv.Load(); err != nil {
		zap.L().Warn("⚠️ .env file not found", zap.Error(err))
	} else {
		zap.L().Info("✅ .env file loaded")
	}

	// Validate required env vars (DB, Redis, JWT)
	if err := utils.ValidateRequiredEnv(); err != nil {
		zap.L().Fatal("Environment validation failed", zap.Error(err))
	}
	auth.MustLoadJWTSecret()
	zap.L().Info("✅ JWT secret validated")

	// --- Load configurations ---
	appCfg, err := utils.LoadAppConfig("configs/app.yaml")
	if err != nil {
		zap.L().Fatal("Failed to load app config", zap.Error(err))
	}
	dbCfg, err := utils.LoadDatabaseConfig("configs/database.yaml")
	if err != nil {
		zap.L().Fatal("Failed to load database config", zap.Error(err))
	}
	redisCfg, err := utils.LoadRedisConfig()
	if err != nil {
		zap.L().Fatal("Failed to load Redis config", zap.Error(err))
	}
	indicatorsCfg, err := utils.LoadIndicatorsConfig("configs/indicators.yaml")
	if err != nil {
		zap.L().Fatal("Failed to load indicators config", zap.Error(err))
	}
	symbolsCfg, err := utils.LoadSymbolsConfig("configs/symbols.yaml")
	if err != nil {
		zap.L().Fatal("Failed to load symbols config", zap.Error(err))
	}

	// Re-initialise logger with the final configuration
	utils.InitLogger(appCfg.Log.Level, appCfg.Log.Output)
	zap.L().Info("📦 ML-Bot service starting up...")

	// ---- Initialize audit logger ----
	audit.Init(zap.L())

	// --- Create the application container ---
	application, err := app.New(appCfg, dbCfg, redisCfg, indicatorsCfg, symbolsCfg)
	if err != nil {
		zap.L().Fatal("Failed to initialise application", zap.Error(err))
	}

	// --- Graceful shutdown context ---
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// --- Start all services ---
	if err := application.Start(ctx); err != nil {
		zap.L().Fatal("Failed to start application", zap.Error(err))
	}

	// --- Wait for shutdown signal ---
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigChan
		zap.L().Info("Received shutdown signal", zap.String("signal", sig.String()))
		cancel()
	}()

	<-ctx.Done()
	zap.L().Info("Shutting down ML-Bot service gracefully...")
	if err := application.Stop(context.Background()); err != nil {
		zap.L().Error("Error during shutdown", zap.Error(err))
	}
	zap.L().Info("ML-Bot service stopped.")
}
