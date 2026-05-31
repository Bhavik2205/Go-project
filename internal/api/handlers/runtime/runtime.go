package runtime

import (
	"encoding/json"
	"net/http"
	"runtime"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/audit"
	"github.com/Bhavik2205/ML-Bot/internal/contracts"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/middleware"
	"github.com/Bhavik2205/ML-Bot/internal/realtime"
	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"go.uber.org/zap"
)

type runtimeConfigResponse struct {
	Version         string   `json:"version"`
	Mode            string   `json:"mode"` // "simulation" or "live"
	CandleIntervals []string `json:"candleIntervals"`
	WebSocket       struct {
		Path       string `json:"path"`
		PingPeriod int    `json:"pingPeriodSeconds"`
		PongWait   int    `json:"pongWaitSeconds"`
		WriteWait  int    `json:"writeWaitSeconds"`
		MaxMsgSize int    `json:"maxMsgSizeBytes"`
	} `json:"websocket"`
	Server struct {
		HTTPPort            int `json:"httpPort"`
		MaxRequestBodyBytes int `json:"maxRequestBodyBytes"`
	} `json:"server"`
	Market struct {
		Simulate                  bool    `json:"simulate"`
		SimulationSpeedMultiplier float64 `json:"simulationSpeedMultiplier"`
	} `json:"market"`
	Ingestion struct {
		MarketDataBatchSize       int `json:"marketDataBatchSize"`
		MarketDataFlushIntervalMS int `json:"marketDataFlushIntervalMs"`
		DBWorkerCount             int `json:"dbWorkerCount"`
		WSBroadcastWorkerCount    int `json:"wsBroadcastWorkerCount"`
	} `json:"ingestion"`
}

type metricsResponse struct {
	WebSocketClients map[string]int `json:"websocket_clients"`
	DBLatencyMs      int64          `json:"db_latency_ms"`
	UptimeSeconds    float64        `json:"uptime_seconds"`
	Goroutines       int            `json:"goroutines"`
	MemoryAllocMB    float64        `json:"memory_alloc_mb"`
}

func HandleRuntimeConfig(appCfg *utils.AppConfig) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		userID := middleware.UserIDFromContext(r.Context())
		if userID == 0 {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		requestID := r.Header.Get("X-Request-ID")
		if requestID == "" {
			requestID = "req_" + time.Now().Format("20060102150405.000")
		}

		resp := runtimeConfigResponse{
			Version:         "0.1.0",
			CandleIntervals: appCfg.Candles.Intervals,
			Server: struct {
				HTTPPort            int `json:"httpPort"`
				MaxRequestBodyBytes int `json:"maxRequestBodyBytes"`
			}{
				HTTPPort:            appCfg.Server.HTTPPort,
				MaxRequestBodyBytes: appCfg.Server.MaxRequestBodyBytes,
			},
			Market: struct {
				Simulate                  bool    `json:"simulate"`
				SimulationSpeedMultiplier float64 `json:"simulationSpeedMultiplier"`
			}{
				Simulate:                  appCfg.Market.Simulate,
				SimulationSpeedMultiplier: appCfg.Market.SimulationSpeedMultiplier,
			},
			Ingestion: struct {
				MarketDataBatchSize       int `json:"marketDataBatchSize"`
				MarketDataFlushIntervalMS int `json:"marketDataFlushIntervalMs"`
				DBWorkerCount             int `json:"dbWorkerCount"`
				WSBroadcastWorkerCount    int `json:"wsBroadcastWorkerCount"`
			}{
				MarketDataBatchSize:       appCfg.Ingestion.MarketDataBatchSize,
				MarketDataFlushIntervalMS: appCfg.Ingestion.MarketDataFlushIntervalMS,
				DBWorkerCount:             appCfg.Ingestion.DBWorkerCount,
				WSBroadcastWorkerCount:    appCfg.Ingestion.WSBroadcastWorkerCount,
			},
			WebSocket: struct {
				Path       string `json:"path"`
				PingPeriod int    `json:"pingPeriodSeconds"`
				PongWait   int    `json:"pongWaitSeconds"`
				WriteWait  int    `json:"writeWaitSeconds"`
				MaxMsgSize int    `json:"maxMsgSizeBytes"`
			}{
				Path:       appCfg.Server.WebSocketPath,
				PingPeriod: 45,
				PongWait:   60,
				WriteWait:  10,
				MaxMsgSize: 512 * 1024,
			},
		}

		if appCfg.Market.Simulate {
			resp.Mode = "simulation"
		} else {
			resp.Mode = "live"
		}

		// Add default values if not set (they are already set in utils.LoadAppConfig)
		// No secrets (passwords, tokens, keys) are included.
		// Audit log
		audit.LogEvent(r.Context(),
			"CONFIG_GET",
			"runtime",
			"",
			"READ",
			"SUCCESS",
			map[string]any{
				"user_id": userID,
			},
			"",
		)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if err := json.NewEncoder(w).Encode(contracts.NewSuccess(requestID, resp)); err != nil {
			zap.L().Error("Failed to encode runtime config response", zap.Error(err))
		}
	}
}

// HandleRuntimeMetrics returns runtime metrics (protected)
func HandleRuntimeMetrics(hub *realtime.Hub, dbClient *db.DBClient, startupTime time.Time) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		userID := middleware.UserIDFromContext(r.Context())
		if userID == 0 {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		requestID := r.Header.Get("X-Request-ID")
		if requestID == "" {
			requestID = "req_" + time.Now().Format("20060102150405.000")
		}

		// Count WebSocket clients using getter methods
		tickClients := 0
		if hub.DataIngestor != nil {
			tickClients = hub.DataIngestor.GetWebSocketClientCount()
		}
		candleClients := 0
		if hub.CandleGenerator != nil {
			candleClients = hub.CandleGenerator.GetWebSocketClientCount()
		}
		indicatorClients := 0
		if hub.IndicatorManager != nil {
			indicatorClients = hub.IndicatorManager.GetWebSocketClientCount()
		}
		heatmapClients := hub.GetHeatmapClientCount()

		// Measure DB latency (simple ping)
		start := time.Now()
		var result int
		err := dbClient.DB.Raw("SELECT 1").Scan(&result).Error
		dbLatency := time.Since(start).Milliseconds()
		if err != nil {
			dbLatency = -1
			zap.L().Warn("Failed to measure DB latency", zap.Error(err))
		}

		// Uptime
		uptime := time.Since(startupTime).Seconds()

		// Goroutines
		goroutines := runtime.NumGoroutine()

		// Memory
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		allocMB := float64(m.Alloc) / 1024 / 1024

		metrics := metricsResponse{
			WebSocketClients: map[string]int{
				"ticks":      tickClients,
				"candles":    candleClients,
				"indicators": indicatorClients,
				"heatmap":    heatmapClients,
			},
			DBLatencyMs:   dbLatency,
			UptimeSeconds: uptime,
			Goroutines:    goroutines,
			MemoryAllocMB: allocMB,
		}

		audit.LogEvent(r.Context(),
			"METRICS_GET",
			"runtime",
			"",
			"READ",
			"SUCCESS",
			map[string]any{
				"user_id": userID,
			},
			"",
		)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if err := json.NewEncoder(w).Encode(contracts.NewSuccess(requestID, metrics)); err != nil {
			zap.L().Error("Failed to encode runtime metrics response", zap.Error(err))
		}
	}
}
