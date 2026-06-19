package utils

import (
	"errors"
	"fmt"
	"net"
	"os"
	"time"

	"github.com/redis/go-redis/v9"
	"gopkg.in/yaml.v3"
)

// AppConfig holds the application-wide settings
type AppConfig struct {
	Server struct {
		HTTPPort            int    `yaml:"http_port"`
		MaxRequestBodyBytes int    `yaml:"max_request_body_bytes"` // New field for max request body size
		WebSocketPath       string `yaml:"websocket_path"`
	} `yaml:"server"`
	Log struct {
		Level  string `yaml:"level"`
		Output string `yaml:"output"`
	} `yaml:"log"`
	Ingestion struct {
		MarketDataBatchSize          int `yaml:"market_data_batch_size"`
		MarketDataFlushIntervalMS    int `yaml:"market_data_flush_interval_ms"`
		MaxTickSequenceCacheDuration int `yaml:"max_tick_sequence_cache_duration_s"`
		TickSequenceCleanupInterval  int `yaml:"tick_sequence_cleanup_interval_s"`
		DBWorkerCount                int `yaml:"db_worker_count"`                  // Number of workers for DB writes
		DBFlushChannelSize           int `yaml:"db_flush_channel_size"`            // Size of channel for DB writes
		WSBroadcastWorkerCount       int `yaml:"ws_broadcast_worker_count"`        // Number of workers for WebSocket broadcasting
		WSBroadcastChannelSize       int `yaml:"ws_broadcast_channel_size"`        // Size of channel for WebSocket broadcasting
		RedisReconnectInitialDelayMs int `yaml:"redis_reconnect_initial_delay_ms"` // Initial delay for Redis reconnect
		RedisReconnectMaxDelayMs     int `yaml:"redis_reconnect_max_delay_ms"`     // Max delay for Redis reconnect
		RedisReconnectMaxAttempts    int `yaml:"redis_reconnect_max_attempts"`     // Max attempts for Redis reconnect
		TickIngestionTimeoutMs       int `yaml:"tick_ingestion_timeout_ms"`        // Timeout for tick ingestion to prevent blocking
		DBFlushTimeoutMs             int `yaml:"db_flush_timeout_ms"`              // Timeout for blocking DB send operations to prevent deadlocks
		TickWorkerCount              int `yaml:"tick_worker_count"`
		ProtobufQueueSize            int `yaml:"protobuf_queue_size"`
		EncoderWorkerCount           int `yaml:"encoder_worker_count"`
	} `yaml:"ingestion"`
	Monitor struct {
		BroadcastInterval time.Duration `yaml:"broadcast_interval"`
	} `yaml:"monitor"`
	Candles struct {
		Intervals          []string `yaml:"intervals"` // e.g., ["1m", "5m", "15m", "1h", "1d"]
		GracePeriodMs      int      `yaml:"grace_period_ms"`
		FinalizeIntervalMs int      `yaml:"finalize_interval_ms"`
	}
	Market struct {
		Simulate                  bool    `yaml:"simulate"`
		SimulationSpeedMultiplier float64 `yaml:"simulation_speed_multiplier"`
		TickBus                   string  `yaml:"tick_bus"` // "inprocess" or "redis"
	} `yaml:"market"`
}

// DatabaseConfig holds database connection settings
type DatabaseConfig struct {
	MaxConnections           int `yaml:"max_open_connections"`
	IdleConnections          int `yaml:"max_idle_connections"`
	ConnectionTimeoutSeconds int `yaml:"connection_timeout_seconds"`
	// These will be loaded from environment variables for security
	Host     string
	Port     string
	User     string
	Password string
	DBName   string
}

// RedisConfig holds Redis connection settings
type RedisConfig struct {
	Host     string
	Port     string
	Password string
	DB       int
}

// StrategyConfig holds parameters for various trading strategies (excluding indicator params).
type StrategyConfig struct {
	Intraday struct {
		StopLossPercentage           float64 `yaml:"stop_loss_percentage"`
		TargetProfitPercentage       float64 `yaml:"target_profit_percentage"`
		MaxTradesPerDay              int     `yaml:"max_trades_per_day"`
		MaxLossPerDayPercentage      float64 `yaml:"max_loss_per_day_percentage"`
		TradeSizePercentageOfCapital float64 `yaml:"trade_size_percentage_of_capital"`
	} `yaml:"intraday"`
	Swing struct {
		HoldingPeriodDays     int     `yaml:"holding_period_days"`
		MinReturnPercentage   float64 `yaml:"min_return_percentage"`
		MaxDrawdownPercentage float64 `yaml:"max_drawdown_percentage"`
	} `yaml:"swing"`
	MarketDataTimeframe string `yaml:"market_data_timeframe"`
}

// Define individual indicator config structs with an "Enabled" field.
// This matches the structure expected by indicators_manager.go and indicators/interface.go.
type SMAConfig struct {
	Enabled bool `yaml:"enabled"`
	Period  int  `yaml:"period"`
}

type EMAConfig struct {
	Enabled     bool `yaml:"enabled"`
	ShortPeriod int  `yaml:"short_period"`
	LongPeriod  int  `yaml:"long_period"`
}

type RSIConfig struct {
	Enabled       bool    `yaml:"enabled"`
	Period        int     `yaml:"period"`
	BuyThreshold  float64 `yaml:"buy_threshold"`
	SellThreshold float64 `yaml:"sell_threshold"`
}

type MACDConfig struct {
	Enabled      bool `yaml:"enabled"`
	FastPeriod   int  `yaml:"fast_period"`
	SlowPeriod   int  `yaml:"slow_period"`
	SignalPeriod int  `yaml:"signal_period"`
}

type ATRConfig struct {
	Enabled bool `yaml:"enabled"`
	Period  int  `yaml:"period"`
}

type StochasticConfig struct {
	Enabled bool `yaml:"enabled"`
	KPeriod int  `yaml:"k_period"`
	DPeriod int  `yaml:"d_period"`
}

type BollingerBandsConfig struct {
	Enabled   bool    `yaml:"enabled"`
	Period    int     `yaml:"period"`
	NumStdDev float64 `yaml:"num_std_dev"`
}

type ADXConfig struct {
	Enabled bool `yaml:"enabled"`
	Period  int  `yaml:"period"`
}

// Add OBV and VWAP configs
type OBVConfig struct {
	Enabled bool `yaml:"enabled"`
}

type VWAPConfig struct {
	Enabled bool `yaml:"enabled"`
}

// IndicatorsConfig now embeds these new individual config structs.
type IndicatorsConfig struct {
	OutputWorkerCount       int                  `yaml:"output_worker_count"`        // Number of workers for indicator output processing
	OutputChannelBufferSize int                  `yaml:"output_channel_buffer_size"` // Buffer size for indicator output channel
	SMA                     SMAConfig            `yaml:"sma"`
	EMA                     EMAConfig            `yaml:"ema"`
	RSI                     RSIConfig            `yaml:"rsi"`
	MACD                    MACDConfig           `yaml:"macd"`
	ATR                     ATRConfig            `yaml:"atr"`
	Stochastic              StochasticConfig     `yaml:"stochastic"`
	BollingerBands          BollingerBandsConfig `yaml:"bollinger_bands"`
	ADX                     ADXConfig            `yaml:"adx"`
	OBV                     OBVConfig            `yaml:"obv"`  // New
	VWAP                    VWAPConfig           `yaml:"vwap"` // New
}

// ZerodhaConfig holds Zerodha API connection settings
type ZerodhaConfig struct {
	BaseURL               string `yaml:"base_url"`
	WebSocketURL          string `yaml:"websocket_url"`
	OAuthRedirectURL      string `yaml:"oauth_redirect_url"`
	RequestTimeoutSeconds int    `yaml:"request_timeout_seconds"`
}

// LoadAppConfig loads application configurations from a YAML file
func LoadAppConfig(path string) (*AppConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read app config file %s: %w", path, err)
	}
	var cfg AppConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal app config %s: %w", path, err)
	}

	// Set default values if not explicitly set in config
	if cfg.Ingestion.DBWorkerCount == 0 {
		cfg.Ingestion.DBWorkerCount = 4
	}
	if cfg.Ingestion.DBFlushChannelSize == 0 {
		cfg.Ingestion.DBFlushChannelSize = 100
	}
	if cfg.Ingestion.WSBroadcastWorkerCount == 0 {
		cfg.Ingestion.WSBroadcastWorkerCount = 8
	}
	if cfg.Ingestion.WSBroadcastChannelSize == 0 {
		cfg.Ingestion.WSBroadcastChannelSize = 10000 // A large buffer for high-frequency
	}
	if cfg.Ingestion.RedisReconnectInitialDelayMs == 0 {
		cfg.Ingestion.RedisReconnectInitialDelayMs = 100 // Default to 100ms
	}
	if cfg.Ingestion.RedisReconnectMaxDelayMs == 0 {
		cfg.Ingestion.RedisReconnectMaxDelayMs = 5000 // Default to 5 seconds
	}
	if cfg.Ingestion.RedisReconnectMaxAttempts == 0 {
		cfg.Ingestion.RedisReconnectMaxAttempts = 10 // Default to 10 attempts
	}
	if len(cfg.Candles.Intervals) == 0 {
		cfg.Candles.Intervals = []string{"1s", "5s", "15s", "30s", "1m", "5m", "15m", "30m", "1h"}
	}
	if cfg.Candles.GracePeriodMs == 0 {
		cfg.Candles.GracePeriodMs = 100
	}
	if cfg.Candles.FinalizeIntervalMs == 0 {
		cfg.Candles.FinalizeIntervalMs = 1000
	}
	if cfg.Market.TickBus == "" {
		cfg.Market.TickBus = "inprocess"
	}

	return &cfg, nil
}

// LoadDatabaseConfig loads database configurations from a YAML file and env vars
func LoadDatabaseConfig(path string) (*DatabaseConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read database config file %s: %w", path, err)
	}
	var cfg DatabaseConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal database config %s: %w", path, err)
	}

	// Load sensitive details from environment variables
	cfg.Host = os.Getenv("DB_HOST")
	cfg.Port = os.Getenv("DB_PORT")
	cfg.User = os.Getenv("DB_USER")
	cfg.Password = os.Getenv("DB_PASSWORD")
	cfg.DBName = os.Getenv("DB_NAME")

	if cfg.Host == "" || cfg.Port == "" || cfg.User == "" || cfg.Password == "" || cfg.DBName == "" {
		return nil, errors.New("missing one or more required database environment variables (DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME)")
	}

	if cfg.MaxConnections <= 0 {
		return nil, fmt.Errorf("invalid max_open_connections: %d", cfg.MaxConnections)
	}

	if cfg.IdleConnections < 0 {
		return nil, fmt.Errorf("invalid max_idle_connections: %d", cfg.IdleConnections)
	}

	return &cfg, nil
}

// LoadRedisConfig loads Redis configurations from environment variables
// LoadRedisConfig loads Redis configurations from REDIS_URL environment variable.
func LoadRedisConfig() (*RedisConfig, error) {
	redisURL := os.Getenv("REDIS_URL")
	if redisURL == "" {
		// Fallback to individual REDIS_HOST, REDIS_PORT, REDIS_PASSWORD if REDIS_URL is not set
		host := os.Getenv("REDIS_HOST")
		port := os.Getenv("REDIS_PORT")
		password := os.Getenv("REDIS_PASSWORD")

		if host == "" || port == "" {
			return nil, errors.New("neither REDIS_URL nor REDIS_HOST/REDIS_PORT environment variables are set")
		}
		return &RedisConfig{
			Host:     host,
			Port:     port,
			Password: password,
			DB:       0, // Default DB if not specified
		}, nil
	}

	// Parse the Redis URL
	opts, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse Redis URL: %w", err)
	}

	// Populate RedisConfig fields from parsed options
	cfg := &RedisConfig{
		Password: opts.Password,
		DB:       opts.DB,
	}

	// Split Addr (e.g., "host:port") into Host and Port
	host, port, err := net.SplitHostPort(opts.Addr)
	if err != nil {
		// If SplitHostPort fails, opts.Addr might just be a hostname or IP.
		// In some cases, opts.Addr might only contain the host if the port is default.
		// For robustness, try to assign Addr to Host if it can't be split.
		// However, for redis-cloud, it will always be host:port.
		// Handle this gracefully or return an error if expected to be host:port.
		return nil, fmt.Errorf("failed to split Redis address '%s' into host and port: %w", opts.Addr, err)
	}
	cfg.Host = host
	cfg.Port = port

	return cfg, nil
}

// LoadStrategyConfig loads strategy configurations from a YAML file.
func LoadStrategyConfig(path string) (*StrategyConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read strategy config file %s: %w", path, err)
	}
	var cfg StrategyConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal strategy config %s: %w", path, err)
	}
	return &cfg, nil
}

// LoadIndicatorsConfig loads indicator configurations from a YAML file. (NEW FUNCTION)
func LoadIndicatorsConfig(path string) (*IndicatorsConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read indicators config file %s: %w", path, err)
	}
	var cfg IndicatorsConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal indicators config %s: %w", path, err)
	}
	// Set defaults if not provided
	if cfg.OutputWorkerCount == 0 {
		cfg.OutputWorkerCount = 30
	}
	if cfg.OutputChannelBufferSize == 0 {
		cfg.OutputChannelBufferSize = 5000
	}
	return &cfg, nil
}

// // LoadZerodhaConfig loads Zerodha API configurations from a YAML file.
// func LoadZerodhaConfig(path string) (*ZerodhaConfig, error) {
//     data, err := os.ReadFile(path)
//     if err != nil {
//         return nil, fmt.Errorf("failed to read Zerodha config file %s: %w", path, err)
//     }
//     var cfg ZerodhaConfig
//     if err := yaml.Unmarshal(data, &cfg); err != nil {
//         return nil, fmt.Errorf("failed to unmarshal Zerodha config %s: %w", path, err)
//     }
//     return &cfg, nil
// }
