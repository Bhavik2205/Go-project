// internal/utils/config.go
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
		HTTPPort      int    `yaml:"http_port"`
		WebSocketPath string `yaml:"websocket_path"`
	} `yaml:"server"`
	Log struct {
		Level  string `yaml:"level"`
		Output string `yaml:"output"`
	} `yaml:"log"`
	Ingestion struct {
		MarketDataBatchSize          int `yaml:"market_data_batch_size"`
		MarketDataFlushIntervalMS    int `yaml:"market_data_flush_interval_ms"`
		MaxTickSequenceCacheDuration int `yaml:"max_tick_sequence_cache_duration"`
		TickSequenceCleanupInterval  int `yaml:"tick_sequence_cleanup_interval_s"`
	} `yaml:"ingestion"`
	Monitor struct { // ADD THIS
		BroadcastInterval time.Duration `yaml:"broadcast_interval"`
	} `yaml:"monitor"`
	Candles struct {
		Intervals []string `yaml:"intervals"` // e.g., ["1m", "5m", "15m", "1h", "1d"]
	}
}

// DatabaseConfig holds database connection settings
type DatabaseConfig struct {
	MaxConnections           int `yaml:"max_connections"`
	IdleConnections          int `yaml:"idle_connections"`
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

// IndicatorsConfig holds default parameters for various technical indicators. (NEW STRUCT)
type IndicatorsConfig struct {
	SMA struct {
		Period int `yaml:"period"`
	} `yaml:"sma"`
	EMA struct {
		ShortPeriod int `yaml:"short_period"`
		LongPeriod  int `yaml:"long_period"`
	} `yaml:"ema"`
	RSI struct {
		Period        int     `yaml:"period"`
		BuyThreshold  float64 `yaml:"buy_threshold"`
		SellThreshold float64 `yaml:"sell_threshold"`
	} `yaml:"rsi"`
	MACD struct {
		FastPeriod   int `yaml:"fast_period"`
		SlowPeriod   int `yaml:"slow_period"`
		SignalPeriod int `yaml:"signal_period"`
	} `yaml:"macd"`
	ATR struct {
		Period int `yaml:"period"`
	} `yaml:"atr"`
	Stochastic struct {
		KPeriod int `yaml:"k_period"`
		DPeriod int `yaml:"d_period"`
	} `yaml:"stochastic"`
	BollingerBands struct {
		Period    int     `yaml:"period"`
		NumStdDev float64 `yaml:"num_std_dev"`
	} `yaml:"bollinger_bands"`
	ADX struct {
		Period int `yaml:"period"`
	} `yaml:"adx"`
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

	// IMPORTANT: For Redis Cloud, typically the `rediss://` or `redis://` scheme
	// implies TLS/SSL. If using `rediss://`, the `go-redis` client will
	// automatically handle TLS. If just `redis://` and Redis Cloud requires TLS,
	// you might need to manually set opts.TLSConfig in NewRedisClient.
	// However, sticking to the URL and let ParseURL handle it is generally best.
	// Since we are not passing 'opts' directly, we need to ensure RedisClient
	// can handle TLS if necessary. For now, assume it's handled by redis.Options.

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
