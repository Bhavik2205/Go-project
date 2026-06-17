// internal/db/postgres.go
package db

import (
	"context"
	"fmt"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"go.uber.org/zap"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

// DBClient represents the PostgreSQL database client.
type DBClient struct {
	*gorm.DB
	Pool *pgxpool.Pool
}

// NewPostgresClient initializes and returns a new PostgreSQL client.
func NewPostgresClient(cfg *utils.DatabaseConfig) (*DBClient, error) {
	dsn := fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=disable TimeZone=Asia/Kolkata",
		cfg.Host, cfg.Port, cfg.User, cfg.Password, cfg.DBName)

	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	sqlDB, err := db.DB()
	if err != nil {
		return nil, fmt.Errorf("failed to get sql.DB from gorm: %w", err)
	}

	// Safety defaults
	maxConns := cfg.MaxConnections
	if maxConns <= 0 {
		maxConns = 10
		zap.L().Warn("MaxConnections not set, using default", zap.Int("default", maxConns))
	}
	idleConns := cfg.IdleConnections
	if idleConns <= 0 {
		idleConns = 2
		zap.L().Warn("IdleConnections not set, using default", zap.Int("default", idleConns))
	}
	connTimeout := cfg.ConnectionTimeoutSeconds
	if connTimeout <= 0 {
		connTimeout = 30
		zap.L().Warn("ConnectionTimeoutSeconds not set, using default", zap.Int("default", connTimeout))
	}

	sqlDB.SetMaxOpenConns(maxConns)
	sqlDB.SetMaxIdleConns(idleConns)
	sqlDB.SetConnMaxLifetime(time.Duration(connTimeout) * time.Second)

	poolCfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to parse pgx pool config: %w", err)
	}
	poolCfg.MaxConns = int32(maxConns)
	poolCfg.MinConns = int32(idleConns)
	poolCfg.MaxConnLifetime = time.Duration(connTimeout) * time.Second

	zap.L().Info("PostgreSQL pool settings",
		zap.Int32("max_conns", poolCfg.MaxConns),
		zap.Int32("min_conns", poolCfg.MinConns),
		zap.Duration("max_lifetime", poolCfg.MaxConnLifetime),
	)

	pool, err := pgxpool.NewWithConfig(context.Background(), poolCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create pgx pool: %w", err)
	}
	if err := pool.Ping(context.Background()); err != nil {
		pool.Close()
		return nil, fmt.Errorf("failed to ping pgx pool: %w", err)
	}

	zap.L().Info("Connected to PostgreSQL database")
	return &DBClient{DB: db, Pool: pool}, nil
}

// CreateTempStageTable creates a temporary staging table on a given connection.
// The table persists for the lifetime of the connection.
func (c *DBClient) CreateTempStageTable(ctx context.Context, conn *pgx.Conn, stageName string) error {
	_, err := conn.Exec(ctx, fmt.Sprintf(
		"CREATE TEMP TABLE IF NOT EXISTS %s (LIKE market_data INCLUDING DEFAULTS) ON COMMIT PRESERVE ROWS",
		stageName,
	))
	if err != nil {
		return fmt.Errorf("create temp stage %s: %w", stageName, err)
	}
	return nil
}

// CopyMarketDataBatchWithConn copies rows into a staging table and merges into market_data
// using a provided connection (the worker's persistent connection).
// The staging table must already exist on this connection.
func (c *DBClient) CopyMarketDataBatchWithConn(ctx context.Context, conn *pgx.Conn, stageName string, rows []MarketData) error {
	if len(rows) == 0 {
		return nil
	}
	if conn == nil {
		return fmt.Errorf("connection is nil")
	}

	tx, err := conn.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin copy tx: %w", err)
	}
	defer func() {
		_ = tx.Rollback(context.Background())
	}()

	// Truncate the staging table so we start fresh for this batch
	if _, err := tx.Exec(ctx, fmt.Sprintf("TRUNCATE %s", stageName)); err != nil {
		return fmt.Errorf("truncate stage: %w", err)
	}

	columns := []string{
		"instrument_token", "timestamp", "tick_sequence_id",
		"last_price", "last_traded_quantity", "volume", "average_trade_price", "net_change",
		"open", "high", "low", "close", "open_interest",
		"bid_price1", "bid_quantity1", "bid_orders1",
		"bid_price2", "bid_quantity2", "bid_orders2",
		"bid_price3", "bid_quantity3", "bid_orders3",
		"bid_price4", "bid_quantity4", "bid_orders4",
		"bid_price5", "bid_quantity5", "bid_orders5",
		"ask_price1", "ask_quantity1", "ask_orders1",
		"ask_price2", "ask_quantity2", "ask_orders2",
		"ask_price3", "ask_quantity3", "ask_orders3",
		"ask_price4", "ask_quantity4", "ask_orders4",
		"ask_price5", "ask_quantity5", "ask_orders5",
		"total_buy_quantity", "total_sell_quantity", "data_source",
	}

	copyCount, err := tx.CopyFrom(
		ctx,
		pgx.Identifier{stageName},
		columns,
		pgx.CopyFromSlice(len(rows), func(i int) ([]any, error) {
			row := rows[i]
			return []any{
				row.InstrumentToken, row.Timestamp, row.TickSequenceID,
				row.LastPrice, row.LastTradedQuantity, row.Volume, row.AverageTradePrice, row.NetChange,
				row.Open, row.High, row.Low, row.Close, row.OpenInterest,
				row.BidPrice1, row.BidQuantity1, row.BidOrders1,
				row.BidPrice2, row.BidQuantity2, row.BidOrders2,
				row.BidPrice3, row.BidQuantity3, row.BidOrders3,
				row.BidPrice4, row.BidQuantity4, row.BidOrders4,
				row.BidPrice5, row.BidQuantity5, row.BidOrders5,
				row.AskPrice1, row.AskQuantity1, row.AskOrders1,
				row.AskPrice2, row.AskQuantity2, row.AskOrders2,
				row.AskPrice3, row.AskQuantity3, row.AskOrders3,
				row.AskPrice4, row.AskQuantity4, row.AskOrders4,
				row.AskPrice5, row.AskQuantity5, row.AskOrders5,
				row.TotalBuyQuantity, row.TotalSellQuantity, row.DataSource,
			}, nil
		}),
	)
	if err != nil {
		return fmt.Errorf("copy into temp stage: %w", err)
	}

	if _, err := tx.Exec(ctx, fmt.Sprintf(`
		INSERT INTO market_data
		SELECT * FROM %s
		ON CONFLICT (instrument_token, timestamp, tick_sequence_id) DO NOTHING
	`, stageName)); err != nil {
		return fmt.Errorf("merge staged market_data: %w", err)
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit copy tx: %w", err)
	}

	zap.L().Debug("pgx CopyFrom merged market_data batch",
		zap.String("stage", stageName),
		zap.Int64("rows", copyCount))
	return nil
}

// CopyMarketDataBatch is deprecated but kept for API compatibility.
// Use CopyMarketDataBatchWithConn for worker persistent connections.
func (c *DBClient) CopyMarketDataBatch(ctx context.Context, stageName string, rows []MarketData) error {
	// This would acquire a new connection, breaking temp table reuse.
	// We'll implement a fallback that creates the temp table on the fly (per batch) to avoid panic.
	if len(rows) == 0 {
		return nil
	}
	if c.Pool == nil {
		return fmt.Errorf("pgx pool is not initialized")
	}

	tx, err := c.Pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin copy tx: %w", err)
	}
	defer func() {
		_ = tx.Rollback(context.Background())
	}()

	// Create temp table inside this transaction
	if _, err := tx.Exec(ctx, fmt.Sprintf(
		"CREATE TEMP TABLE %s (LIKE market_data INCLUDING DEFAULTS) ON COMMIT DROP",
		stageName,
	)); err != nil {
		return fmt.Errorf("create temp stage: %w", err)
	}

	// COPY and INSERT as before
	columns := []string{
		"instrument_token", "timestamp", "tick_sequence_id",
		"last_price", "last_traded_quantity", "volume", "average_trade_price", "net_change",
		"open", "high", "low", "close", "open_interest",
		"bid_price1", "bid_quantity1", "bid_orders1",
		"bid_price2", "bid_quantity2", "bid_orders2",
		"bid_price3", "bid_quantity3", "bid_orders3",
		"bid_price4", "bid_quantity4", "bid_orders4",
		"bid_price5", "bid_quantity5", "bid_orders5",
		"ask_price1", "ask_quantity1", "ask_orders1",
		"ask_price2", "ask_quantity2", "ask_orders2",
		"ask_price3", "ask_quantity3", "ask_orders3",
		"ask_price4", "ask_quantity4", "ask_orders4",
		"ask_price5", "ask_quantity5", "ask_orders5",
		"total_buy_quantity", "total_sell_quantity", "data_source",
	}

	copyCount, err := tx.CopyFrom(
		ctx,
		pgx.Identifier{stageName},
		columns,
		pgx.CopyFromSlice(len(rows), func(i int) ([]any, error) {
			row := rows[i]
			return []any{
				row.InstrumentToken, row.Timestamp, row.TickSequenceID,
				row.LastPrice, row.LastTradedQuantity, row.Volume, row.AverageTradePrice, row.NetChange,
				row.Open, row.High, row.Low, row.Close, row.OpenInterest,
				row.BidPrice1, row.BidQuantity1, row.BidOrders1,
				row.BidPrice2, row.BidQuantity2, row.BidOrders2,
				row.BidPrice3, row.BidQuantity3, row.BidOrders3,
				row.BidPrice4, row.BidQuantity4, row.BidOrders4,
				row.BidPrice5, row.BidQuantity5, row.BidOrders5,
				row.AskPrice1, row.AskQuantity1, row.AskOrders1,
				row.AskPrice2, row.AskQuantity2, row.AskOrders2,
				row.AskPrice3, row.AskQuantity3, row.AskOrders3,
				row.AskPrice4, row.AskQuantity4, row.AskOrders4,
				row.AskPrice5, row.AskQuantity5, row.AskOrders5,
				row.TotalBuyQuantity, row.TotalSellQuantity, row.DataSource,
			}, nil
		}),
	)
	if err != nil {
		return fmt.Errorf("copy into temp stage: %w", err)
	}

	if _, err := tx.Exec(ctx, fmt.Sprintf(`
		INSERT INTO market_data
		SELECT * FROM %s
		ON CONFLICT (instrument_token, timestamp, tick_sequence_id) DO NOTHING
	`, stageName)); err != nil {
		return fmt.Errorf("merge staged market_data: %w", err)
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit copy tx: %w", err)
	}

	zap.L().Debug("pgx CopyFrom merged market_data batch (fallback)",
		zap.String("stage", stageName),
		zap.Int64("rows", copyCount))
	return nil
}

// AutoMigrate runs database migrations for the given models.
func (c *DBClient) AutoMigrate(models ...interface{}) error {
	zap.L().Info("Starting database migrations")
	if err := c.DB.AutoMigrate(models...); err != nil {
		return fmt.Errorf("failed to auto migrate database: %w", err)
	}
	zap.L().Info("Database migrations completed successfully")
	return nil
}

// Close closes the underlying database connections.
func (c *DBClient) Close() error {
	var poolErr error
	if c.Pool != nil {
		c.Pool.Close()
	}

	sqlDB, err := c.DB.DB()
	if err != nil {
		return err
	}
	if err := sqlDB.Close(); err != nil {
		poolErr = err
	}
	return poolErr
}
