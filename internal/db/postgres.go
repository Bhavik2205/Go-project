// internal/db/postgres.go
package db

import (
	"fmt"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"go.uber.org/zap"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

// DBClient represents the PostgreSQL database client
type DBClient struct {
	*gorm.DB
}

// NewPostgresClient initializes and returns a new PostgreSQL client
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

	sqlDB.SetMaxOpenConns(cfg.MaxConnections)
	sqlDB.SetMaxIdleConns(cfg.IdleConnections)
	sqlDB.SetConnMaxLifetime(time.Duration(cfg.ConnectionTimeoutSeconds) * time.Second)

	zap.L().Info("✅ Connected to PostgreSQL database...")
	return &DBClient{DB: db}, nil
}

// AutoMigrate runs database migrations for the given models.
func (c *DBClient) AutoMigrate(models ...interface{}) error {
	zap.L().Info("⚙️ Starting database migrations...")
	if err := c.DB.AutoMigrate(models...); err != nil {
		return fmt.Errorf("failed to auto migrate database: %w", err)
	}
	zap.L().Info("✅ Database migrations completed successfully.")
	return nil
}
