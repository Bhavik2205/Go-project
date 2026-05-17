package server

import (
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/auth"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/gorilla/mux"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

// BuildTestRouter builds a minimal router for handler tests without real DB/Redis/Zerodha.
func BuildTestRouter() http.Handler {
	router := mux.NewRouter()
	router.Use(enableCORS)
	router.Use(recoverMiddleware)
	// Handle OPTIONS preflight for all routes
	router.Methods(http.MethodOptions).HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	registerVersionedRoutes(router)
	return router
}

// NowForTest returns the current time for use in test setup.
func NowForTest() time.Time { return time.Now() }

// TestAccessToken generates a valid access token for the given userID in tests.
func TestAccessToken(t *testing.T, userID uint) string {
	t.Helper()
	if os.Getenv("JWT_SECRET") == "" {
		os.Setenv("JWT_SECRET", "test-secret-32-bytes-long-enough!")
	}
	tok, err := auth.GenerateAccessToken(userID)
	if err != nil {
		t.Fatalf("TestAccessToken: %v", err)
	}
	return tok
}

// NewTestSQLiteDB creates an in-memory SQLite DB with User table for handler tests.
func NewTestSQLiteDB(t *testing.T) *db.DBClient {
	t.Helper()
	gdb, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Silent),
	})
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	if err := gdb.AutoMigrate(&db.User{}); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	return &db.DBClient{DB: gdb}
}
