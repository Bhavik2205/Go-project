package httpapi

import (
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/cache"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/realtime"
	"github.com/Bhavik2205/ML-Bot/internal/utils"
)

// HTTPDeps contains all dependencies needed by REST handlers.
type HTTPDeps struct {
	DBClient      *db.DBClient
	RedisClient   *cache.RedisClient
	ZerodhaClient api.ZerodhaClientInterface // interface defined in api/zerodha.go
	AppConfig     *utils.AppConfig
	StartupTime   time.Time
	Hub           *realtime.Hub
}
