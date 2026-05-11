package server

import (
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/utils"
)

var (
	serverStartTime = time.Now()
	appConfig       *utils.AppConfig
)

func SetStartupTime(t time.Time) {
	serverStartTime = t
}

func SetAppConfig(cfg *utils.AppConfig) {
	appConfig = cfg
}
