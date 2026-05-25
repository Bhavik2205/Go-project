// internal/audit/audit.go
package audit

import (
	"context"

	"github.com/Bhavik2205/ML-Bot/internal/auth"
	"github.com/Bhavik2205/ML-Bot/internal/middleware"
	"go.uber.org/zap"
)

// Logger is the global audit logger (set at startup).
var Logger *zap.Logger

// Init sets the logger for audit package.
func Init(logger *zap.Logger) {
	Logger = logger
}

// LogEvent logs a structured audit event.
// action: e.g., "login", "logout", "order_placed", "order_cancelled", "broker_connected"
// resource: e.g., "user", "order", "broker"
// details: any additional key-value pairs.
func LogEvent(ctx context.Context, action, resource string, status string, details ...zap.Field) {
	fields := []zap.Field{
		zap.String("audit", "true"),
		zap.String("action", action),
		zap.String("resource", resource),
		zap.String("status", status), // "success" or "failure"
		zap.String("request_id", middleware.RequestIDFromContext(ctx)),
	}
	userID, _ := auth.GetUserIDFromContext(ctx)
	if userID != 0 {
		fields = append(fields, zap.Uint64("user_id", userID))
	}
	fields = append(fields, details...)
	Logger.Info("audit event", fields...)
}
