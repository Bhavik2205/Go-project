// internal/audit/audit.go
package audit

import (
	"context"
	"encoding/json"

	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/middleware"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

// Logger is the global audit logger (set at startup).
var (
	logger *zap.Logger
	repo   *Repository
)

// Init sets the logger for audit package.
func Init(log *zap.Logger, db *gorm.DB) {
	logger = log
	repo = NewRepository(db)
}

// LogEvent logs a structured audit event.
// action: e.g., "login", "logout", "order_placed", "order_cancelled", "broker_connected"
// resource: e.g., "user", "order", "broker"
// details: any additional key-value pairs.
func LogEvent(ctx context.Context, eventType, resourceType, resourceID, action, status string, metadata map[string]any, errMsg string) {
	userID := middleware.UserIDFromContext(ctx)
	var userIDPtr *uint
	if userID != 0 {
		userIDPtr = &userID
	}

	metaJSON, _ := json.Marshal(metadata)

	auditEvent := &db.AuditEvent{
		UserID:       userIDPtr,
		EventType:    eventType,
		ResourceType: resourceType,
		ResourceID:   resourceID,
		Action:       action,
		Status:       status,
		IPAddress:    middleware.IPFromContext(ctx), // you'd need to add IP extraction
		UserAgent:    middleware.UserAgentFromContext(ctx),
		RequestID:    middleware.RequestIDFromContext(ctx),
		Metadata:     metaJSON,
		ErrorMessage: errMsg,
	}

	// Async insert to avoid blocking
	go func() {
		if err := repo.Log(context.Background(), auditEvent); err != nil {
			logger.Error("Failed to write audit event to DB", zap.Error(err))
		}
	}()

	// Also log to zap (for debugging)
	logger.Info("audit event",
		zap.String("event_type", eventType),
		zap.Uint("user_id", userID),
		zap.String("action", action),
		zap.String("status", status),
	)
}
