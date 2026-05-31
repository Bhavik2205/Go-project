package audit

import (
	"context"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/db"
	"gorm.io/gorm"
)

type Repository struct {
	db *gorm.DB
}

func NewRepository(db *gorm.DB) *Repository {
	return &Repository{db: db}
}

func (r *Repository) Log(ctx context.Context, event *db.AuditEvent) error {
	// Set timestamps if not set
	if event.CreatedAt.IsZero() {
		event.CreatedAt = time.Now()
	}
	return r.db.WithContext(ctx).Create(event).Error
}
