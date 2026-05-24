package runtime

import "context"

// Service defines the lifecycle methods for any managed component.
type Service interface {
	// Name returns a human‑readable identifier.
	Name() string

	// Start initialises the service and runs its background goroutines.
	Start(ctx context.Context) error

	// Stop gracefully shuts down the service.
	Stop(ctx context.Context) error

	// Health returns a map of health indicators (optional).
	Health() map[string]interface{}
}
