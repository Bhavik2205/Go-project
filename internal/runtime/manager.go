package runtime

import (
	"context"
	"fmt"
	"sync"

	"go.uber.org/zap"
)

// RuntimeManager manages a collection of services, starting them in registration
// order and stopping them in reverse order.
type RuntimeManager struct {
	services []Service
	mu       sync.RWMutex
	started  bool
}

// NewRuntimeManager creates a new manager.
func NewRuntimeManager() *RuntimeManager {
	return &RuntimeManager{}
}

// Register adds a service to the manager.
func (rm *RuntimeManager) Register(s Service) {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	rm.services = append(rm.services, s)
	zap.L().Debug("Service registered", zap.String("name", s.Name()))
}

// StartAll starts every registered service in order.
func (rm *RuntimeManager) StartAll(ctx context.Context) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	for _, s := range rm.services {
		zap.L().Info("Starting service", zap.String("name", s.Name()))
		if err := s.Start(ctx); err != nil {
			return fmt.Errorf("failed to start %s: %w", s.Name(), err)
		}
		zap.L().Info("Service started", zap.String("name", s.Name()))
	}
	rm.started = true
	return nil
}

// StopAll stops every registered service in reverse order. Errors are logged.
func (rm *RuntimeManager) StopAll(ctx context.Context) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	for i := len(rm.services) - 1; i >= 0; i-- {
		s := rm.services[i]
		zap.L().Info("Stopping service", zap.String("name", s.Name()))
		if err := s.Stop(ctx); err != nil {
			zap.L().Error("Error stopping service", zap.String("name", s.Name()), zap.Error(err))
		} else {
			zap.L().Info("Service stopped", zap.String("name", s.Name()))
		}
	}
	rm.started = false
	return nil
}

// Started returns whether StartAll has been called successfully.
func (rm *RuntimeManager) Started() bool {
	rm.mu.RLock()
	defer rm.mu.RUnlock()
	return rm.started
}

// Health returns health info from all services.
func (rm *RuntimeManager) Health() map[string]interface{} {
	rm.mu.RLock()
	defer rm.mu.RUnlock()
	health := make(map[string]interface{})
	for _, s := range rm.services {
		if h := s.Health(); h != nil {
			health[s.Name()] = h
		}
	}
	return health
}
