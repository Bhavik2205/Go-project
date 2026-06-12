package observability

import (
	"context"
	"runtime"
	"time"

	"go.uber.org/zap"
)

// StartRuntimeMetricsCollector starts a background goroutine that collects
// runtime metrics every 5 seconds.
func StartRuntimeMetricsCollector(ctx context.Context) {
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				collectRuntimeMetrics()
			case <-ctx.Done():
				zap.L().Info("Runtime metrics collector stopped")
				return
			}
		}
	}()
}

func collectRuntimeMetrics() {
	// Goroutines
	GoroutineCount.Set(float64(runtime.NumGoroutine()))

	// Memory stats
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	MemoryHeapAlloc.Set(float64(m.HeapAlloc))
	MemoryHeapInuse.Set(float64(m.HeapInuse))
	MemorySys.Set(float64(m.Sys))
	GCRunsTotal.Set(float64(m.NumGC))
}

// RecoverPanic recovers from a panic, logs it, and increments the panic counter.
// Use with defer at the start of critical goroutines.
//
// Example:
//   func worker() {
//       defer observability.RecoverPanic("worker-name")
//       // ... work ...
//   }
func RecoverPanic(component string) {
	if r := recover(); r != nil {
		PanicCounter.Inc()
		zap.L().Error("Panic recovered", zap.String("component", component), zap.Any("panic", r))
	}
}
