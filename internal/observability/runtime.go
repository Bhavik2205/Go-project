package observability

import (
	"context"
	"runtime"
	"runtime/debug"
	"time"

	"go.uber.org/zap"
)

// StartRuntimeMetricsCollector starts a background goroutine that collects
// runtime metrics every 5 seconds.
func StartRuntimeMetricsCollector(ctx context.Context) {
	go func() {
		defer RecoverPanic("runtime-metrics-collector")
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
	GoroutineCount.Set(float64(runtime.NumGoroutine()))

	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	MemoryHeapAlloc.Set(float64(m.HeapAlloc))
	MemoryHeapInuse.Set(float64(m.HeapInuse))
	MemorySys.Set(float64(m.Sys))
	GCRunsTotal.Set(float64(m.NumGC))
}

// RecoverPanic recovers from a panic, logs it with a full stack trace,
// increments the panic counter, and continues. Use with defer.
func RecoverPanic(component string) {
	if r := recover(); r != nil {
		PanicCounter.Inc()
		stack := debug.Stack()
		zap.L().Error("Panic recovered",
			zap.String("component", component),
			zap.Any("panic", r),
			zap.ByteString("stack", stack),
		)
	}
}
