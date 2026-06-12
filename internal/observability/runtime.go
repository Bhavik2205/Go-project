package observability

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"runtime/pprof"
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

// lastNumGC tracks the previous GC count so only new pauses are recorded.
var lastNumGC uint32

func collectRuntimeMetrics() {
	GoroutineCount.Set(float64(runtime.NumGoroutine()))

	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	MemoryHeapAlloc.Set(float64(m.HeapAlloc))
	MemoryHeapInuse.Set(float64(m.HeapInuse))
	MemorySys.Set(float64(m.Sys))
	GCRunsTotal.Set(float64(m.NumGC))

	// Record new GC pause durations. PauseNs is a circular buffer of 256 entries.
	if m.NumGC > lastNumGC {
		newRuns := m.NumGC - lastNumGC
		if newRuns > 256 {
			newRuns = 256
		}
		for i := uint32(0); i < newRuns; i++ {
			idx := (m.NumGC - 1 - i) % 256
			if ns := m.PauseNs[idx]; ns > 0 {
				GCPauseSeconds.Observe(float64(ns) / 1e9)
			}
		}
		lastNumGC = m.NumGC
	}
}

// RecoverPanic recovers from a panic, logs the full stack trace, increments
// the panic counter, and writes goroutine + heap profiles to /tmp as crash
// artifacts for post-mortem analysis.
func RecoverPanic(component string) {
	if r := recover(); r != nil {
		PanicCounter.Inc()
		stack := debug.Stack()
		zap.L().Error("Panic recovered",
			zap.String("component", component),
			zap.Any("panic", r),
			zap.ByteString("stack", stack),
		)
		writeCrashArtifacts(component)
	}
}

// writeCrashArtifacts dumps goroutine and heap profiles to /tmp on panic.
// Files are named by component and timestamp so multiple panics don't overwrite each other.
func writeCrashArtifacts(component string) {
	ts := time.Now().Format("20060102-150405")
	base := fmt.Sprintf("/tmp/crash-%s-%s", component, ts)

	if f, err := os.Create(base + ".goroutines"); err == nil {
		_ = pprof.Lookup("goroutine").WriteTo(f, 1)
		f.Close()
	}
	if f, err := os.Create(base + ".heap"); err == nil {
		_ = pprof.Lookup("heap").WriteTo(f, 0)
		f.Close()
	}
	zap.L().Info("Crash artifacts written", zap.String("base_path", base))
}
