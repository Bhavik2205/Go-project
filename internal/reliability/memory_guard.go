package reliability

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/observability"
	"go.uber.org/zap"
)

// MemoryGuard monitors heap memory usage and triggers warnings/degradation.
type MemoryGuard struct {
	// highWaterMark is updated via CAS so it remains correct even if multiple
	// goroutines ever observe a new peak concurrently.
	highWaterMark atomic.Uint64

	// Callbacks for escalation
	onWarning         func(msg string)
	onCritical        func(msg string)
	onDegradation     func(msg string)
	onDegradationHook func()
	onRestoreHook     func()
	stopCh            chan struct{}
	once              sync.Once
}

type MemoryGuardConfig struct {
	WarningPercent     float64       // 0.8 = 80%
	CriticalPercent    float64       // 0.9
	DegradationPercent float64       // 0.95
	CheckInterval      time.Duration // e.g., 5s
	OnWarning          func(msg string)
	OnCritical         func(msg string)
	OnDegradation      func(msg string)
}

func NewMemoryGuard(cfg MemoryGuardConfig) *MemoryGuard {
	if cfg.CheckInterval == 0 {
		cfg.CheckInterval = 5 * time.Second
	}
	if cfg.WarningPercent == 0 {
		cfg.WarningPercent = 0.80
	}
	if cfg.CriticalPercent == 0 {
		cfg.CriticalPercent = 0.90
	}
	if cfg.DegradationPercent == 0 {
		cfg.DegradationPercent = 0.95
	}
	g := &MemoryGuard{
		stopCh: make(chan struct{}),
	}
	g.onWarning = cfg.OnWarning
	g.onCritical = cfg.OnCritical
	g.onDegradation = cfg.OnDegradation

	// Set guard status to running
	observability.MemoryGuardStatus.Set(1)

	go g.run(cfg)
	return g
}

func (g *MemoryGuard) run(cfg MemoryGuardConfig) {
	ticker := time.NewTicker(cfg.CheckInterval)
	defer ticker.Stop()

	var lastDegradationTriggered bool

	for {
		select {
		case <-ticker.C:
			var m runtime.MemStats
			runtime.ReadMemStats(&m)

			sys := m.Sys
			heapSys := m.HeapSys
			heapAlloc := m.HeapAlloc

			for {
				old := g.highWaterMark.Load()
				if heapAlloc <= old {
					break
				}
				if g.highWaterMark.CompareAndSwap(old, heapAlloc) {
					observability.MemoryHighWaterMark.Set(float64(heapAlloc))
					break
				}
			}

			heapBudget := heapSys
			if heapBudget == 0 {
				heapBudget = sys
			}
			warning := uint64(float64(heapBudget) * cfg.WarningPercent)
			critical := uint64(float64(heapBudget) * cfg.CriticalPercent)
			degradation := uint64(float64(heapBudget) * cfg.DegradationPercent)

			// Update Prometheus gauges (already exist in observability)
			observability.MemoryHeapAlloc.Set(float64(heapAlloc))
			observability.MemorySys.Set(float64(sys))

			// Update reliability-specific metrics
			var usagePct float64
			if heapBudget > 0 {
				usagePct = float64(heapAlloc) / float64(heapBudget) * 100
			}
			observability.MemoryUsagePercent.Set(usagePct)
			observability.ReliabilityLastCheck.Set(float64(time.Now().Unix()))

			// Check levels
			if heapAlloc >= degradation && !lastDegradationTriggered {
				if g.onDegradation != nil {
					g.onDegradation("heap memory above degradation threshold, shedding non‑critical load")
				}
				if g.onDegradationHook != nil {
					g.onDegradationHook()
				}
				lastDegradationTriggered = true
			} else if heapAlloc >= critical && g.onCritical != nil {
				g.onCritical("heap memory above critical threshold")
				lastDegradationTriggered = false
			} else if heapAlloc >= warning && g.onWarning != nil {
				g.onWarning("heap memory above warning threshold")
				lastDegradationTriggered = false
			} else {
				if lastDegradationTriggered && g.onRestoreHook != nil {
					g.onRestoreHook()
				}
				lastDegradationTriggered = false
			}

			zap.L().Debug("memory stats",
				zap.Uint64("heap_alloc_mb", heapAlloc/1024/1024),
				zap.Uint64("heap_sys_mb", heapSys/1024/1024),
				zap.Uint64("sys_mb", sys/1024/1024),
				zap.Float64("usage_pct", usagePct),
			)
		case <-g.stopCh:
			observability.MemoryGuardStatus.Set(0)
			return
		}
	}
}

// SetLoadSheddingHooks wires ResourceBudget callbacks into the guard.
func (g *MemoryGuard) SetLoadSheddingHooks(shed func(), restore func()) {
	g.onDegradationHook = shed
	g.onRestoreHook = restore
}

func (g *MemoryGuard) Stop() {
	g.once.Do(func() { close(g.stopCh) })
}

func (g *MemoryGuard) HighWaterMarkMB() uint64 {
	return g.highWaterMark.Load() / 1024 / 1024
}
