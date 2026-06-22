package reliability

import (
	"sync"
	"time"

	"go.uber.org/zap"
)

// ResourceBudget orchestrates all guards and provides a single entry point.
type ResourceBudget struct {
	memoryGuard    *MemoryGuard
	goroutineGuard *GoroutineGuard
	queueGuard     *QueueGuard

	mu sync.RWMutex

	// callbacks for load shedding
	shedLoadFunc func()
	restoreFunc  func()

	started time.Time
}

type ResourceBudgetConfig struct {
	HeapBudgetBytes          uint64
	MemoryWarningPercent     float64
	MemoryCriticalPercent    float64
	MemoryDegradationPercent float64
	GoroutineMaxGrowthRate   float64 // max percent growth per check interval before spike fires
	QueueThresholds          map[string]float64
	CheckInterval            struct {
		Memory    string // parsed to duration
		Goroutine string
		Queue     string
	}
	OnMemoryWarning     func(msg string)
	OnMemoryCritical    func(msg string)
	OnMemoryDegradation func(msg string)
	OnGoroutineSpike    func(pctGrowth float64)
	OnQueueWarning      func(name string, usage float64)
}

// DefaultConfig returns sensible production defaults.
func DefaultConfig() ResourceBudgetConfig {
	return ResourceBudgetConfig{
		HeapBudgetBytes:          512 * 1024 * 1024,
		MemoryWarningPercent:     0.88,
		MemoryCriticalPercent:    0.93,
		MemoryDegradationPercent: 0.97,
		GoroutineMaxGrowthRate:   10.0, // fire spike if goroutine count grows >10% per check interval
		QueueThresholds: map[string]float64{
			"tick_broadcast":   0.80,
			"db_flush":         0.80,
			"candle_flush":     0.80,
			"indicator_output": 0.80,
		},
		OnMemoryWarning:     defaultMemoryWarning,
		OnMemoryCritical:    defaultMemoryCritical,
		OnMemoryDegradation: defaultMemoryDegradation,
		OnGoroutineSpike:    defaultGoroutineSpike,
		OnQueueWarning:      defaultQueueWarning,
	}
}

func defaultMemoryWarning(msg string) {
	zap.L().Warn("memory warning", zap.String("msg", msg))
}

func defaultMemoryCritical(msg string) {
	zap.L().Error("memory critical", zap.String("msg", msg))
}

func defaultMemoryDegradation(msg string) {
	zap.L().Error("memory degradation – load shedding activated", zap.String("msg", msg))
}

func defaultGoroutineSpike(pctGrowth float64) {
	zap.L().Warn("goroutine spike", zap.Float64("pct_growth", pctGrowth))
}

func defaultQueueWarning(name string, usage float64) {
	zap.L().Warn("queue high usage", zap.String("queue", name), zap.Float64("usage_pct", usage*100))
}

// NewResourceBudget creates and starts all guards.
func NewResourceBudget(cfg ResourceBudgetConfig) *ResourceBudget {
	rb := &ResourceBudget{
		started: time.Now(),
	}

	// Memory guard
	memCfg := MemoryGuardConfig{
		WarningPercent:     cfg.MemoryWarningPercent,
		CriticalPercent:    cfg.MemoryCriticalPercent,
		DegradationPercent: cfg.MemoryDegradationPercent,
		CheckInterval:      parseDurationOrDefault(cfg.CheckInterval.Memory, 5*time.Second),
		OnWarning:          cfg.OnMemoryWarning,
		OnCritical:         cfg.OnMemoryCritical,
		OnDegradation:      cfg.OnMemoryDegradation,
	}
	rb.memoryGuard = NewMemoryGuard(memCfg)

	// Goroutine guard
	goroCfg := GoroutineGuardConfig{
		CheckInterval: parseDurationOrDefault(cfg.CheckInterval.Goroutine, 30*time.Second),
		MaxGrowthRate: cfg.GoroutineMaxGrowthRate,
		OnSpike:       cfg.OnGoroutineSpike,
	}
	rb.goroutineGuard = NewGoroutineGuard(goroCfg)

	// Queue guard
	queueCfg := QueueGuardConfig{
		CheckInterval: parseDurationOrDefault(cfg.CheckInterval.Queue, 5*time.Second),
		Thresholds:    cfg.QueueThresholds,
		OnWarning:     cfg.OnQueueWarning,
	}
	rb.queueGuard = NewQueueGuard(queueCfg)

	return rb
}

// RegisterQueues attaches queue providers to the queue guard.
func (rb *ResourceBudget) RegisterQueues(providers map[string]QueueDepthProvider) {
	for name, prov := range providers {
		rb.queueGuard.RegisterQueue(name, prov)
	}
}

// SetLoadSheddingCallback allows the system to shed non‑critical work when memory degradation occurs.
func (rb *ResourceBudget) SetLoadSheddingCallback(shed func(), restore func()) {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	rb.shedLoadFunc = shed
	rb.restoreFunc = restore
	rb.memoryGuard.SetLoadSheddingHooks(shed, restore)
}

// Health returns the current health status of all guards.
func (rb *ResourceBudget) Health() map[string]interface{} {
	rb.mu.RLock()
	defer rb.mu.RUnlock()

	status := map[string]interface{}{
		"started":        rb.started,
		"uptime_seconds": time.Since(rb.started).Seconds(),
	}

	if rb.memoryGuard != nil {
		status["memory"] = map[string]interface{}{
			"running":            true,
			"high_water_mark_mb": rb.memoryGuard.HighWaterMarkMB(),
		}
	}
	if rb.goroutineGuard != nil {
		status["goroutine"] = map[string]interface{}{
			"running":     true,
			"growth_rate": rb.goroutineGuard.GrowthRate(),
		}
	}
	if rb.queueGuard != nil {
		status["queue"] = map[string]interface{}{
			"running": true,
			"queues":  rb.queueGuard.RegisteredQueues(),
		}
	}

	return status
}

// Stop gracefully terminates all guards.
func (rb *ResourceBudget) Stop() {
	if rb.memoryGuard != nil {
		rb.memoryGuard.Stop()
	}
	if rb.goroutineGuard != nil {
		rb.goroutineGuard.Stop()
	}
	if rb.queueGuard != nil {
		rb.queueGuard.Stop()
	}
	zap.L().Info("resource budget stopped")
}

func parseDurationOrDefault(s string, def time.Duration) time.Duration {
	if s == "" {
		return def
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		return def
	}
	return d
}
