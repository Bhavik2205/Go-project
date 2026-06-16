package reliability

import (
	"runtime"
	"sync"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/observability"
	"go.uber.org/zap"
)

type GoroutineGuard struct {
	lastCount  uint64
	lastCheck  time.Time
	mu         sync.RWMutex
	growthRate float64 // goroutines per minute — protected by mu
	onSpike    func(rate float64)
	stopCh     chan struct{}
	once       sync.Once
}

type GoroutineGuardConfig struct {
	CheckInterval time.Duration
	MaxGrowthRate float64 // max allowed percent growth per check interval before spike fires
	OnSpike       func(pctGrowth float64)
}

func NewGoroutineGuard(cfg GoroutineGuardConfig) *GoroutineGuard {
	if cfg.CheckInterval == 0 {
		cfg.CheckInterval = 30 * time.Second
	}
	g := &GoroutineGuard{
		lastCount: uint64(runtime.NumGoroutine()),
		lastCheck: time.Now(),
		stopCh:    make(chan struct{}),
		onSpike:   cfg.OnSpike,
	}

	// Set guard status to running
	observability.GoroutineGuardStatus.Set(1)

	go g.run(cfg)
	return g
}

func (g *GoroutineGuard) run(cfg GoroutineGuardConfig) {
	ticker := time.NewTicker(cfg.CheckInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			now := time.Now()
			current := uint64(runtime.NumGoroutine())
			elapsed := now.Sub(g.lastCheck).Minutes()
			if elapsed > 0 && g.lastCount > 0 {
				delta := int64(current) - int64(g.lastCount)
				pctGrowth := (float64(current)/float64(g.lastCount) - 1) * 100
				ratePerMin := float64(delta) / elapsed

				g.mu.Lock()
				g.growthRate = ratePerMin
				g.mu.Unlock()

				observability.GoroutineCount.Set(float64(current))
				observability.GoroutineGrowthRate.Set(ratePerMin)
				observability.ReliabilityLastCheck.Set(float64(now.Unix()))

				if pctGrowth > cfg.MaxGrowthRate {
					if g.onSpike != nil {
						g.onSpike(pctGrowth)
					}
					zap.L().Warn("goroutine spike detected",
						zap.Uint64("current", current),
						zap.Uint64("previous", g.lastCount),
						zap.Float64("rate_per_min", ratePerMin),
						zap.Float64("pct_growth", pctGrowth))
				}
			}
			g.lastCount = current
			g.lastCheck = now
		case <-g.stopCh:
			observability.GoroutineGuardStatus.Set(0)
			return
		}
	}
}

func (g *GoroutineGuard) Stop() {
	g.once.Do(func() { close(g.stopCh) })
}

func (g *GoroutineGuard) GrowthRate() float64 {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.growthRate
}
