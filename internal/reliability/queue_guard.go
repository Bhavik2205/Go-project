package reliability

import (
	"sync"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/observability"
	"go.uber.org/zap"
)

// QueueDepthProvider returns the current depth and capacity of a named queue.
type QueueDepthProvider interface {
	Len() int
	Cap() int
}

type QueueGuard struct {
	queues      map[string]QueueDepthProvider
	mu          sync.RWMutex
	thresholds  map[string]float64 // name -> capacity fraction (0.8 = 80%)
	aboveThresh map[string]bool    // tracks per-queue breach state
	onWarning   func(name string, usage float64)
	stopCh      chan struct{}
	once        sync.Once
}

type QueueGuardConfig struct {
	CheckInterval time.Duration
	// thresholds: e.g., "db_flush": 0.8, "broadcast": 0.9
	Thresholds map[string]float64
	OnWarning  func(name string, usage float64)
}

func NewQueueGuard(cfg QueueGuardConfig) *QueueGuard {
	if cfg.CheckInterval == 0 {
		cfg.CheckInterval = 5 * time.Second
	}
	g := &QueueGuard{
		queues:      make(map[string]QueueDepthProvider),
		thresholds:  cfg.Thresholds,
		aboveThresh: make(map[string]bool),
		onWarning:   cfg.OnWarning,
		stopCh:      make(chan struct{}),
	}

	// Set guard status to running
	observability.QueueGuardStatus.Set(1)

	go g.run(cfg.CheckInterval)
	return g
}

// RegisterQueue adds a named queue to be monitored.
func (g *QueueGuard) RegisterQueue(name string, provider QueueDepthProvider) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.queues[name] = provider
}

// RegisteredQueues returns the names of all registered queues (for health reporting).
func (g *QueueGuard) RegisteredQueues() []string {
	g.mu.RLock()
	defer g.mu.RUnlock()
	names := make([]string, 0, len(g.queues))
	for name := range g.queues {
		names = append(names, name)
	}
	return names
}

func (g *QueueGuard) run(interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			// Snapshot providers under lock, then release before polling depths
			// so Len()/Cap() calls (which may touch live channels) don't hold the mutex.
			g.mu.Lock()
			type snapshot struct {
				name string
				prov QueueDepthProvider
			}
			snaps := make([]snapshot, 0, len(g.queues))
			for name, prov := range g.queues {
				snaps = append(snaps, snapshot{name, prov})
			}
			g.mu.Unlock()

			for _, s := range snaps {
				length := s.prov.Len()
				capacity := s.prov.Cap()
				if capacity == 0 {
					continue
				}
				usage := float64(length) / float64(capacity)

				switch s.name {
				case "tick_broadcast":
					observability.TickQueueDepth.Set(float64(length))
				case "db_flush":
					observability.DBFlushQueueDepth.Set(float64(length))
				case "candle_flush":
					observability.CandleQueueDepth.Set(float64(length))
				case "indicator_output":
					observability.IndicatorQueueDepth.Set(float64(length))
				}

				observability.ReliabilityLastCheck.Set(float64(time.Now().Unix()))

				g.mu.Lock()
				if thr, ok := g.thresholds[s.name]; ok {
					wasAbove := g.aboveThresh[s.name]
					nowAbove := usage >= thr
					if nowAbove && !wasAbove {
						if g.onWarning != nil {
							g.onWarning(s.name, usage)
						}
						zap.L().Warn("queue usage above threshold",
							zap.String("queue", s.name),
							zap.Float64("usage_pct", usage*100),
							zap.Int("length", length),
							zap.Int("capacity", capacity))
					}
					g.aboveThresh[s.name] = nowAbove
				}
				g.mu.Unlock()
			}
		case <-g.stopCh:
			observability.QueueGuardStatus.Set(0)
			return
		}
	}
}

func (g *QueueGuard) Stop() {
	g.once.Do(func() { close(g.stopCh) })
}
