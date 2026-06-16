package observability

import (
	"context"
	"runtime"
	"sync"
	"time"
)

// HealthStatus represents the health of a system component.
type HealthStatus string

const (
	HealthOK        HealthStatus = "ok"
	HealthDegraded  HealthStatus = "degraded"
	HealthUnhealthy HealthStatus = "unhealthy"
)

// HealthCheck represents the result of a health check.
type HealthCheck struct {
	Status  HealthStatus           `json:"status"`
	Message string                 `json:"message"`
	Details map[string]interface{} `json:"details,omitempty"`
}

// DependencyPinger allows health checks to ping external dependencies.
type DependencyPinger interface {
	PingDB(ctx context.Context) error
	PingRedis(ctx context.Context) error
}

// QueueDepthProvider returns current depths and capacities of internal queues.
type QueueDepthProvider interface {
	TickQueueLen() int
	TickQueueCap() int
	DBFlushQueueLen() int
	DBFlushQueueCap() int
	CandleQueueLen() int
	CandleQueueCap() int
	IndicatorQueueLen() int
	IndicatorQueueCap() int
}

var (
	globalPinger        DependencyPinger
	globalQueueProvider QueueDepthProvider

	// depCache caches the last dependency health result for up to depCacheTTL
	// to avoid pinging DB/Redis on every health scrape.
	depCache    sync.Map // key: string -> depCacheEntry
	depCacheTTL = 10 * time.Second
)

type depCacheEntry struct {
	result map[string]interface{}
	at     time.Time
}

// RegisterDependencyPinger sets the pinger used by health checks.
func RegisterDependencyPinger(p DependencyPinger) { globalPinger = p }

// RegisterQueueDepthProvider sets the queue depth provider used by health checks.
func RegisterQueueDepthProvider(q QueueDepthProvider) { globalQueueProvider = q }

// SystemHealth returns a comprehensive health check of the system.
func SystemHealth() map[string]interface{} {
	return map[string]interface{}{
		"timestamp":  time.Now().Unix(),
		"memory":     getMemoryHealth(),
		"goroutines": getGoroutineHealth(),
		"ticks":      getTickHealth(),
		"queues":     getQueueHealth(),
		"database":   getDependencyHealth("database"),
		"redis":      getDependencyHealth("redis"),
	}
}

func getMemoryHealth() map[string]interface{} {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	var heapPercent float64
	if m.HeapSys > 0 {
		heapPercent = float64(m.HeapAlloc) / float64(m.HeapSys)
	}
	status := HealthOK
	if heapPercent > 0.9 {
		status = HealthUnhealthy
	} else if heapPercent > 0.75 {
		status = HealthDegraded
	}

	return map[string]interface{}{
		"status":        status,
		"heap_alloc_mb": float64(m.HeapAlloc) / 1024 / 1024,
		"heap_total_mb": float64(m.HeapSys) / 1024 / 1024,
		"sys_mb":        float64(m.Sys) / 1024 / 1024,
		"gc_count":      m.NumGC,
	}
}

func getGoroutineHealth() map[string]interface{} {
	count := runtime.NumGoroutine()
	status := HealthOK
	if count > 10000 {
		status = HealthUnhealthy
	} else if count > 5000 {
		status = HealthDegraded
	}
	return map[string]interface{}{
		"status": status,
		"count":  count,
	}
}

func getTickHealth() map[string]interface{} {
	nanoVal := lastTickTimeNano.Load()
	if nanoVal == 0 {
		return map[string]interface{}{
			"status":  HealthUnhealthy,
			"message": "No ticks received yet",
		}
	}
	lastTick := time.Unix(0, nanoVal)
	since := time.Since(lastTick)
	status := HealthOK
	message := "Feed is live"
	// Thresholds align with TickFeedDead metric (set=1 at >5s).
	// >5s = degraded (stale), >30s = unhealthy (dead).
	if since > 30*time.Second {
		status = HealthUnhealthy
		message = "Feed dead: no ticks for >30s"
	} else if since > 5*time.Second {
		status = HealthDegraded
		message = "Feed stale: no ticks for >5s"
	}
	return map[string]interface{}{
		"status":          status,
		"message":         message,
		"last_tick_ago_ms": since.Milliseconds(),
	}
}

func getQueueHealth() map[string]interface{} {
	if globalQueueProvider == nil {
		return map[string]interface{}{"status": HealthDegraded, "message": "queue provider not registered"}
	}
	q := globalQueueProvider
	return map[string]interface{}{
		"tick":      queueStat(q.TickQueueLen(), q.TickQueueCap()),
		"db_flush":  queueStat(q.DBFlushQueueLen(), q.DBFlushQueueCap()),
		"candle":    queueStat(q.CandleQueueLen(), q.CandleQueueCap()),
		"indicator": queueStat(q.IndicatorQueueLen(), q.IndicatorQueueCap()),
	}
}

func queueStat(length, capacity int) map[string]interface{} {
	var usagePct float64
	if capacity > 0 {
		usagePct = float64(length) / float64(capacity) * 100
	}
	status := HealthOK
	if usagePct > 90 {
		status = HealthUnhealthy
	} else if usagePct > 70 {
		status = HealthDegraded
	}
	return map[string]interface{}{
		"status":    status,
		"length":    length,
		"capacity":  capacity,
		"usage_pct": usagePct,
	}
}

func getDependencyHealth(dep string) map[string]interface{} {
	if globalPinger == nil {
		return map[string]interface{}{"status": HealthDegraded, "message": "pinger not registered"}
	}

	if v, ok := depCache.Load(dep); ok {
		e := v.(depCacheEntry)
		if time.Since(e.at) < depCacheTTL {
			return e.result
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	var err error
	switch dep {
	case "database":
		err = globalPinger.PingDB(ctx)
	case "redis":
		err = globalPinger.PingRedis(ctx)
	}

	var result map[string]interface{}
	if err != nil {
		result = map[string]interface{}{"status": HealthUnhealthy, "message": err.Error()}
	} else {
		result = map[string]interface{}{"status": HealthOK}
	}
	depCache.Store(dep, depCacheEntry{result: result, at: time.Now()})
	return result
}
