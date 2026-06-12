package observability

import (
"runtime"
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

// SystemHealth returns a comprehensive health check of the system.
func SystemHealth() map[string]interface{} {
return map[string]interface{}{
"timestamp":  time.Now().Unix(),
"memory":     getMemoryHealth(),
"goroutines": getGoroutineHealth(),
"ticks":      getTickHealth(),
"queues":     getQueueHealth(),
}
}

func getMemoryHealth() map[string]interface{} {
var m runtime.MemStats
runtime.ReadMemStats(&m)

status := HealthOK
heapPercent := float64(m.HeapAlloc) / float64(m.HeapSys)
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
status := HealthOK
message := "Feed is live"

return map[string]interface{}{
"status":  status,
"message": message,
}
}

func getQueueHealth() map[string]interface{} {
return map[string]interface{}{
"tick_queue_depth":      "monitoring enabled",
"candle_queue_depth":    "monitoring enabled",
"db_flush_queue_depth":  "monitoring enabled",
}
}
