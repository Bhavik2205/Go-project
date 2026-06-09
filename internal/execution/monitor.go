package monitor

import (
	"runtime"
	"time"

	"github.com/shirou/gopsutil/cpu"
	"github.com/shirou/gopsutil/mem"
	"go.uber.org/zap"
)

func StartSystemMonitor(interval time.Duration, alertFunc func(msg string), statsProvider interface{ DroppedTicks() uint64 }) {
	go func() {
		defer func() {
			if r := recover(); r != nil {
				zap.L().Error("Panic in system monitor goroutine", zap.Any("recover", r))
			}
		}()

		var lastDropped uint64
		var lastTime time.Time

		for {
			percent, err := cpu.Percent(0, false)
			if err != nil || len(percent) == 0 {
				zap.L().Error("Failed to get CPU percent", zap.Error(err))
				time.Sleep(interval)
				continue
			}
			vm, err := mem.VirtualMemory()
			if err != nil {
				zap.L().Error("Failed to get memory stats", zap.Error(err))
				time.Sleep(interval)
				continue
			}
			numGoroutine := runtime.NumGoroutine()

			if percent[0] > 90 {
				alertFunc("High CPU usage detected")
			}
			if vm.UsedPercent > 90 {
				alertFunc("High memory usage detected")
			}
			if numGoroutine > 10000 {
				alertFunc("Too many goroutines")
			}

			// Tick bus drops
			var dropped uint64
			var dropRate float64
			if statsProvider != nil {
				dropped = statsProvider.DroppedTicks()
				if !lastTime.IsZero() {
					elapsed := time.Since(lastTime).Seconds()
					if elapsed > 0 {
						dropRate = float64(dropped-lastDropped) / elapsed
					}
				}
				if dropRate > 1000 {
					alertFunc("High tick drop rate detected")
				}
				lastDropped = dropped
				lastTime = time.Now()
			}

			fields := []zap.Field{
				zap.Float64("cpu_percent", percent[0]),
				zap.Float64("mem_percent", vm.UsedPercent),
				zap.Int("goroutines", numGoroutine),
			}
			if statsProvider != nil {
				fields = append(fields,
					zap.Uint64("dropped_ticks_total", dropped),
					zap.Float64("dropped_ticks_per_sec", dropRate),
				)
			}
			zap.L().Info("System stats", fields...)
			time.Sleep(interval)
		}
	}()
}
