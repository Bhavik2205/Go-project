package monitor

import (
	"runtime"
	"time"

	"github.com/shirou/gopsutil/cpu"
	"github.com/shirou/gopsutil/mem"
	"go.uber.org/zap"
)

func StartSystemMonitor(interval time.Duration, alertFunc func(msg string)) {
	go func() {
		defer func() {
			if r := recover(); r != nil {
				zap.L().Error("Panic in system monitor goroutine", zap.Any("recover", r))
			}
		}()
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

			zap.L().Info("System stats",
				zap.Float64("cpu_percent", percent[0]),
				zap.Float64("mem_percent", vm.UsedPercent),
				zap.Int("goroutines", numGoroutine),
			)
			time.Sleep(interval)
		}
	}()
}
