package monitor

import (
	"log"
	"runtime"
	"time"

	"github.com/shirou/gopsutil/cpu"
	"github.com/shirou/gopsutil/mem"
)

func StartSystemMonitor(interval time.Duration, alertFunc func(msg string)) {
	go func() {
		for {
			// CPU
			percent, _ := cpu.Percent(0, false)
			// Memory
			vm, _ := mem.VirtualMemory()
			// Goroutines
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

			log.Printf("CPU: %.2f%%, Mem: %.2f%%, Goroutines: %d", percent[0], vm.UsedPercent, numGoroutine)
			time.Sleep(interval)
		}
	}()
}
