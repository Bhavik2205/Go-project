package wal

import (
	"context"
	"time"

	"go.uber.org/zap"
)

func StartPeriodicFlush(
	ctx context.Context,
	w Writer,
	interval time.Duration,
) {

	ticker := time.NewTicker(interval)

	go func() {
		defer ticker.Stop()

		for {
			select {

			case <-ticker.C:

				if err := w.Flush(); err != nil {
					zap.L().Error(
						"WAL flush failed",
						zap.Error(err),
					)
				}

			case <-ctx.Done():
				return
			}
		}
	}()
}
