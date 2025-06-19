package server

import (
	"net/http"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/data"
	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

var heatmapUpgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

func HeatmapWebSocketHandler(marketHeatmap *data.MarketHeatmap) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		conn, err := heatmapUpgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()

		ticker := time.NewTicker(200 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				snapshot := marketHeatmap.Snapshot()
				zap.L().Info("Sending heatmap snapshot", zap.Int("count", len(snapshot)))
				if err := conn.WriteJSON(snapshot); err != nil {
					zap.L().Error("HeatMap WebSocket write error", zap.Error(err))
					return // Client disconnected
				}
			}
		}
	}
}
