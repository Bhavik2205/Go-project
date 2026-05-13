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
		defer func() {
			if rec := recover(); rec != nil {
				zap.L().Error("Panic in Heatmap WebSocket handler", zap.Any("recover", rec))
			}
		}()
		conn, err := heatmapUpgrader.Upgrade(w, r, nil)
		if err != nil {
			zap.L().Error("WebSocket upgrade failed", zap.Error(err))
			return
		}
		defer conn.Close()

		conn.SetReadLimit(wsMaxMsgSize)
		_ = conn.SetReadDeadline(time.Now().Add(wsPongWait))
		conn.SetPongHandler(func(string) error {
			_ = conn.SetReadDeadline(time.Now().Add(wsPongWait))
			return nil
		})

		// Drain incoming frames in background so pong handler fires.
		ctx := r.Context()
		go func() {
			for {
				if _, _, err := conn.ReadMessage(); err != nil {
					return
				}
			}
		}()

		broadcastTicker := time.NewTicker(200 * time.Millisecond)
		pingTicker := time.NewTicker(wsPingPeriod)
		defer broadcastTicker.Stop()
		defer pingTicker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-pingTicker.C:
				_ = conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
				if err := conn.WriteMessage(websocket.PingMessage, nil); err != nil {
					return
				}
			case <-broadcastTicker.C:
				snapshot := marketHeatmap.Snapshot()
				_ = conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
				if err := conn.WriteJSON(snapshot); err != nil {
					zap.L().Debug("Heatmap WebSocket write error", zap.Error(err))
					return
				}
			}
		}
	}
}
