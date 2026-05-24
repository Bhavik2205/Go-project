// internal/realtime/hub.go
package realtime

import (
	"net/http"
	"sync"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/cache"
	"github.com/Bhavik2205/ML-Bot/internal/data"
	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

// Hub manages all WebSocket connections.
type Hub struct {
	DataIngestor     *data.MarketDataIngestor
	CandleGenerator  *data.CandleGenerator
	IndicatorManager *data.IndicatorManager
	RedisClient      *cache.RedisClient
	heatmapClients   *sync.Map // new
}

// NewHub creates a new Hub.
func NewHub(
	dataIngestor *data.MarketDataIngestor,
	candleGenerator *data.CandleGenerator,
	indicatorManager *data.IndicatorManager,
	redisClient *cache.RedisClient,
) *Hub {
	return &Hub{
		DataIngestor:     dataIngestor,
		CandleGenerator:  candleGenerator,
		IndicatorManager: indicatorManager,
		RedisClient:      redisClient,
		heatmapClients:   &sync.Map{},
	}
}

// ServeTicks handles WebSocket connections for live ticks.
func (h *Hub) ServeTicks(w http.ResponseWriter, r *http.Request) {
	defer func() {
		if rec := recover(); rec != nil {
			zap.L().Error("Panic in WebSocket handler (ticks)", zap.Any("recover", rec))
		}
	}()

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		zap.L().Error("WebSocket upgrade error for ticks", zap.Error(err))
		return
	}
	defer h.DataIngestor.UnregisterWebSocketClient(conn)

	conn.SetReadLimit(wsMaxMsgSize)
	_ = conn.SetReadDeadline(time.Now().Add(wsPongWait))
	conn.SetPongHandler(func(string) error {
		_ = conn.SetReadDeadline(time.Now().Add(wsPongWait))
		return nil
	})

	h.DataIngestor.RegisterWebSocketClient(conn)

	for {
		if _, _, err := conn.ReadMessage(); err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				zap.L().Debug("WebSocket unexpected close (ticks)", zap.Error(err))
			}
			break
		}
	}
}

// ServeCandles handles WebSocket connections for candles.
func (h *Hub) ServeCandles(w http.ResponseWriter, r *http.Request) {
	defer func() {
		if rec := recover(); rec != nil {
			zap.L().Error("Panic in WebSocket handler (candles)", zap.Any("recover", rec))
		}
	}()

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		zap.L().Error("WebSocket upgrade error for candles", zap.Error(err))
		return
	}
	if h.CandleGenerator == nil {
		zap.L().Error("CandleGenerator is not initialized")
		conn.Close()
		return
	}
	defer h.CandleGenerator.UnregisterWebSocketClient(conn)

	conn.SetReadLimit(wsMaxMsgSize)
	_ = conn.SetReadDeadline(time.Now().Add(wsPongWait))
	conn.SetPongHandler(func(string) error {
		_ = conn.SetReadDeadline(time.Now().Add(wsPongWait))
		return nil
	})

	h.CandleGenerator.RegisterWebSocketClient(conn)
	zap.L().Info("New WebSocket client connected for candle data", zap.String("remote_addr", conn.RemoteAddr().String()))

	for {
		if _, _, err := conn.ReadMessage(); err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				zap.L().Debug("WebSocket unexpected close (candles)", zap.Error(err))
			}
			break
		}
	}
}

// ServeIndicators handles WebSocket connections for indicators.
func (h *Hub) ServeIndicators(w http.ResponseWriter, r *http.Request) {
	defer func() {
		if rec := recover(); rec != nil {
			zap.L().Error("Panic in WebSocket handler (indicators)", zap.Any("recover", rec))
		}
	}()

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		zap.L().Error("WebSocket upgrade error for indicators", zap.Error(err))
		return
	}
	if h.IndicatorManager == nil {
		zap.L().Error("IndicatorManager is not initialized")
		conn.Close()
		return
	}
	defer h.IndicatorManager.UnregisterWebSocketClient(conn)

	conn.SetReadLimit(wsMaxMsgSize)
	_ = conn.SetReadDeadline(time.Now().Add(wsPongWait))
	conn.SetPongHandler(func(string) error {
		_ = conn.SetReadDeadline(time.Now().Add(wsPongWait))
		return nil
	})

	h.IndicatorManager.RegisterWebSocketClient(conn)
	zap.L().Info("New WebSocket client connected for indicator data", zap.String("remote_addr", conn.RemoteAddr().String()))

	for {
		if _, _, err := conn.ReadMessage(); err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				zap.L().Debug("WebSocket unexpected close (indicators)", zap.Error(err))
			}
			break
		}
	}
}

func (h *Hub) ServeHeatmap(w http.ResponseWriter, r *http.Request) {
	defer func() {
		if rec := recover(); rec != nil {
			zap.L().Error("Panic in Heatmap WebSocket handler", zap.Any("recover", rec))
		}
	}()

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		zap.L().Error("WebSocket upgrade failed", zap.Error(err))
		return
	}
	// Register client
	h.heatmapClients.Store(conn, struct{}{})
	defer func() {
		h.heatmapClients.Delete(conn)
		conn.Close()
	}()

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
			snapshot := data.GetMarketHeatmap().Snapshot()
			_ = conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
			if err := conn.WriteJSON(snapshot); err != nil {
				zap.L().Debug("Heatmap WebSocket write error", zap.Error(err))
				return
			}
		}
	}
}

func (h *Hub) GetHeatmapClientCount() int {
	count := 0
	h.heatmapClients.Range(func(key, value interface{}) bool {
		count++
		return true
	})
	return count
}
