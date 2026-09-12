package realtime

import (
	"encoding/json"
	"net/http"
	"sync"
	"time"

	monitor "github.com/Bhavik2205/ML-Bot/internal/execution"
	"github.com/Bhavik2205/ML-Bot/internal/strategy"
	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

// TradingSource supplies the state pushed over /ws/trading. Every field may be
// nil, because the strategy engine is opt-in.
type TradingSource struct {
	Engine   *strategy.Engine
	Paper    *monitor.PaperBroker
	Guard    *monitor.Guard
	Executor *monitor.OrderExecutor
	Mode     string
}

// tradingSnapshot is one push. It carries everything the trading panels need,
// so a client needs a single socket rather than one per panel.
type tradingSnapshot struct {
	Type      string    `json:"type"`
	Timestamp time.Time `json:"timestamp"`
	Mode      string    `json:"mode"`
	Enabled   bool      `json:"enabled"`

	Engine   *strategy.EngineStats  `json:"engine,omitempty"`
	Paper    *monitor.PaperStats    `json:"paper,omitempty"`
	Guard    *monitor.GuardStats    `json:"guard,omitempty"`
	Executor *monitor.ExecutorStats `json:"executor,omitempty"`
}

// tradingPushInterval is how often a snapshot goes out. Trading state changes
// on candle finalisation, not per tick, so a faster cadence would send
// identical payloads.
const tradingPushInterval = time.Second

// tradingHub fans one snapshot out to all connected clients.
type tradingHub struct {
	src     TradingSource
	clients sync.Map // *websocket.Conn -> chan []byte

	once sync.Once
}

// SetTradingSource attaches the trading components. Call during startup,
// before the HTTP server accepts connections.
func (h *Hub) SetTradingSource(src TradingSource) {
	h.trading.src = src
}

// StartTradingBroadcast begins the periodic push loop. Safe to call more than
// once; only the first call starts the goroutine.
func (h *Hub) StartTradingBroadcast(stop <-chan struct{}) {
	h.trading.once.Do(func() {
		go h.trading.run(stop)
	})
}

func (t *tradingHub) run(stop <-chan struct{}) {
	ticker := time.NewTicker(tradingPushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			t.broadcast()
		case <-stop:
			zap.L().Info("Trading broadcast stopped")
			return
		}
	}
}

// snapshot builds the current state.
func (t *tradingHub) snapshot() tradingSnapshot {
	s := tradingSnapshot{
		Type:      "trading",
		Timestamp: time.Now(),
		Mode:      t.src.Mode,
		Enabled:   t.src.Engine != nil,
	}
	if t.src.Engine != nil {
		e := t.src.Engine.Stats()
		s.Engine = &e
	}
	if t.src.Paper != nil {
		p := t.src.Paper.Stats()
		s.Paper = &p
	}
	if t.src.Guard != nil {
		g := t.src.Guard.Stats()
		s.Guard = &g
	}
	if t.src.Executor != nil {
		x := t.src.Executor.Stats()
		s.Executor = &x
	}
	return s
}

// broadcast marshals once and fans the bytes out. Marshalling per client would
// repeat the same work for every connection.
func (t *tradingHub) broadcast() {
	var n int
	t.clients.Range(func(_, _ any) bool { n++; return false })
	if n == 0 {
		return // nobody listening; skip the snapshot entirely
	}

	payload, err := json.Marshal(t.snapshot())
	if err != nil {
		zap.L().Error("could not encode trading snapshot", zap.Error(err))
		return
	}

	t.clients.Range(func(k, v any) bool {
		ch, ok := v.(chan []byte)
		if !ok {
			return true
		}
		select {
		case ch <- payload:
		default:
			// A client that cannot keep up gets this frame dropped rather
			// than stalling the broadcast for everyone else.
		}
		return true
	})
}

// ServeTrading streams trading state to one client.
func (h *Hub) ServeTrading(w http.ResponseWriter, r *http.Request) {
	defer func() {
		if rec := recover(); rec != nil {
			zap.L().Error("Panic in WebSocket handler (trading)", zap.Any("recover", rec))
		}
	}()

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		zap.L().Error("WebSocket upgrade error for trading", zap.Error(err))
		return
	}

	send := make(chan []byte, 8)
	h.trading.clients.Store(conn, send)

	defer func() {
		h.trading.clients.Delete(conn)
		close(send)
		_ = conn.Close()
	}()

	conn.SetReadLimit(wsMaxMsgSize)
	_ = conn.SetReadDeadline(time.Now().Add(wsPongWait))
	conn.SetPongHandler(func(string) error {
		_ = conn.SetReadDeadline(time.Now().Add(wsPongWait))
		return nil
	})

	// Send one snapshot immediately so a new client renders without waiting
	// for the next tick.
	if first, err := json.Marshal(h.trading.snapshot()); err == nil {
		_ = conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
		_ = conn.WriteMessage(websocket.TextMessage, first)
	}

	go h.tradingWritePump(conn, send)

	// Block on reads so the deferred cleanup runs when the client disconnects.
	for {
		if _, _, err := conn.ReadMessage(); err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				zap.L().Debug("WebSocket unexpected close (trading)", zap.Error(err))
			}
			return
		}
	}
}

// tradingWritePump writes queued frames and keeps the connection alive.
func (h *Hub) tradingWritePump(conn *websocket.Conn, send <-chan []byte) {
	ping := time.NewTicker(wsPingPeriod)
	defer ping.Stop()

	for {
		select {
		case msg, ok := <-send:
			if !ok {
				_ = conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
				_ = conn.WriteMessage(websocket.CloseMessage, nil)
				return
			}
			_ = conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
			if err := conn.WriteMessage(websocket.TextMessage, msg); err != nil {
				return
			}
		case <-ping.C:
			_ = conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
			if err := conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// TradingClientCount reports connected trading-stream clients.
func (h *Hub) TradingClientCount() int {
	n := 0
	h.trading.clients.Range(func(_, _ any) bool { n++; return true })
	return n
}
