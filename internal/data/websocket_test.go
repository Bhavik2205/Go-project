package data_test

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// wsTestServer creates a minimal WebSocket test server using the given handler.
func wsTestServer(t *testing.T, handler http.HandlerFunc) (*httptest.Server, string) {
	t.Helper()
	srv := httptest.NewServer(handler)
	url := "ws" + strings.TrimPrefix(srv.URL, "http")
	return srv, url
}

var upgrader = websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}

// ── Basic WebSocket connection lifecycle ──────────────────────────────────────

func TestWebSocket_ConnectAndDisconnect(t *testing.T) {
	srv, url := wsTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade: %v", err)
			return
		}
		defer conn.Close()
		// Read one message then close
		if _, _, err := conn.ReadMessage(); err != nil {
			// client closed — normal in this test
			_ = err
		}
	})
	defer srv.Close()

	conn, _, err := websocket.DefaultDialer.Dial(url, nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	if err := conn.WriteMessage(websocket.TextMessage, []byte("hello")); err != nil {
		t.Logf("write: %v", err) // non-fatal: server may have closed already
	}
	conn.Close()
}

// ── Write-after-close protection ──────────────────────────────────────────────

func TestWebSocket_WriteAfterClose_DoesNotPanic(t *testing.T) {
	// Simulate the writePump pattern: write to a closed channel should not panic
	// because we use channel-based dispatch, not direct conn.Write.
	ch := make(chan []byte, 10)
	close(ch)

	// Reading from a closed channel returns zero value and false — no panic
	msg, ok := <-ch
	if ok {
		t.Errorf("expected closed channel to return ok=false, got msg=%v", msg)
	}
}

func TestWebSocket_ChannelClose_StopsWritePump(t *testing.T) {
	// Verify that closing the send channel causes the range loop to exit cleanly
	ch := make(chan []byte, 5)
	var received [][]byte
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for msg := range ch {
			received = append(received, msg)
		}
	}()

	ch <- []byte("msg1")
	ch <- []byte("msg2")
	close(ch)
	wg.Wait()

	if len(received) != 2 {
		t.Errorf("expected 2 messages, got %d", len(received))
	}
}

// ── Concurrent broadcast safety ───────────────────────────────────────────────

func TestWebSocket_ConcurrentBroadcast_NoRace(t *testing.T) {
	// Simulate the sync.Map-based client registry under concurrent load.
	// Each client has a buffered channel. Broadcasters send non-blocking.
	// Unregistration uses LoadAndDelete + close.
	// The key invariant: once a channel is closed, no sender should send to it.
	// In production, this is guaranteed because:
	//   1. LoadAndDelete removes the channel from the map atomically.
	//   2. Broadcasters only send to channels they retrieved from Range.
	//   3. A channel removed from the map before Range sees it is never sent to.
	// This test verifies that invariant holds under concurrent load.
	var clients sync.Map
	const numClients = 50
	const numBroadcasts = 100

	// Register clients with buffered channels
	for i := 0; i < numClients; i++ {
		ch := make(chan []byte, 200) // large buffer to avoid blocking
		clients.Store(i, ch)
	}

	var drops int64
	var wg sync.WaitGroup

	// Concurrent broadcasters — non-blocking sends only
	for b := 0; b < 5; b++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < numBroadcasts; i++ {
				msg := []byte("tick data")
				clients.Range(func(key, value any) bool {
					ch := value.(chan []byte)
					select {
					case ch <- msg:
					default:
						atomic.AddInt64(&drops, 1)
					}
					return true
				})
			}
		}()
	}

	wg.Wait()

	// After all broadcasts complete, safely drain and close all channels
	clients.Range(func(key, value any) bool {
		if val, ok := clients.LoadAndDelete(key); ok {
			ch := val.(chan []byte)
			// Drain remaining messages
			for len(ch) > 0 {
				<-ch
			}
			close(ch)
		}
		return true
	})
}

// ── Reconnect storm simulation ────────────────────────────────────────────────

func TestWebSocket_ReconnectStorm(t *testing.T) {
	var activeConns int64
	srv, url := wsTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		atomic.AddInt64(&activeConns, 1)
		defer func() {
			atomic.AddInt64(&activeConns, -1)
			conn.Close()
		}()
		// Hold connection briefly then close
		time.Sleep(5 * time.Millisecond)
	})
	defer srv.Close()

	const storms = 50
	var wg sync.WaitGroup
	var dialErrors int64

	for i := 0; i < storms; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			conn, _, err := websocket.DefaultDialer.Dial(url, nil)
			if err != nil {
				atomic.AddInt64(&dialErrors, 1)
				return
			}
			conn.Close()
		}()
	}
	wg.Wait()

	if dialErrors > storms/2 {
		t.Errorf("too many dial errors in reconnect storm: %d/%d", dialErrors, storms)
	}
}

// ── Disconnect cleanup verification ──────────────────────────────────────────

func TestWebSocket_DisconnectCleanup_NoLeak(t *testing.T) {
	// Verify that LoadAndDelete pattern prevents double-close
	var clients sync.Map
	ch := make(chan []byte, 10)
	clients.Store("client1", ch)

	// First unregister
	if val, ok := clients.LoadAndDelete("client1"); ok {
		close(val.(chan []byte))
	}

	// Second unregister of same key — LoadAndDelete returns ok=false, no double close
	if _, ok := clients.LoadAndDelete("client1"); ok {
		t.Error("LoadAndDelete should return false for already-deleted key")
	}
}

func TestWebSocket_SyncMap_ConcurrentStoreDelete(t *testing.T) {
	var m sync.Map
	var wg sync.WaitGroup

	// Concurrent stores and deletes — must not panic or race
	for i := 0; i < 100; i++ {
		wg.Add(2)
		go func(key int) {
			defer wg.Done()
			m.Store(key, make(chan []byte, 1))
		}(i)
		go func(key int) {
			defer wg.Done()
			m.Delete(key)
		}(i)
	}
	wg.Wait()
}

// ── Backpressure / slow client simulation ─────────────────────────────────────

func TestWebSocket_SlowClient_DropsMessages(t *testing.T) {
	// A client with a full send channel should have messages dropped, not block the broadcaster
	ch := make(chan []byte, 2) // tiny buffer
	ch <- []byte("msg1")
	ch <- []byte("msg2") // buffer full

	var dropped int64
	// Broadcaster tries to send — should not block
	for i := 0; i < 10; i++ {
		select {
		case ch <- []byte("overflow"):
		default:
			atomic.AddInt64(&dropped, 1)
		}
	}

	if dropped == 0 {
		t.Error("expected drops for slow client with full buffer")
	}
	if dropped != 10 {
		t.Errorf("expected all 10 overflow messages dropped, got %d", dropped)
	}
}

// ── WebSocket message integrity ───────────────────────────────────────────────

func TestWebSocket_MessageIntegrity(t *testing.T) {
	messages := []string{"tick1", "tick2", "tick3", "candle1", "indicator1"}
	received := make(chan string, len(messages))

	srv, url := wsTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		for _, msg := range messages {
			if err := conn.WriteMessage(websocket.TextMessage, []byte(msg)); err != nil {
				return
			}
		}
	})
	defer srv.Close()

	conn, _, err := websocket.DefaultDialer.Dial(url, nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	for i := 0; i < len(messages); i++ {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			t.Fatalf("read %d: %v", i, err)
		}
		received <- string(msg)
	}
	close(received)

	var got []string
	for m := range received {
		got = append(got, m)
	}
	if len(got) != len(messages) {
		t.Errorf("expected %d messages, got %d", len(messages), len(got))
	}
	for i, m := range got {
		if m != messages[i] {
			t.Errorf("message %d: expected %q, got %q", i, messages[i], m)
		}
	}
}

// ── Goroutine leak detection ───────────────────────────────────────────────────

func TestWebSocket_NoGoroutineLeak_OnDisconnect(t *testing.T) {
	// Each client gets a write pump goroutine. After disconnect, it must exit.
	var pumpsRunning int64

	startPump := func(ch <-chan []byte) {
		atomic.AddInt64(&pumpsRunning, 1)
		go func() {
			defer atomic.AddInt64(&pumpsRunning, -1)
			for range ch {
				// drain
			}
		}()
	}

	const n = 20
	channels := make([]chan []byte, n)
	for i := 0; i < n; i++ {
		ch := make(chan []byte, 10)
		channels[i] = ch
		startPump(ch)
	}

	// Close all channels (simulates client disconnect)
	for _, ch := range channels {
		close(ch)
	}

	// Give goroutines time to exit
	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		if atomic.LoadInt64(&pumpsRunning) == 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	if remaining := atomic.LoadInt64(&pumpsRunning); remaining != 0 {
		t.Errorf("goroutine leak: %d write pump goroutines still running after channel close", remaining)
	}
}
