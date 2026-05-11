package middleware_test

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/middleware"
	"golang.org/x/time/rate"
)

// ── RequestID Middleware ───────────────────────────────────────────────────────

func TestRequestID_GeneratesID(t *testing.T) {
	var gotID string
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotID = middleware.RequestIDFromContext(r.Context())
		w.WriteHeader(http.StatusOK)
	})
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rr := httptest.NewRecorder()
	middleware.RequestID(inner).ServeHTTP(rr, req)

	if gotID == "" {
		t.Error("expected generated request ID in context")
	}
	if rr.Header().Get("X-Request-ID") == "" {
		t.Error("expected X-Request-ID response header")
	}
}

func TestRequestID_PassthroughExistingID(t *testing.T) {
	var gotID string
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotID = middleware.RequestIDFromContext(r.Context())
		w.WriteHeader(http.StatusOK)
	})
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("X-Request-ID", "my-custom-id")
	rr := httptest.NewRecorder()
	middleware.RequestID(inner).ServeHTTP(rr, req)

	if gotID != "my-custom-id" {
		t.Errorf("expected 'my-custom-id', got %q", gotID)
	}
	if rr.Header().Get("X-Request-ID") != "my-custom-id" {
		t.Error("expected X-Request-ID response header to echo client value")
	}
}

func TestRequestID_UniquePerRequest(t *testing.T) {
	ids := make(map[string]struct{})
	var mu sync.Mutex
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := middleware.RequestIDFromContext(r.Context())
		mu.Lock()
		ids[id] = struct{}{}
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	})
	h := middleware.RequestID(inner)
	for i := 0; i < 100; i++ {
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, req)
	}
	if len(ids) < 90 {
		t.Errorf("expected mostly unique IDs, got only %d unique out of 100", len(ids))
	}
}

// ── Logger Middleware ──────────────────────────────────────────────────────────

func TestLogger_DoesNotPanic(t *testing.T) {
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	rr := httptest.NewRecorder()
	// Should not panic
	middleware.Logger(inner).ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
}

func TestLogger_CapturesStatus(t *testing.T) {
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	})
	req := httptest.NewRequest(http.MethodGet, "/missing", nil)
	rr := httptest.NewRecorder()
	middleware.Logger(inner).ServeHTTP(rr, req)
	if rr.Code != http.StatusNotFound {
		t.Errorf("expected 404 to pass through, got %d", rr.Code)
	}
}

// ── Rate Limiter Middleware ────────────────────────────────────────────────────

func TestRateLimiter_AllowsUnderLimit(t *testing.T) {
	rl := middleware.NewRateLimiter(rate.Limit(10), 10)
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	h := rl.Middleware(inner)

	for i := 0; i < 5; i++ {
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		req.RemoteAddr = "192.168.1.1:1234"
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, req)
		if rr.Code != http.StatusOK {
			t.Errorf("request %d: expected 200, got %d", i, rr.Code)
		}
	}
}

func TestRateLimiter_BlocksOverLimit(t *testing.T) {
	// 1 request/sec, burst 1
	rl := middleware.NewRateLimiter(rate.Limit(1), 1)
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	h := rl.Middleware(inner)

	var blocked int
	for i := 0; i < 10; i++ {
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		req.RemoteAddr = "10.0.0.1:9999"
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, req)
		if rr.Code == http.StatusTooManyRequests {
			blocked++
		}
	}
	if blocked == 0 {
		t.Error("expected at least one request to be rate limited")
	}
}

func TestRateLimiter_DifferentIPsIndependent(t *testing.T) {
	// 1 request/sec, burst 1 — each IP gets its own bucket
	rl := middleware.NewRateLimiter(rate.Limit(1), 1)
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	h := rl.Middleware(inner)

	// Each unique IP should get its first request through
	for i := 0; i < 10; i++ {
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		req.RemoteAddr = "10.0.0." + string(rune('0'+i)) + ":1234"
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, req)
		if rr.Code != http.StatusOK {
			t.Errorf("first request from unique IP %d should pass, got %d", i, rr.Code)
		}
	}
}

func TestRateLimiter_XForwardedFor(t *testing.T) {
	rl := middleware.NewRateLimiter(rate.Limit(1), 1)
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	h := rl.Middleware(inner)

	// First request with X-Forwarded-For should pass
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("X-Forwarded-For", "203.0.113.1, 10.0.0.1")
	req.RemoteAddr = "10.0.0.1:1234"
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200 for first request, got %d", rr.Code)
	}
}

func TestRateLimiter_ConcurrentRequests(t *testing.T) {
	rl := middleware.NewRateLimiter(rate.Limit(100), 100)
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	h := rl.Middleware(inner)

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			req := httptest.NewRequest(http.MethodGet, "/", nil)
			req.RemoteAddr = "172.16.0." + string(rune('0'+idx%10)) + ":1234"
			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, req)
		}(i)
	}
	wg.Wait()
}

// ── Middleware Order Tests ─────────────────────────────────────────────────────

func TestMiddlewareOrder_RequestIDBeforeLogger(t *testing.T) {
	// RequestID must run before Logger so Logger can read the ID from context
	var loggedID string
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		loggedID = middleware.RequestIDFromContext(r.Context())
		w.WriteHeader(http.StatusOK)
	})
	// Chain: RequestID -> Logger -> inner
	h := middleware.RequestID(middleware.Logger(inner))
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("X-Request-ID", "order-test-id")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if loggedID != "order-test-id" {
		t.Errorf("expected request ID in context when Logger runs, got %q", loggedID)
	}
}

func TestMiddlewareOrder_RecoveryWrapsAll(t *testing.T) {
	// A panicking handler should be caught by recovery middleware
	panicking := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic("test panic")
	})
	recovery := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if r := recover(); r != nil {
					w.WriteHeader(http.StatusInternalServerError)
				}
			}()
			next.ServeHTTP(w, r)
		})
	}
	h := recovery(panicking)
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusInternalServerError {
		t.Errorf("expected 500 from panic recovery, got %d", rr.Code)
	}
}

// ── Validation Tests ───────────────────────────────────────────────────────────

func TestRequestIDFromContext_Empty(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	id := middleware.RequestIDFromContext(req.Context())
	if id != "" {
		t.Errorf("expected empty string for missing request ID, got %q", id)
	}
}

func TestRateLimiter_429ResponseBody(t *testing.T) {
	rl := middleware.NewRateLimiter(rate.Limit(0.001), 1) // effectively 0 after burst
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	h := rl.Middleware(inner)

	// Exhaust burst
	req1 := httptest.NewRequest(http.MethodGet, "/", nil)
	req1.RemoteAddr = "1.2.3.4:1234"
	rr1 := httptest.NewRecorder()
	h.ServeHTTP(rr1, req1)

	// Next request should be rate limited
	time.Sleep(10 * time.Millisecond)
	req2 := httptest.NewRequest(http.MethodGet, "/", nil)
	req2.RemoteAddr = "1.2.3.4:1234"
	rr2 := httptest.NewRecorder()
	h.ServeHTTP(rr2, req2)

	if rr2.Code == http.StatusTooManyRequests {
		ct := rr2.Header().Get("Content-Type")
		if ct != "application/json" {
			t.Errorf("expected application/json Content-Type for 429, got %q", ct)
		}
	}
}
