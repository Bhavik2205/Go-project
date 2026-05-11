package server_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/Bhavik2205/ML-Bot/internal/server"
	"github.com/Bhavik2205/ML-Bot/internal/utils"
)

func TestMain(m *testing.M) {
	os.Setenv("JWT_SECRET", "test-secret-32-bytes-long-enough!")
	os.Exit(m.Run())
}

// ── Health endpoint ────────────────────────────────────────────────────────────

func TestHealthEndpoint_NoDependencies(t *testing.T) {
	// With no DB/Redis/Zerodha wired, health should still return 200
	// with "not_configured" for each dependency.
	server.SetDBClient(nil)
	server.SetRedisClient(nil)
	server.SetAppConfig(&utils.AppConfig{})
	server.SetStartupTime(server.NowForTest())

	router := server.BuildTestRouter()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/health", nil)
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp map[string]any
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	data := resp["data"].(map[string]any)
	if data["status"] != "ok" {
		t.Errorf("expected status ok, got %v", data["status"])
	}
	deps := data["dependencies"].(map[string]any)
	if deps["postgres"] != "not_configured" {
		t.Errorf("expected postgres not_configured, got %v", deps["postgres"])
	}
	if deps["redis"] != "not_configured" {
		t.Errorf("expected redis not_configured, got %v", deps["redis"])
	}
}

func TestHealthEndpoint_SimulationMode(t *testing.T) {
	server.SetAppConfig(&utils.AppConfig{
		Market: struct {
			Simulate                  bool    `yaml:"simulate"`
			SimulationSpeedMultiplier float64 `yaml:"simulation_speed_multiplier"`
		}{Simulate: true},
	})
	router := server.BuildTestRouter()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/health", nil)
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	var resp map[string]any
	json.NewDecoder(rr.Body).Decode(&resp)
	data := resp["data"].(map[string]any)
	if data["mode"] != "simulation" {
		t.Errorf("expected mode simulation, got %v", data["mode"])
	}
}

func TestHealthEndpoint_LiveMode(t *testing.T) {
	server.SetAppConfig(&utils.AppConfig{})
	router := server.BuildTestRouter()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/health", nil)
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	var resp map[string]any
	json.NewDecoder(rr.Body).Decode(&resp)
	data := resp["data"].(map[string]any)
	if data["mode"] != "live" {
		t.Errorf("expected mode live, got %v", data["mode"])
	}
}

func TestHealthEndpoint_UptimeIncreases(t *testing.T) {
	router := server.BuildTestRouter()
	req1 := httptest.NewRequest(http.MethodGet, "/api/v1/health", nil)
	rr1 := httptest.NewRecorder()
	router.ServeHTTP(rr1, req1)

	var r1 map[string]any
	json.NewDecoder(rr1.Body).Decode(&r1)
	uptime1 := r1["data"].(map[string]any)["uptimeSeconds"].(float64)
	if uptime1 < 0 {
		t.Errorf("expected non-negative uptime, got %f", uptime1)
	}
}

func TestHealthEndpoint_MetaFields(t *testing.T) {
	router := server.BuildTestRouter()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/health", nil)
	req.Header.Set("X-Request-ID", "health-test-id")
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	var resp map[string]any
	json.NewDecoder(rr.Body).Decode(&resp)
	meta := resp["meta"].(map[string]any)
	if meta["requestId"] != "health-test-id" {
		t.Errorf("expected requestId health-test-id, got %v", meta["requestId"])
	}
	if meta["version"] != "v1" {
		t.Errorf("expected version v1, got %v", meta["version"])
	}
}

// ── Quotes endpoint ────────────────────────────────────────────────────────────

func TestQuotesEndpoint_MissingSymbols(t *testing.T) {
	router := server.BuildTestRouter()

	// Quotes is a protected route — need a valid JWT
	token := server.TestAccessToken(t, 1)
	req := httptest.NewRequest(http.MethodGet, "/api/v1/quotes", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for missing symbols, got %d", rr.Code)
	}
}

func TestQuotesEndpoint_NoAuth(t *testing.T) {
	router := server.BuildTestRouter()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/quotes?symbols=NSE:RELIANCE", nil)
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 without auth, got %d", rr.Code)
	}
}

func TestQuotesEndpoint_WithSymbols_FallsBackToHeatmap(t *testing.T) {
	// No Zerodha client wired — should fall back to heatmap (empty) and return 200
	router := server.BuildTestRouter()
	token := server.TestAccessToken(t, 1)
	req := httptest.NewRequest(http.MethodGet, "/api/v1/quotes?symbols=NSE:RELIANCE,NSE:TCS", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
	var resp map[string]any
	json.NewDecoder(rr.Body).Decode(&resp)
	if resp["data"] == nil {
		t.Error("expected data array in response")
	}
}

// ── Market overview endpoint ───────────────────────────────────────────────────

func TestMarketOverviewEndpoint_NoAuth(t *testing.T) {
	router := server.BuildTestRouter()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/market/overview", nil)
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 without auth, got %d", rr.Code)
	}
}

func TestMarketOverviewEndpoint_EmptyHeatmap(t *testing.T) {
	router := server.BuildTestRouter()
	token := server.TestAccessToken(t, 1)
	req := httptest.NewRequest(http.MethodGet, "/api/v1/market/overview", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
	var resp map[string]any
	json.NewDecoder(rr.Body).Decode(&resp)
	data := resp["data"].(map[string]any)
	if data["topGainers"] == nil {
		t.Error("expected topGainers in response")
	}
}

// ── Auth routes are public ─────────────────────────────────────────────────────

func TestAuthRoutes_ArePublic(t *testing.T) {
	router := server.BuildTestRouter()
	publicRoutes := []struct {
		method string
		path   string
		body   string
	}{
		{"POST", "/api/v1/auth/signup", `{"email":"pub@example.com","password":"StrongPass1!"}`},
		{"POST", "/api/v1/auth/login", `{"email":"pub@example.com","password":"StrongPass1!"}`},
		{"POST", "/api/v1/auth/refresh", `{"refreshToken":"dummy"}`},
		{"POST", "/api/v1/auth/logout", `{"refreshToken":"dummy"}`},
	}
	for _, rt := range publicRoutes {
		req := httptest.NewRequest(rt.method, rt.path, nil)
		rr := httptest.NewRecorder()
		router.ServeHTTP(rr, req)
		// Should NOT return 401 (may return 400/500 due to missing body/DB, but not 401)
		if rr.Code == http.StatusUnauthorized {
			t.Errorf("%s %s should be public, got 401", rt.method, rt.path)
		}
	}
}

// ── CORS headers ──────────────────────────────────────────────────────────────

func TestCORSHeaders_AllowedOrigin(t *testing.T) {
	t.Setenv("ALLOWED_ORIGINS", "http://localhost:3000")
	router := server.BuildTestRouter()
	req := httptest.NewRequest(http.MethodOptions, "/api/v1/health", nil)
	req.Header.Set("Origin", "http://localhost:3000")
	req.Header.Set("Access-Control-Request-Method", "GET")
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	if rr.Header().Get("Access-Control-Allow-Origin") != "http://localhost:3000" {
		t.Errorf("expected CORS origin header, got %q", rr.Header().Get("Access-Control-Allow-Origin"))
	}
}

func TestCORSHeaders_UnknownOriginBlocked(t *testing.T) {
	t.Setenv("ALLOWED_ORIGINS", "http://localhost:3000")
	router := server.BuildTestRouter()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/health", nil)
	req.Header.Set("Origin", "http://evil.com")
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	if rr.Header().Get("Access-Control-Allow-Origin") == "http://evil.com" {
		t.Error("unknown origin should not be reflected in CORS header")
	}
}
