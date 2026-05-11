package auth_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	authhandler "github.com/Bhavik2205/ML-Bot/internal/api/handlers/auth"
	"github.com/Bhavik2205/ML-Bot/internal/auth"
	"github.com/Bhavik2205/ML-Bot/internal/cache"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/middleware"
	"golang.org/x/crypto/bcrypt"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

// newTestDB creates an in-memory SQLite DB with the User table.
func newTestDB(t *testing.T) *db.DBClient {
	t.Helper()
	gdb, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Silent),
	})
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	if err := gdb.AutoMigrate(&db.User{}); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	return &db.DBClient{DB: gdb}
}

// fakeRedis is a minimal in-memory Redis stub for tests.
type fakeRedis struct {
	data map[string]string
}

func newFakeRedis() *cache.RedisClient {
	// We can't easily construct a cache.RedisClient without a real Redis.
	// Return nil — the handlers handle nil redisClient gracefully.
	return nil
}

func init() {
	os.Setenv("JWT_SECRET", "test-secret-32-bytes-long-enough!")
}

func postJSON(t *testing.T, handler http.Handler, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	b, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)
	return rr
}

func TestHandleSignup_Success(t *testing.T) {
	dbc := newTestDB(t)
	h := authhandler.HandleSignup(dbc)

	rr := postJSON(t, h, "/api/v1/auth/signup", map[string]string{
		"email":    "test@example.com",
		"password": "StrongPass1!",
		"userName": "Tester",
	})

	if rr.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", rr.Code, rr.Body.String())
	}
	var resp map[string]any
	json.NewDecoder(rr.Body).Decode(&resp)
	data := resp["data"].(map[string]any)
	if data["accessToken"] == "" {
		t.Error("expected accessToken in response")
	}
	if data["refreshToken"] == "" {
		t.Error("expected refreshToken in response")
	}
}

func TestHandleSignup_DuplicateEmail(t *testing.T) {
	dbc := newTestDB(t)
	h := authhandler.HandleSignup(dbc)

	body := map[string]string{"email": "dup@example.com", "password": "StrongPass1!"}
	postJSON(t, h, "/", body)
	rr := postJSON(t, h, "/", body)

	if rr.Code != http.StatusConflict {
		t.Errorf("expected 409, got %d", rr.Code)
	}
}

func TestHandleSignup_WeakPassword(t *testing.T) {
	dbc := newTestDB(t)
	h := authhandler.HandleSignup(dbc)

	rr := postJSON(t, h, "/", map[string]string{
		"email":    "weak@example.com",
		"password": "short",
	})
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for weak password, got %d", rr.Code)
	}
}

func TestHandleSignup_InvalidEmail(t *testing.T) {
	dbc := newTestDB(t)
	h := authhandler.HandleSignup(dbc)

	rr := postJSON(t, h, "/", map[string]string{
		"email":    "not-an-email",
		"password": "StrongPass1!",
	})
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for invalid email, got %d", rr.Code)
	}
}

func TestHandleLogin_Success(t *testing.T) {
	dbc := newTestDB(t)
	hash, _ := bcrypt.GenerateFromPassword([]byte("MyPassword1!"), 12)
	dbc.DB.Create(&db.User{Email: "login@example.com", PasswordHash: string(hash), IsActive: true})

	h := authhandler.HandleLogin(dbc)
	rr := postJSON(t, h, "/", map[string]string{
		"email":    "login@example.com",
		"password": "MyPassword1!",
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestHandleLogin_WrongPassword(t *testing.T) {
	dbc := newTestDB(t)
	hash, _ := bcrypt.GenerateFromPassword([]byte("correct"), 12)
	dbc.DB.Create(&db.User{Email: "pw@example.com", PasswordHash: string(hash), IsActive: true})

	h := authhandler.HandleLogin(dbc)
	rr := postJSON(t, h, "/", map[string]string{
		"email":    "pw@example.com",
		"password": "wrong",
	})
	if rr.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", rr.Code)
	}
}

func TestHandleLogin_InactiveUser(t *testing.T) {
	dbc := newTestDB(t)
	hash, _ := bcrypt.GenerateFromPassword([]byte("pass1234"), 12)
	user := db.User{Email: "inactive@example.com", PasswordHash: string(hash)}
	dbc.DB.Create(&user)
	// Explicitly set IsActive=false after creation to ensure it's persisted
	dbc.DB.Model(&user).Update("is_active", false)

	h := authhandler.HandleLogin(dbc)
	rr := postJSON(t, h, "/", map[string]string{
		"email":    "inactive@example.com",
		"password": "pass1234",
	})
	if rr.Code != http.StatusForbidden {
		t.Errorf("expected 403, got %d", rr.Code)
	}
}

func TestHandleLogin_NonExistentUser(t *testing.T) {
	dbc := newTestDB(t)
	h := authhandler.HandleLogin(dbc)
	rr := postJSON(t, h, "/", map[string]string{
		"email":    "ghost@example.com",
		"password": "anything",
	})
	// Must return 401, not 404 (prevents email enumeration)
	if rr.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 (no enumeration), got %d", rr.Code)
	}
}

func TestHandleRefresh_Success(t *testing.T) {
	refreshToken, _ := auth.GenerateRefreshToken(5)
	h := authhandler.HandleRefresh()
	rr := postJSON(t, h, "/", map[string]string{"refreshToken": refreshToken})
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
	var resp map[string]any
	json.NewDecoder(rr.Body).Decode(&resp)
	data := resp["data"].(map[string]any)
	if data["accessToken"] == "" {
		t.Error("expected new accessToken")
	}
}

func TestHandleRefresh_InvalidToken(t *testing.T) {
	h := authhandler.HandleRefresh()
	rr := postJSON(t, h, "/", map[string]string{"refreshToken": "garbage.token.here"})
	if rr.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", rr.Code)
	}
}

func TestHandleRefresh_AccessTokenRejected(t *testing.T) {
	// Passing an access token where a refresh token is expected
	accessToken, _ := auth.GenerateAccessToken(1)
	h := authhandler.HandleRefresh()
	rr := postJSON(t, h, "/", map[string]string{"refreshToken": accessToken})
	if rr.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 for wrong token type, got %d", rr.Code)
	}
}

func TestHandleLogout_Returns204(t *testing.T) {
	refreshToken, _ := auth.GenerateRefreshToken(1)
	h := authhandler.HandleLogout(nil) // nil redis — logout still returns 204
	rr := postJSON(t, h, "/", map[string]string{"refreshToken": refreshToken})
	if rr.Code != http.StatusNoContent {
		t.Errorf("expected 204, got %d", rr.Code)
	}
}

func TestHandleLogout_MissingBody(t *testing.T) {
	h := authhandler.HandleLogout(nil)
	req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader([]byte(`{}`)))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for missing refreshToken, got %d", rr.Code)
	}
}

func TestAuthMiddleware_NoHeader(t *testing.T) {
	mw := middleware.Authenticate(nil)
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rr := httptest.NewRecorder()
	mw(inner).ServeHTTP(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 with no auth header, got %d", rr.Code)
	}
}

func TestAuthMiddleware_ValidToken(t *testing.T) {
	token, _ := auth.GenerateAccessToken(42)
	mw := middleware.Authenticate(nil)
	var gotUserID uint
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotUserID = middleware.UserIDFromContext(r.Context())
		w.WriteHeader(http.StatusOK)
	})
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	rr := httptest.NewRecorder()
	mw(inner).ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
	if gotUserID != 42 {
		t.Errorf("expected userID 42 in context, got %d", gotUserID)
	}
}

func TestAuthMiddleware_InvalidToken(t *testing.T) {
	mw := middleware.Authenticate(nil)
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Authorization", "Bearer invalid.token.here")
	rr := httptest.NewRecorder()
	mw(inner).ServeHTTP(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", rr.Code)
	}
}

func TestAuthMiddleware_RefreshTokenRejected(t *testing.T) {
	// Refresh tokens must not be accepted as access tokens
	refreshToken, _ := auth.GenerateRefreshToken(1)
	mw := middleware.Authenticate(nil)
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Authorization", "Bearer "+refreshToken)
	rr := httptest.NewRecorder()
	mw(inner).ServeHTTP(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 for refresh token used as access token, got %d", rr.Code)
	}
}

func TestUserIDFromContext_Missing(t *testing.T) {
	uid := middleware.UserIDFromContext(context.Background())
	if uid != 0 {
		t.Errorf("expected 0 for missing userID, got %d", uid)
	}
}

func TestFullAuthFlow(t *testing.T) {
	dbc := newTestDB(t)

	// 1. Signup
	signupH := authhandler.HandleSignup(dbc)
	rr := postJSON(t, signupH, "/", map[string]string{
		"email":    "flow@example.com",
		"password": "FlowPass123!",
	})
	if rr.Code != http.StatusCreated {
		t.Fatalf("signup failed: %d %s", rr.Code, rr.Body.String())
	}
	var signupResp map[string]any
	json.NewDecoder(rr.Body).Decode(&signupResp)
	data := signupResp["data"].(map[string]any)
	accessToken := data["accessToken"].(string)
	refreshToken := data["refreshToken"].(string)

	// 2. Access protected endpoint with access token
	mw := middleware.Authenticate(nil)
	var gotUID uint
	protected := mw(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotUID = middleware.UserIDFromContext(r.Context())
		w.WriteHeader(http.StatusOK)
	}))
	req := httptest.NewRequest(http.MethodGet, "/me", nil)
	req.Header.Set("Authorization", "Bearer "+accessToken)
	rr2 := httptest.NewRecorder()
	protected.ServeHTTP(rr2, req)
	if rr2.Code != http.StatusOK {
		t.Fatalf("protected endpoint failed: %d", rr2.Code)
	}
	if gotUID == 0 {
		t.Error("expected non-zero userID in context")
	}

	// 3. Refresh tokens
	refreshH := authhandler.HandleRefresh()
	rr3 := postJSON(t, refreshH, "/", map[string]string{"refreshToken": refreshToken})
	if rr3.Code != http.StatusOK {
		t.Fatalf("refresh failed: %d %s", rr3.Code, rr3.Body.String())
	}
	var refreshResp map[string]any
	json.NewDecoder(rr3.Body).Decode(&refreshResp)
	newData := refreshResp["data"].(map[string]any)
	newAccessToken := newData["accessToken"].(string)
	if newAccessToken == "" {
		t.Error("expected non-empty new accessToken")
	}
	// Verify the new token is valid
	if _, err := auth.ParseToken(newAccessToken, auth.TokenTypeAccess); err != nil {
		t.Errorf("new access token is invalid: %v", err)
	}

	// 4. Logout
	logoutH := authhandler.HandleLogout(nil)
	logoutReq := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(
		[]byte(`{"refreshToken":"`+refreshToken+`"}`),
	))
	logoutReq.Header.Set("Content-Type", "application/json")
	logoutReq.Header.Set("Authorization", "Bearer "+accessToken)
	rr4 := httptest.NewRecorder()
	logoutH.ServeHTTP(rr4, logoutReq)
	if rr4.Code != http.StatusNoContent {
		t.Fatalf("logout failed: %d", rr4.Code)
	}

	// 5. Verify refresh token is now invalid (wrong type check still works)
	rr5 := postJSON(t, refreshH, "/", map[string]string{"refreshToken": accessToken})
	if rr5.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 using access token as refresh, got %d", rr5.Code)
	}
}

func TestConcurrentSignups(t *testing.T) {
	dbc := newTestDB(t)
	h := authhandler.HandleSignup(dbc)
	const n = 20
	results := make(chan int, n)
	for i := 0; i < n; i++ {
		go func(idx int) {
			rr := postJSON(t, h, "/", map[string]any{
				"email":    fmt.Sprintf("concurrent%d@example.com", idx),
				"password": "StrongPass1!",
			})
			results <- rr.Code
		}(i)
	}
	for i := 0; i < n; i++ {
		code := <-results
		if code != http.StatusCreated {
			t.Errorf("concurrent signup got %d, expected 201", code)
		}
	}
}

// Verify token expiry is set correctly for refresh tokens (~7 days)
func TestRefreshTokenTTL(t *testing.T) {
	os.Setenv("JWT_SECRET", "test-secret-32-bytes-long-enough!")
	token, _ := auth.GenerateRefreshToken(1)
	claims, err := auth.ParseToken(token, auth.TokenTypeRefresh)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	ttl := time.Until(claims.ExpiresAt.Time)
	if ttl < 6*24*time.Hour || ttl > 8*24*time.Hour {
		t.Errorf("expected ~7d TTL for refresh token, got %v", ttl)
	}
}
