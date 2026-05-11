package auth_test

import (
	"os"
	"testing"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/auth"
)

func TestGenerateAndParseAccessToken(t *testing.T) {
	os.Setenv("JWT_SECRET", "test-secret-32-bytes-long-enough!")
	token, err := auth.GenerateAccessToken(42)
	if err != nil {
		t.Fatalf("GenerateAccessToken: %v", err)
	}
	claims, err := auth.ParseToken(token, auth.TokenTypeAccess)
	if err != nil {
		t.Fatalf("ParseToken: %v", err)
	}
	if claims.UserID != 42 {
		t.Errorf("expected userID 42, got %d", claims.UserID)
	}
	if claims.TokenType != auth.TokenTypeAccess {
		t.Errorf("expected type %q, got %q", auth.TokenTypeAccess, claims.TokenType)
	}
}

func TestGenerateAndParseRefreshToken(t *testing.T) {
	os.Setenv("JWT_SECRET", "test-secret-32-bytes-long-enough!")
	token, err := auth.GenerateRefreshToken(99)
	if err != nil {
		t.Fatalf("GenerateRefreshToken: %v", err)
	}
	claims, err := auth.ParseToken(token, auth.TokenTypeRefresh)
	if err != nil {
		t.Fatalf("ParseToken: %v", err)
	}
	if claims.UserID != 99 {
		t.Errorf("expected userID 99, got %d", claims.UserID)
	}
}

func TestParseToken_WrongType(t *testing.T) {
	os.Setenv("JWT_SECRET", "test-secret-32-bytes-long-enough!")
	token, _ := auth.GenerateAccessToken(1)
	_, err := auth.ParseToken(token, auth.TokenTypeRefresh)
	if err != auth.ErrWrongType {
		t.Errorf("expected ErrWrongType, got %v", err)
	}
}

func TestParseToken_InvalidSignature(t *testing.T) {
	os.Setenv("JWT_SECRET", "test-secret-32-bytes-long-enough!")
	token, _ := auth.GenerateAccessToken(1)
	// Tamper with the token
	tampered := token[:len(token)-4] + "XXXX"
	_, err := auth.ParseToken(tampered, auth.TokenTypeAccess)
	if err == nil {
		t.Error("expected error for tampered token, got nil")
	}
}

func TestParseToken_Expired(t *testing.T) {
	// Use a very short TTL by temporarily overriding — we can't easily do this
	// without exporting signToken, so we verify the expiry is set correctly.
	os.Setenv("JWT_SECRET", "test-secret-32-bytes-long-enough!")
	token, _ := auth.GenerateAccessToken(1)
	claims, err := auth.ParseToken(token, auth.TokenTypeAccess)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Access token should expire in ~15 minutes
	ttl := time.Until(claims.ExpiresAt.Time)
	if ttl < 14*time.Minute || ttl > 16*time.Minute {
		t.Errorf("expected ~15m TTL, got %v", ttl)
	}
}

func TestParseToken_EmptySecret_UsesDefault(t *testing.T) {
	os.Unsetenv("JWT_SECRET")
	token, err := auth.GenerateAccessToken(7)
	if err != nil {
		t.Fatalf("GenerateAccessToken with default secret: %v", err)
	}
	claims, err := auth.ParseToken(token, auth.TokenTypeAccess)
	if err != nil {
		t.Fatalf("ParseToken with default secret: %v", err)
	}
	if claims.UserID != 7 {
		t.Errorf("expected userID 7, got %d", claims.UserID)
	}
}

func TestConcurrentTokenGeneration(t *testing.T) {
	os.Setenv("JWT_SECRET", "test-secret-32-bytes-long-enough!")
	const n = 100
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		go func(id uint) {
			tok, err := auth.GenerateAccessToken(id)
			if err != nil {
				errs <- err
				return
			}
			_, err = auth.ParseToken(tok, auth.TokenTypeAccess)
			errs <- err
		}(uint(i))
	}
	for i := 0; i < n; i++ {
		if err := <-errs; err != nil {
			t.Errorf("concurrent token error: %v", err)
		}
	}
}
