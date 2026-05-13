package auth

import (
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
)

const (
	accessTokenTTL   = 15 * time.Minute
	refreshTokenTTL  = 7 * 24 * time.Hour
	TokenTypeAccess  = "access"
	TokenTypeRefresh = "refresh"

	// jwtIssuer identifies your application in every token.
	// Change this to match your actual app/domain name.
	jwtIssuer = "go-trading-bot"
)

var (
	ErrInvalidToken = errors.New("invalid or expired token")
	ErrWrongType    = errors.New("wrong token type")
)

// Claims holds the data embedded inside every JWT token.
// UserID identifies which user the token belongs to.
// TokenType tells us whether this is an access or refresh token.
// JTI (JWT ID) is a unique ID per token — needed for refresh token revocation.
// jwt.RegisteredClaims holds standard fields like expiry, issuer, subject.
type Claims struct {
	UserID    uint   `json:"uid"`
	TokenType string `json:"typ"`
	JTI       string `json:"jti"` // unique token ID — do not remove, used by AUD-002
	jwt.RegisteredClaims
}

// MustLoadJWTSecret reads the JWT signing secret from the environment.
//
// IMPORTANT: this function panics (crashes the server) if JWT_SECRET is not set.
// That is intentional. A server running without a real secret is a security hole.
// Set JWT_SECRET to a random string of at least 32 characters in your .env file.
//
// Generate one with: openssl rand -hex 32
func MustLoadJWTSecret() []byte {
	s := os.Getenv("JWT_SECRET")
	if s == "" {
		// panic here is correct — we want the server to refuse to start,
		// not silently run with a known fallback secret.
		panic("JWT_SECRET environment variable is not set. " +
			"Generate one with: openssl rand -hex 32")
	}
	if len(s) < 32 {
		panic(fmt.Sprintf(
			"JWT_SECRET is too short (%d chars). Use at least 32 characters. "+
				"Generate one with: openssl rand -hex 32", len(s)))
	}
	return []byte(s)
}

// jwtSecret is the internal helper used by sign/parse functions.
// It calls MustLoadJWTSecret every time so a missing env var is always caught.
func jwtSecret() []byte {
	return MustLoadJWTSecret()
}

// GenerateAccessToken creates a short-lived token (15 min) for API requests.
// Call this after a successful login and send it to the frontend.
func GenerateAccessToken(userID uint) (string, error) {
	return signToken(userID, TokenTypeAccess, accessTokenTTL)
}

// GenerateRefreshToken creates a long-lived token (7 days) for getting new access tokens.
// Store this securely — in an httpOnly cookie, never in localStorage.
func GenerateRefreshToken(userID uint) (string, error) {
	return signToken(userID, TokenTypeRefresh, refreshTokenTTL)
}

func signToken(userID uint, tokenType string, ttl time.Duration) (string, error) {
	// Generate a unique ID for this specific token.
	// This is what lets us revoke a specific refresh token later (AUD-002).
	jti, err := uuid.NewRandom()
	if err != nil {
		return "", fmt.Errorf("failed to generate token ID: %w", err)
	}

	now := time.Now()
	claims := Claims{
		UserID:    userID,
		TokenType: tokenType,
		JTI:       jti.String(),
		RegisteredClaims: jwt.RegisteredClaims{
			// Subject: who this token is about (the user's ID as a string)
			Subject: fmt.Sprintf("%d", userID),
			// Issuer: which application issued this token
			Issuer: jwtIssuer,
			// ExpiresAt: when this token stops being valid
			ExpiresAt: jwt.NewNumericDate(now.Add(ttl)),
			// IssuedAt: when this token was created
			IssuedAt: jwt.NewNumericDate(now),
			// NotBefore: token is not valid before this time (same as issued)
			NotBefore: jwt.NewNumericDate(now),
			// ID: standard location for the unique token ID
			ID: jti.String(),
		},
	}
	return jwt.NewWithClaims(jwt.SigningMethodHS256, claims).SignedString(jwtSecret())
}

// ParseToken validates a token string and returns its claims.
//
// expectedType must be TokenTypeAccess or TokenTypeRefresh.
// Never use an access token where a refresh token is expected or vice versa —
// this check prevents token type confusion attacks.
func ParseToken(tokenStr, expectedType string) (*Claims, error) {
	token, err := jwt.ParseWithClaims(tokenStr, &Claims{}, func(t *jwt.Token) (any, error) {
		// Reject tokens signed with any algorithm other than HMAC.
		// Without this check, an attacker could send a token signed with "none"
		// (no signature) and it would pass validation.
		if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, ErrInvalidToken
		}
		return jwtSecret(), nil
	}, jwt.WithIssuer(jwtIssuer)) //also validate that the issuer matches

	if err != nil || !token.Valid {
		return nil, ErrInvalidToken
	}

	claims, ok := token.Claims.(*Claims)
	if !ok {
		return nil, ErrInvalidToken
	}

	// Reject tokens of the wrong type.
	// Example: a refresh token must never be accepted where an access token is expected.
	if claims.TokenType != expectedType {
		return nil, ErrWrongType
	}
	return claims, nil
}
