package auth

import (
	"errors"
	"os"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

const (
	accessTokenTTL  = 15 * time.Minute
	refreshTokenTTL = 7 * 24 * time.Hour
	TokenTypeAccess  = "access"
	TokenTypeRefresh = "refresh"
)

var (
	ErrInvalidToken = errors.New("invalid or expired token")
	ErrWrongType    = errors.New("wrong token type")
)

type Claims struct {
	UserID    uint   `json:"uid"`
	TokenType string `json:"typ"`
	jwt.RegisteredClaims
}

func jwtSecret() []byte {
	s := os.Getenv("JWT_SECRET")
	if s == "" {
		s = "change-me-in-production"
	}
	return []byte(s)
}

func GenerateAccessToken(userID uint) (string, error) {
	return signToken(userID, TokenTypeAccess, accessTokenTTL)
}

func GenerateRefreshToken(userID uint) (string, error) {
	return signToken(userID, TokenTypeRefresh, refreshTokenTTL)
}

func signToken(userID uint, tokenType string, ttl time.Duration) (string, error) {
	claims := Claims{
		UserID:    userID,
		TokenType: tokenType,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(ttl)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
		},
	}
	return jwt.NewWithClaims(jwt.SigningMethodHS256, claims).SignedString(jwtSecret())
}

// ParseToken validates the token and returns its claims.
// expectedType should be TokenTypeAccess or TokenTypeRefresh.
func ParseToken(tokenStr, expectedType string) (*Claims, error) {
	token, err := jwt.ParseWithClaims(tokenStr, &Claims{}, func(t *jwt.Token) (any, error) {
		if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, ErrInvalidToken
		}
		return jwtSecret(), nil
	})
	if err != nil || !token.Valid {
		return nil, ErrInvalidToken
	}
	claims, ok := token.Claims.(*Claims)
	if !ok {
		return nil, ErrInvalidToken
	}
	if claims.TokenType != expectedType {
		return nil, ErrWrongType
	}
	return claims, nil
}
