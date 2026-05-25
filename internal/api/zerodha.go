package api

import (
	"errors"
	"fmt"
	"os"

	"github.com/Bhavik2205/ML-Bot/internal/cache"
	kiteconnect "github.com/zerodha/gokiteconnect/v4"
	kiteticker "github.com/zerodha/gokiteconnect/v4/ticker"
	"go.uber.org/zap"
)

type ZerodhaClient struct {
	Kite        *kiteconnect.Client
	AccessToken string
	APIKey      string
	// APISecret removed – sensitive, not needed after client creation
	Ticker *kiteticker.Ticker
}

type ZerodhaClientInterface interface {
	FindInstrumentToken(symbol string, exchanges []string) (*InstrumentInfo, error)
	GetUserProfile() (kiteconnect.UserProfile, error)      // Use kiteconnect.User instead of kitemodels.UserProfile
	GetQuote(symbols ...string) (kiteconnect.Quote, error) // Use kiteconnect.Quote
	SubscribeToTicks(instruments []*InstrumentInfo, redisClient *cache.RedisClient) error
}

// ✅ Exchange request_token for access_token (run once each day)
func GetAccessToken(apiKey, apiSecret, requestToken string) (string, error) {
	kc := kiteconnect.New(apiKey)
	session, err := kc.GenerateSession(requestToken, apiSecret)
	if err != nil {
		zap.L().Error("Error generating session", zap.Error(err))
		return "", fmt.Errorf("token exchange failed: %w", err)
	}
	return session.AccessToken, nil
}

// ✅ Initialize Zerodha client using access token
func NewZerodhaClient(apiKey, apiSecret, accessToken string) *ZerodhaClient {
	kc := kiteconnect.New(apiKey)
	kc.SetAccessToken(accessToken)

	return &ZerodhaClient{
		Kite:        kc,
		AccessToken: accessToken,
		APIKey:      apiKey,
		// APISecret intentionally omitted
	}
}

// ✅ Utility to load access token from file
func LoadAccessTokenFromFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		zap.L().Error("Failed to read access token file", zap.Error(err))
		return "", errors.New("access token file not found — run get_token.go first")
	}
	return string(data), nil
}

// Implement GetUserProfile
func (z *ZerodhaClient) GetUserProfile() (kiteconnect.UserProfile, error) {
	if z.Kite == nil {
		return kiteconnect.UserProfile{}, errors.New("kite client not initialized")
	}
	return z.Kite.GetUserProfile()
}

// Implement GetQuote
func (z *ZerodhaClient) GetQuote(symbols ...string) (kiteconnect.Quote, error) {
	if z.Kite == nil {
		return kiteconnect.Quote{}, errors.New("kite client not initialized")
	}
	return z.Kite.GetQuote(symbols...)
}
