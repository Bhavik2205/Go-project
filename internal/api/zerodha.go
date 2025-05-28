package api

import (
	"errors"
	"fmt"
	"os"

	kiteconnect "github.com/zerodha/gokiteconnect/v4"
	kiteticker "github.com/zerodha/gokiteconnect/v4/ticker"
)

type ZerodhaClient struct {
	Kite        *kiteconnect.Client
	AccessToken string
	APIKey      string
	APISecret   string
	Ticker      *kiteticker.Ticker
}

// ✅ Exchange request_token for access_token (run once each day)
func GetAccessToken(apiKey, apiSecret, requestToken string) (string, error) {
	kc := kiteconnect.New(apiKey)
	session, err := kc.GenerateSession(requestToken, apiSecret)
	if err != nil {
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
		APISecret:   apiSecret,
	}
}

// ✅ Utility to load access token from file
func LoadAccessTokenFromFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", errors.New("access token file not found — run get_token.go first")
	}
	return string(data), nil
}
