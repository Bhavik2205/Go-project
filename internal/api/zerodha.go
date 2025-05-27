package api

import (
	kiteconnect "github.com/zerodha/gokiteconnect/v4"
)

type ZerodhaClient struct {
	Kite        *kiteconnect.Client
	AccessToken string
	APIKey      string
	APISecret   string
}

// ✅ Initialize the Zerodha client with tokens
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
