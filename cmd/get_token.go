package main

import (
	"fmt"
	"log"
	"os"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/joho/godotenv"
)

func main() {
	if err := godotenv.Load(); err != nil {
		fmt.Println("Warning: .env file not loaded, relying on system env variables")
	}

	apiKey := os.Getenv("ZERODHA_API_KEY")
	apiSecret := os.Getenv("ZERODHA_API_SECRET")
	// this expires every day, so you need to generate it each day
	requestToken := "Rjj005MV2AS1xZagRnPiaPv2B9V1uXh1" //generate this by loging in to https://kite.trade/connect/login?api_key=<your_api_key>

	fmt.Println("Api Key:", apiKey)
	fmt.Println("Api Secret:", apiSecret)
	accessToken, err := api.GetAccessToken(apiKey, apiSecret, requestToken)
	if err != nil {
		log.Fatalf("❌ Failed to get access token: %v", err)
	}

	fmt.Println("✅ Access token:", accessToken)

	// Save it to file
	err = os.WriteFile(".access_token", []byte(accessToken), 0644)
	if err != nil {
		log.Fatalf("❌ Failed to save token: %v", err)
	}
}
