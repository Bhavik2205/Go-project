package main

import (
	"fmt"
	"log"
	"os"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/server"
	"github.com/joho/godotenv"
)

func main() {
	if err := godotenv.Load(); err != nil {
		fmt.Println("⚠️  .env file not found, using system env vars")
	}

	apiKey := os.Getenv("ZERODHA_API_KEY")
	apiSecret := os.Getenv("ZERODHA_API_SECRET")

	accessToken, err := api.LoadAccessTokenFromFile(".access_token")
	if err != nil {
		log.Fatal(err)
	}

	client := api.NewZerodhaClient(apiKey, apiSecret, accessToken)

	user, err := client.Kite.GetUserProfile()
	if err != nil {
		log.Fatalf("❌ Invalid session or token expired: %v", err)
	}
	fmt.Printf("✅ Logged in as: %s (%s)\n", user.UserName, user.UserID)

	// Start WebSocket server for frontend clients
	go server.StartWebSocketServer()

	// Multiple symbols
	symbols := []string{"NIFTY 50", "NIFTY BANK", "RELIANCE", "TCS"}
	preferredExchanges := []string{"NSE"}

	var infos []*api.InstrumentInfo
	for _, symbol := range symbols {
		info, err := client.FindInstrumentToken(symbol, preferredExchanges)
		if err != nil {
			log.Printf("❌ Failed to find token for %s: %v", symbol, err)
			continue
		}
		fmt.Printf("🔍 Subscribing to %s on %s (Token: %d)\n", info.Symbol, info.Exchange, info.Token)
		infos = append(infos, info)
	}

	if len(infos) == 0 {
		log.Fatal("❌ No valid instruments to subscribe to.")
	}

	err = client.SubscribeToTicks(infos)
	if err != nil {
		log.Fatalf("❌ WebSocket error: %v", err)
	}

	select {} // block forever
}
