// package main

// import (
// 	"context"
// 	"fmt"
// 	"time"

// 	"github.com/Bhavik2205/ML-Bot/internal/data"
// )

// func main() {
// 	// // Set environment variables (or make sure they're set externally)
// 	// os.Setenv("MARKETAUX_API_KEY", "your_marketaux_api_key")
// 	// os.Setenv("FINNHUB_API_KEY", "your_finnhub_api_key")
// 	// os.Setenv("EODHD_API_KEY", "your_eodhd_api_key")
// 	// os.Setenv("GOOGLE_CSE_API_KEY", "your_google_cse_api_key")
// 	// os.Setenv("GOOGLE_CSE_ID", "your_google_cse_id")

// 	// Create context with timeout
// 	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
// 	defer cancel()

// 	// Choose a test company symbol or name
// 	company := "AAPL" // Apple Inc.

// 	// Run the pipeline
// 	articles, err := data.RunNewsPipeline(ctx, company, nil)
// 	if err != nil {
// 		fmt.Printf("Error running news pipeline: %v\n", err)
// 	}

// 	// Print results
// 	for i, article := range articles {
// 		fmt.Printf("\n[%d] %s\n%s\n%s\nPublished at: %s\n",
// 			i+1, article.Title, article.Description, article.URL, article.PublishedAt.Format(time.RFC1123))
// 	}
// }

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
	// Select exchange preference
	preferredExchanges := []string{"NSE"}
	// Lookup instrument token and exchange
	symbol := "TCS"
	info, err := client.FindInstrumentToken(symbol, preferredExchanges)
	if err != nil {
		log.Fatalf("❌ Failed to find instrument token: %v", err)
	}
	fmt.Printf("🔍 Subscribing to %s on %s (Token: %d)\n", info.Symbol, info.Exchange, info.Token)

	// Subscribe to ticks with symbol+exchange map
	// Subscribe to ticks
	err = client.SubscribeToTicks([]*api.InstrumentInfo{info})
	if err != nil {
		log.Fatalf("❌ WebSocket error: %v", err)
	}

	// block forever so we keep receiving ticks
	select {}
}
