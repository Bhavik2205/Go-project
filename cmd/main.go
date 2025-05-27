package main

import (
	"context"
	"fmt"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/data"
)

func main() {
	// // Set environment variables (or make sure they're set externally)
	// os.Setenv("MARKETAUX_API_KEY", "your_marketaux_api_key")
	// os.Setenv("FINNHUB_API_KEY", "your_finnhub_api_key")
	// os.Setenv("EODHD_API_KEY", "your_eodhd_api_key")
	// os.Setenv("GOOGLE_CSE_API_KEY", "your_google_cse_api_key")
	// os.Setenv("GOOGLE_CSE_ID", "your_google_cse_id")

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	// Choose a test company symbol or name
	company := "AAPL" // Apple Inc.

	// Run the pipeline
	articles, err := data.RunNewsPipeline(ctx, company, nil)
	if err != nil {
		fmt.Printf("Error running news pipeline: %v\n", err)
	}

	// Print results
	for i, article := range articles {
		fmt.Printf("\n[%d] %s\n%s\n%s\nPublished at: %s\n",
			i+1, article.Title, article.Description, article.URL, article.PublishedAt.Format(time.RFC1123))
	}
}
