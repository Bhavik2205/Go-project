package main

import (
	"fmt"
	"os"

	"github.com/joho/godotenv"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/data"
	"github.com/Bhavik2205/ML-Bot/internal/model"
)

func main() {
	err := godotenv.Load()
	if err != nil {
		fmt.Println("❌ Error loading .env file")
		return
	}

	apiKey := os.Getenv("NEWS_API_KEY")
	if apiKey == "" {
		fmt.Println("Please set the NEWS_API_KEY environment variable.")
		return
	}

	newsArticles, err := api.FetchFinancialNews(apiKey)
	if err != nil {
		fmt.Println("Error fetching news:", err)
		return
	}

	fmt.Printf("Fetched %d news articles.\n", len(newsArticles))

	for i, article := range newsArticles {
		// Combine title and description
		text := article.Title + " " + article.Description
		cleanText := data.CleanText(text)
		fmt.Print("Clean: ", cleanText)
		sentiment, confidence, err := model.AnalyzeSentiment(cleanText)
		if err != nil {
			fmt.Printf("Error analyzing article %d: %v\n", i+1, err)
			continue
		}

		fmt.Printf("\nArticle #%d:\n", i+1)
		fmt.Printf("Title: %s\n", article.Title)
		fmt.Printf("Source: %s | Published: %s\n", article.Source.Name, article.PublishedAt.Format("2006-01-02"))
		fmt.Printf("Sentiment: %s (%.2f confidence)\n", sentiment, confidence)
	}
}
