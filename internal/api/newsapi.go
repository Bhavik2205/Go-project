package api

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"time"
)

type NewsArticle struct {
	Title       string    `json:"title"`
	Description string    `json:"description"`
	PublishedAt time.Time `json:"publishedAt"`
	URL         string    `json:"url"`
	Source      struct {
		Name string `json:"name"`
	} `json:"source"`
}

type NewsAPIResponse struct {
	Status       string        `json:"status"`
	TotalResults int           `json:"totalResults"`
	Articles     []NewsArticle `json:"articles"`
}

func FetchFinancialNews(apiKey string) ([]NewsArticle, error) {
	url := fmt.Sprintf("https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey=%s", apiKey)
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var newsResp NewsAPIResponse
	err = json.Unmarshal(body, &newsResp)
	if err != nil {
		return nil, err
	}

	return newsResp.Articles, nil
}
