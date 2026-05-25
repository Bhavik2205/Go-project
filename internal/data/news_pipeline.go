package data

// import (
// 	"context"
// 	"crypto/sha1"
// 	"encoding/json"
// 	"errors"
// 	"fmt"
// 	"io"
// 	"net/http"
// 	"os"
// 	"strconv"
// 	"sync"
// 	"time"

// 	"go.uber.org/zap"

// 	"github.com/joho/godotenv"
// 	"github.com/prometheus/client_golang/prometheus"
// )

// // Prometheus metrics
// var (
// 	newsFetchCount = prometheus.NewCounterVec(
// 		prometheus.CounterOpts{
// 			Name: "news_fetch_requests_total",
// 			Help: "Total number of news fetch requests per source",
// 		},
// 		[]string{"source"},
// 	)
// 	newsFetchErrors = prometheus.NewCounterVec(
// 		prometheus.CounterOpts{
// 			Name: "news_fetch_errors_total",
// 			Help: "Total number of errors while fetching news per source",
// 		},
// 		[]string{"source"},
// 	)
// 	newsFetchDuration = prometheus.NewHistogramVec(
// 		prometheus.HistogramOpts{
// 			Name:    "news_fetch_duration_seconds",
// 			Help:    "Duration of news fetch HTTP requests per source",
// 			Buckets: prometheus.ExponentialBuckets(0.1, 2, 8), // 0.1s to ~12.8s
// 		},
// 		[]string{"source"},
// 	)
// )

// func mustGetEnvAsInt(key string, fallback int) int {
// 	val := os.Getenv(key)
// 	if val == "" {
// 		return fallback
// 	}
// 	if v, err := strconv.Atoi(val); err == nil {
// 		return v
// 	}
// 	logger.Warnw("Invalid int for env var", "key", key, "value", val)
// 	return fallback
// }

// func getBoolEnv(key string, defaultVal bool) bool {
// 	valStr := os.Getenv(key)
// 	if valStr == "" {
// 		return defaultVal
// 	}
// 	val, err := strconv.ParseBool(valStr)
// 	if err != nil {
// 		logger.Warnw("Unable to parse env var as bool, using default", "key", key, "value", valStr, "default", defaultVal)
// 		return defaultVal
// 	}
// 	return val
// }

// func init() {
// 	// Load .env file
// 	if err := godotenv.Load(); err != nil {
// 		zap.L().Warn("Failed to load .env file, relying on system env variables")
// 	}

// 	for _, k := range []string{
// 		"USE_MARKETAUX", "MARKETAUX_LIMIT",
// 		"USE_FINNHUB", "FINNHUB_LIMIT",
// 		"USE_EODHD", "EODHD_LIMIT",
// 		"USE_GOOGLE_CSE", "GOOGLE_CSE_LIMIT",
// 	} {
// 		zap.L().Debug("Env var", zap.String("key", k), zap.String("value", os.Getenv(k)))
// 	}

// 	// Register Prometheus metrics
// 	prometheus.MustRegister(newsFetchCount, newsFetchErrors, newsFetchDuration)

// 	// Initialize zap logger (production config)
// 	l, err := zap.NewProduction()
// 	if err != nil {
// 		panic(fmt.Sprintf("failed to initialize logger: %v", err))
// 	}
// 	logger = l.Sugar()

// 	// Initialize HTTP client with timeout
// 	client = &http.Client{
// 		Timeout: 10 * time.Second,
// 	}

// 	// Validate API keys early
// 	validateAPIKeys()

// 	config = &NewsPipelineConfig{
// 		FreeMode:       true,
// 		UseMarketaux:   getBoolEnv("USE_MARKETAUX", false),
// 		MarketauxLimit: mustGetEnvAsInt("MARKETAUX_LIMIT", 100),

// 		UseFinnhub:   getBoolEnv("USE_FINNHUB", false),
// 		FinnhubLimit: mustGetEnvAsInt("FINNHUB_LIMIT", 100),

// 		UseEODHD:   getBoolEnv("USE_EODHD", false),
// 		EODHDLimit: mustGetEnvAsInt("EODHD_LIMIT", 20),

// 		UseGoogleCSE:   getBoolEnv("USE_GOOGLE_CSE", false),
// 		GoogleCSELimit: mustGetEnvAsInt("GOOGLE_CSE_LIMIT", 50),
// 	}

// }

// func validateAPIKeys() {
// 	requiredKeys := []string{
// 		"MARKETAUX_API_KEY",
// 		"FINNHUB_API_KEY",
// 		"EODHD_API_KEY",
// 		"GOOGLE_CSE_API_KEY",
// 		"GOOGLE_CSE_CX_ID",
// 	}

// 	for _, k := range requiredKeys {
// 		if os.Getenv(k) == "" {
// 			logger.Warnw("API key environment variable is not set", "key", k)
// 		}
// 	}
// }

// // NewsArticle represents a normalized structure for news from any source
// type NewsArticle struct {
// 	Source      string    `json:"source"`
// 	Title       string    `json:"title"`
// 	Description string    `json:"description"`
// 	URL         string    `json:"url"`
// 	PublishedAt time.Time `json:"published_at"`
// }

// // NewsPipelineConfig defines dynamic config options for each news source
// type NewsPipelineConfig struct {
// 	FreeMode       bool
// 	UseMarketaux   bool
// 	MarketauxLimit int
// 	UseFinnhub     bool
// 	FinnhubLimit   int
// 	UseEODHD       bool
// 	EODHDLimit     int
// 	UseGoogleCSE   bool
// 	GoogleCSELimit int
// 	SkipMarketaux  bool
// 	SkipFinnhub    bool
// 	SkipEODHD      bool
// 	SkipGoogleCSE  bool
// }

// var (
// 	config *NewsPipelineConfig
// 	logger *zap.SugaredLogger
// 	client *http.Client
// )

// // RunNewsPipeline fetches news concurrently from multiple sources, aggregates, deduplicates and returns unique articles.
// func RunNewsPipeline(ctx context.Context, company string, cfg *NewsPipelineConfig) ([]NewsArticle, error) {
// 	if cfg == nil {
// 		cfg = config
// 	}

// 	zap.L().Info("Starting news pipeline", zap.String("company", company))

// 	// Clone config to avoid mutating passed config with limit decrement
// 	cfgCopy := *cfg

// 	var (
// 		mu          sync.Mutex
// 		allArticles []NewsArticle
// 		errs        []error
// 		wg          sync.WaitGroup
// 	)

// 	fetchers := []struct {
// 		name  string
// 		use   bool
// 		skip  bool
// 		limit int
// 		fetch func(context.Context, string, *NewsPipelineConfig) ([]NewsArticle, error)
// 	}{
// 		{"Marketaux", cfgCopy.UseMarketaux, cfgCopy.SkipMarketaux, cfgCopy.MarketauxLimit, fetchFromMarketaux},
// 		{"Finnhub", cfgCopy.UseFinnhub, cfgCopy.SkipFinnhub, cfgCopy.FinnhubLimit, fetchFromFinnhub},
// 		{"EODHD", cfgCopy.UseEODHD, cfgCopy.SkipEODHD, cfgCopy.EODHDLimit, fetchFromEODHD},
// 		{"GoogleCSE", cfgCopy.UseGoogleCSE, cfgCopy.SkipGoogleCSE, cfgCopy.GoogleCSELimit, fetchFromGoogleCSE},
// 	}

// 	// ✅ Check if all sources are disabled or skipped
// 	allDisabled := true
// 	for _, fetcher := range fetchers {
// 		if fetcher.use && !fetcher.skip && fetcher.limit > 0 {
// 			allDisabled = false
// 			break
// 		}
// 	}
// 	if allDisabled {
// 		return nil, fmt.Errorf("no news source is enabled; please enable at least one source in configuration")
// 	}

// 	for _, fetcher := range fetchers {
// 		if !fetcher.use || fetcher.skip || fetcher.limit <= 0 {
// 			zap.L().Debug("Skipping source", zap.String("source", fetcher.name))
// 			continue
// 		}

// 		wg.Add(1)
// 		go func(name string, fetch func(context.Context, string, *NewsPipelineConfig) ([]NewsArticle, error)) {
// 			defer wg.Done()
// 			defer func() {
// 				if r := recover(); r != nil {
// 					zap.L().Error("Panic in news fetcher goroutine", zap.String("source", name), zap.Any("recover", r))
// 				}
// 			}()
// 			start := time.Now()
// 			newsFetchCount.WithLabelValues(name).Inc()

// 			articles, err := fetch(ctx, company, &cfgCopy)
// 			duration := time.Since(start).Seconds()
// 			newsFetchDuration.WithLabelValues(name).Observe(duration)

// 			mu.Lock()
// 			defer mu.Unlock()

// 			if err != nil {
// 				newsFetchErrors.WithLabelValues(name).Inc()
// 				zap.L().Error("Error fetching news", zap.String("source", name), zap.Error(err))
// 				errs = append(errs, fmt.Errorf("%s: %w", name, err))

// 				return
// 			}

// 			zap.L().Info("Fetched articles", zap.String("source", name), zap.Int("count", len(articles)), zap.Float64("duration_sec", duration))
// 			allArticles = append(allArticles, articles...)
// 		}(fetcher.name, fetcher.fetch)
// 	}

// 	wg.Wait()

// 	uniqueArticles := deduplicateArticles(allArticles)
// 	zap.L().Info("Deduplicated articles", zap.Int("count", len(uniqueArticles)))

// 	return uniqueArticles, errors.Join(errs...)
// }

// // --- Fetch implementations ---

// func fetchFromMarketaux(ctx context.Context, company string, cfg *NewsPipelineConfig) ([]NewsArticle, error) {
// 	if cfg.MarketauxLimit <= 0 {
// 		return nil, errors.New("marketaux limit reached")
// 	}
// 	// We don't mutate config here since we already cloned in RunNewsPipeline
// 	apiKey := os.Getenv("MARKETAUX_API_KEY")
// 	if apiKey == "" {
// 		return nil, errors.New("MARKETAUX_API_KEY not set")
// 	}
// 	url := fmt.Sprintf("https://api.marketaux.com/v1/news/all?filter_entities=true&entities=%s&api_token=%s", company, apiKey)

// 	body, err := doGetWithRetry(ctx, url)
// 	if err != nil {
// 		return nil, err
// 	}

// 	var resp struct {
// 		Data []struct {
// 			Title       string `json:"title"`
// 			Description string `json:"description"`
// 			URL         string `json:"url"`
// 			Source      string `json:"source"`
// 			PublishedAt string `json:"published_at"`
// 		} `json:"data"`
// 	}
// 	if err := json.Unmarshal(body, &resp); err != nil {
// 		return nil, fmt.Errorf("Marketaux JSON unmarshal failed: %w", err)
// 	}

// 	articles := make([]NewsArticle, 0, len(resp.Data))
// 	for _, item := range resp.Data {
// 		t, err := time.Parse(time.RFC3339, item.PublishedAt)
// 		if err != nil {
// 			zap.L().Warn("Failed to parse Marketaux published_at", zap.String("value", item.PublishedAt), zap.Error(err))
// 			t = time.Time{}
// 		}
// 		articles = append(articles, NewsArticle{
// 			Source:      item.Source,
// 			Title:       item.Title,
// 			Description: item.Description,
// 			URL:         item.URL,
// 			PublishedAt: t,
// 		})
// 	}
// 	return articles, nil
// }

// func fetchFromFinnhub(ctx context.Context, company string, cfg *NewsPipelineConfig) ([]NewsArticle, error) {
// 	if cfg.FinnhubLimit <= 0 {
// 		return nil, errors.New("finnhub limit reached")
// 	}
// 	apiKey := os.Getenv("FINNHUB_API_KEY")
// 	if apiKey == "" {
// 		return nil, errors.New("FINNHUB_API_KEY not set")
// 	}

// 	from := time.Now().AddDate(0, 0, -3).Format("2006-01-02")
// 	to := time.Now().Format("2006-01-02")
// 	url := fmt.Sprintf("https://finnhub.io/api/v1/company-news?symbol=%s&from=%s&to=%s&token=%s", company, from, to, apiKey)

// 	body, err := doGetWithRetry(ctx, url)
// 	if err != nil {
// 		return nil, err
// 	}

// 	var resp []struct {
// 		Headline string `json:"headline"`
// 		Source   string `json:"source"`
// 		URL      string `json:"url"`
// 		Datetime int64  `json:"datetime"` // unix timestamp
// 		Summary  string `json:"summary"`
// 	}

// 	if err := json.Unmarshal(body, &resp); err != nil {
// 		return nil, fmt.Errorf("Finnhub JSON unmarshal failed: %w", err)
// 	}

// 	articles := make([]NewsArticle, 0, len(resp))
// 	for _, item := range resp {
// 		t := time.Unix(item.Datetime, 0)
// 		articles = append(articles, NewsArticle{
// 			Source:      item.Source,
// 			Title:       item.Headline,
// 			Description: item.Summary,
// 			URL:         item.URL,
// 			PublishedAt: t,
// 		})
// 	}

// 	return articles, nil
// }

// func fetchFromEODHD(ctx context.Context, company string, cfg *NewsPipelineConfig) ([]NewsArticle, error) {
// 	if cfg.EODHDLimit <= 0 {
// 		return nil, errors.New("eodhd limit reached")
// 	}
// 	apiKey := os.Getenv("EODHD_API_KEY")
// 	if apiKey == "" {
// 		return nil, errors.New("EODHD_API_KEY not set")
// 	}

// 	url := fmt.Sprintf("https://eodhistoricaldata.com/api/news?api_token=%s&symbols=%s&period=d&limit=%d", apiKey, company, cfg.EODHDLimit)

// 	body, err := doGetWithRetry(ctx, url)
// 	if err != nil {
// 		return nil, err
// 	}

// 	var resp []struct {
// 		Title   string `json:"title"`
// 		URL     string `json:"url"`
// 		Source  string `json:"source"`
// 		PubDate string `json:"published_at"`
// 	}

// 	if err := json.Unmarshal(body, &resp); err != nil {
// 		return nil, fmt.Errorf("EODHD JSON unmarshal failed: %w", err)
// 	}

// 	articles := make([]NewsArticle, 0, len(resp))
// 	for _, item := range resp {
// 		t, err := time.Parse(time.RFC3339, item.PubDate)
// 		if err != nil {
// 			zap.L().Warn("Failed to parse EODHD published_at", zap.String("value", item.PubDate), zap.Error(err))
// 			t = time.Time{}
// 		}
// 		articles = append(articles, NewsArticle{
// 			Source:      item.Source,
// 			Title:       item.Title,
// 			Description: "",
// 			URL:         item.URL,
// 			PublishedAt: t,
// 		})
// 	}

// 	return articles, nil
// }

// func fetchFromGoogleCSE(ctx context.Context, company string, cfg *NewsPipelineConfig) ([]NewsArticle, error) {
// 	if cfg.GoogleCSELimit <= 0 {
// 		return nil, errors.New("google cse limit reached")
// 	}

// 	apiKey := os.Getenv("GOOGLE_CSE_API_KEY")
// 	cseID := os.Getenv("GOOGLE_CSE_CX_ID")
// 	if apiKey == "" || cseID == "" {
// 		return nil, errors.New("GOOGLE_CSE_API_KEY or GOOGLE_CSE_CX_ID not set")
// 	}

// 	url := fmt.Sprintf("https://www.googleapis.com/customsearch/v1?q=%s&cx=%s&key=%s&num=%d&sort=date", company, cseID, apiKey, cfg.GoogleCSELimit)

// 	body, err := doGetWithRetry(ctx, url)
// 	if err != nil {
// 		return nil, err
// 	}

// 	var resp struct {
// 		Items []struct {
// 			Title         string `json:"title"`
// 			Snippet       string `json:"snippet"`
// 			Link          string `json:"link"`
// 			DisplayLink   string `json:"displayLink"`
// 			FormattedTime string `json:"formattedTime,omitempty"`
// 			Pagemap       struct {
// 				Metatags []struct {
// 					ArticlePublishedTime string `json:"article:published_time"`
// 				} `json:"metatags"`
// 			} `json:"pagemap"`
// 		} `json:"items"`
// 	}

// 	if err := json.Unmarshal(body, &resp); err != nil {
// 		return nil, fmt.Errorf("Google CSE JSON unmarshal failed: %w", err)
// 	}

// 	articles := make([]NewsArticle, 0, len(resp.Items))
// 	for _, item := range resp.Items {
// 		var publishedAt time.Time

// 		// Try metatags first for published_time
// 		if len(item.Pagemap.Metatags) > 0 {
// 			pt := item.Pagemap.Metatags[0].ArticlePublishedTime
// 			if pt != "" {
// 				t, err := time.Parse(time.RFC3339, pt)
// 				if err == nil {
// 					publishedAt = t
// 				} else {
// 					zap.L().Warn("Failed to parse Google CSE article:published_time", zap.String("value", pt), zap.Error(err))
// 				}
// 			}
// 		}

// 		// fallback to zero time if no publishedAt found
// 		articles = append(articles, NewsArticle{
// 			Source:      item.DisplayLink,
// 			Title:       item.Title,
// 			Description: item.Snippet,
// 			URL:         item.Link,
// 			PublishedAt: publishedAt,
// 		})
// 	}

// 	return articles, nil
// }

// // doGetWithRetry is a helper that does a GET request with a retry on failure
// func doGetWithRetry(ctx context.Context, url string) ([]byte, error) {
// 	var lastErr error
// 	for i := 0; i < 3; i++ {
// 		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
// 		if err != nil {
// 			return nil, err
// 		}
// 		resp, err := client.Do(req)
// 		if err != nil {
// 			lastErr = err
// 			time.Sleep(500 * time.Millisecond)
// 			continue
// 		}

// 		if resp.StatusCode != http.StatusOK {
// 			body, _ := io.ReadAll(resp.Body)
// 			resp.Body.Close()
// 			lastErr = fmt.Errorf("status %d: %s", resp.StatusCode, string(body))
// 			time.Sleep(500 * time.Millisecond)
// 			continue
// 		}

// 		body, err := io.ReadAll(resp.Body)
// 		resp.Body.Close()
// 		if err != nil {
// 			lastErr = err
// 			time.Sleep(500 * time.Millisecond)
// 			continue
// 		}

// 		return body, nil
// 	}
// 	return nil, lastErr
// }

// // deduplicateArticles removes duplicates by URL and Title hash
// func deduplicateArticles(articles []NewsArticle) []NewsArticle {
// 	seen := make(map[string]struct{})
// 	result := make([]NewsArticle, 0, len(articles))

// 	for _, a := range articles {
// 		// Create a unique key by URL + title hash
// 		key := fmt.Sprintf("%s_%x", a.URL, sha1.Sum([]byte(a.Title)))
// 		if _, found := seen[key]; !found {
// 			seen[key] = struct{}{}
// 			result = append(result, a)
// 		}
// 	}

// 	return result
// }
