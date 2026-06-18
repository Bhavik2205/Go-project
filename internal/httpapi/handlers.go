// internal/httpapi/handlers.go
package httpapi

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/cache"
	"github.com/Bhavik2205/ML-Bot/internal/contracts"
	"github.com/Bhavik2205/ML-Bot/internal/data"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/marketdata/candles"
	"go.uber.org/zap"
)

type healthDependencies struct {
	Postgres string `json:"postgres"`
	Redis    string `json:"redis"`
	Zerodha  string `json:"zerodha"`
}

type healthData struct {
	Status        string             `json:"status"`
	Service       string             `json:"service"`
	Version       string             `json:"version"`
	UptimeSeconds int64              `json:"uptimeSeconds"`
	Mode          string             `json:"mode"`
	Dependencies  healthDependencies `json:"dependencies"`
}

type brokerStatusData struct {
	Broker         string     `json:"broker"`
	Connected      bool       `json:"connected"`
	BrokerUserID   string     `json:"brokerUserId,omitempty"`
	AccountName    string     `json:"accountName,omitempty"`
	SessionExpiry  *time.Time `json:"sessionExpiry,omitempty"`
	LastSyncedAt   *time.Time `json:"lastSyncedAt,omitempty"`
	TradingEnabled bool       `json:"tradingEnabled"`
}

type quoteOHLC struct {
	Open  float64 `json:"open"`
	High  float64 `json:"high"`
	Low   float64 `json:"low"`
	Close float64 `json:"close"`
}

type quoteItem struct {
	Symbol          string    `json:"symbol"`
	InstrumentToken int       `json:"instrumentToken"`
	LastPrice       float64   `json:"lastPrice"`
	NetChange       float64   `json:"netChange"`
	PercentChange   float64   `json:"percentChange"`
	VolumeTraded    int64     `json:"volumeTraded"`
	OHLC            quoteOHLC `json:"ohlc"`
	UpdatedAt       time.Time `json:"updatedAt"`
}

type marketBreadth struct {
	Advancers int `json:"advancers"`
	Decliners int `json:"decliners"`
	Unchanged int `json:"unchanged"`
}

type marketOverviewItem struct {
	Symbol        string    `json:"symbol"`
	LastPrice     float64   `json:"lastPrice"`
	PercentChange float64   `json:"percentChange"`
	Volume        int64     `json:"volume"`
	Bid           float64   `json:"bid"`
	Ask           float64   `json:"ask"`
	UpdatedAt     time.Time `json:"updatedAt"`
}

type marketOverviewData struct {
	Indices            []marketOverviewItem `json:"indices"`
	TopGainers         []marketOverviewItem `json:"topGainers"`
	TopLosers          []marketOverviewItem `json:"topLosers"`
	MostActiveByVolume []marketOverviewItem `json:"mostActiveByVolume"`
	MarketBreadth      marketBreadth        `json:"marketBreadth"`
	UpdatedAt          time.Time            `json:"updatedAt"`
}

// ---- Handlers ----

func handleV1Health(w http.ResponseWriter, r *http.Request, deps HTTPDeps) {
	requestID := getRequestID(r)
	mode := "live"
	if deps.AppConfig != nil && deps.AppConfig.Market.Simulate {
		mode = "simulation"
	}

	resp := healthData{
		Status:        "ok",
		Service:       "go-project",
		Version:       "0.1.0",
		UptimeSeconds: int64(time.Since(deps.StartupTime).Seconds()),
		Mode:          mode,
		Dependencies: healthDependencies{
			Postgres: dependencyStatusPostgres(deps.DBClient),
			Redis:    dependencyStatusRedis(deps.RedisClient),
			Zerodha:  dependencyStatusZerodha(deps.ZerodhaClient),
		},
	}
	writeSuccess(w, http.StatusOK, requestID, resp)
}

func handleV1BrokerStatus(w http.ResponseWriter, r *http.Request, deps HTTPDeps) {
	requestID := getRequestID(r)
	now := time.Now()
	status := brokerStatusData{
		Broker:         "ZERODHA",
		Connected:      false,
		LastSyncedAt:   &now,
		TradingEnabled: false,
	}

	client, ok := deps.ZerodhaClient.(*api.ZerodhaClient)
	if !ok || client == nil || client.Kite == nil {
		writeSuccess(w, http.StatusOK, requestID, status)
		return
	}

	profile, err := client.Kite.GetUserProfile()
	if err != nil {
		zap.L().Warn("Unable to fetch Zerodha profile for broker status", zap.Error(err))
		writeSuccess(w, http.StatusOK, requestID, status)
		return
	}

	accountName := profile.UserName
	if accountName == "" {
		accountName = profile.UserShortName
	}

	status.Connected = true
	status.BrokerUserID = profile.UserID
	status.AccountName = accountName
	status.TradingEnabled = true
	writeSuccess(w, http.StatusOK, requestID, status)
}

func handleV1Quotes(w http.ResponseWriter, r *http.Request, deps HTTPDeps) {
	requestID := getRequestID(r)
	symbolsParam := r.URL.Query().Get("symbols")
	if strings.TrimSpace(symbolsParam) == "" {
		writeError(w, http.StatusBadRequest, requestID, "VALIDATION_ERROR", "symbols query parameter is required", map[string]string{"field": "symbols"})
		return
	}

	symbols := parseSymbols(symbolsParam)
	items := fetchQuotes(symbols, deps)
	writeSuccess(w, http.StatusOK, requestID, items)
}

func handleV1MarketOverview(w http.ResponseWriter, r *http.Request, deps HTTPDeps) {
	requestID := getRequestID(r)
	snapshot := data.GetMarketHeatmap().Snapshot()
	items := make([]marketOverviewItem, 0, len(snapshot))
	var updatedAt time.Time
	breadth := marketBreadth{}

	for _, stock := range snapshot {
		item := marketOverviewItem{
			Symbol:        normalizeToAPISymbol(stock.Symbol),
			LastPrice:     float64(stock.LastPrice) / candles.PriceScale,
			PercentChange: stock.PriceChangePct,
			Volume:        stock.Volume,
			Bid:           float64(stock.BidPrice) / candles.PriceScale,
			Ask:           float64(stock.AskPrice) / candles.PriceScale,
			UpdatedAt:     stock.LastUpdated,
		}
		items = append(items, item)
		if stock.LastUpdated.After(updatedAt) {
			updatedAt = stock.LastUpdated
		}
		switch {
		case stock.PriceChangePct > 0:
			breadth.Advancers++
		case stock.PriceChangePct < 0:
			breadth.Decliners++
		default:
			breadth.Unchanged++
		}
	}

	byGainers := append([]marketOverviewItem(nil), items...)
	byLosers := append([]marketOverviewItem(nil), items...)
	byVolume := append([]marketOverviewItem(nil), items...)

	sort.Slice(byGainers, func(i, j int) bool { return byGainers[i].PercentChange > byGainers[j].PercentChange })
	sort.Slice(byLosers, func(i, j int) bool { return byLosers[i].PercentChange < byLosers[j].PercentChange })
	sort.Slice(byVolume, func(i, j int) bool { return byVolume[i].Volume > byVolume[j].Volume })

	indices := make([]marketOverviewItem, 0, 4)
	for _, item := range items {
		if strings.Contains(item.Symbol, "NIFTY") || strings.Contains(item.Symbol, "SENSEX") {
			indices = append(indices, item)
		}
	}

	resp := marketOverviewData{
		Indices:            limitMarketItems(indices, 6),
		TopGainers:         limitMarketItems(byGainers, 5),
		TopLosers:          limitMarketItems(byLosers, 5),
		MostActiveByVolume: limitMarketItems(byVolume, 5),
		MarketBreadth:      breadth,
		UpdatedAt:          updatedAt,
	}
	// Ensure slices are never null in JSON
	if resp.Indices == nil {
		resp.Indices = []marketOverviewItem{}
	}
	if resp.TopGainers == nil {
		resp.TopGainers = []marketOverviewItem{}
	}
	if resp.TopLosers == nil {
		resp.TopLosers = []marketOverviewItem{}
	}
	if resp.MostActiveByVolume == nil {
		resp.MostActiveByVolume = []marketOverviewItem{}
	}
	writeSuccess(w, http.StatusOK, requestID, resp)
}

func handleV1OpenAPISpec(w http.ResponseWriter, r *http.Request) {
	spec := map[string]any{
		"openapi": "3.0.3",
		"info": map[string]any{
			"title":       "TradingBot API",
			"version":     "0.1.0",
			"description": "Interactive-first API surface for signal-execution-desk.",
		},
		"servers": []map[string]string{
			{"url": "http://localhost:8000"},
		},
		"paths": map[string]any{
			"/api/v1/health": map[string]any{
				"get": map[string]any{
					"summary":     "Service health and dependency status",
					"operationId": "getHealth",
				},
			},
			"/api/v1/brokers/zerodha/status": map[string]any{
				"get": map[string]any{
					"summary":     "Current broker connectivity status",
					"operationId": "getBrokerStatus",
				},
			},
			"/api/v1/quotes": map[string]any{
				"get": map[string]any{
					"summary":     "Batch quote lookup",
					"operationId": "getQuotes",
					"parameters": []map[string]any{
						{
							"name":        "symbols",
							"in":          "query",
							"required":    true,
							"description": "Comma-separated symbols like NSE:RELIANCE,NSE:TCS",
							"schema":      map[string]string{"type": "string"},
							"example":     "NSE:RELIANCE,NSE:TCS",
						},
					},
				},
			},
			"/api/v1/market/overview": map[string]any{
				"get": map[string]any{
					"summary":     "Top gainers, losers, activity, and market breadth",
					"operationId": "getMarketOverview",
				},
			},
			"/api/v1/openapi.json": map[string]any{
				"get": map[string]any{
					"summary":     "This OpenAPI-style document",
					"operationId": "getOpenAPI",
				},
			},
		},
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(spec)
}

// ---- Helper functions ----

func getRequestID(r *http.Request) string {
	if value := strings.TrimSpace(r.Header.Get("X-Request-ID")); value != "" {
		return value
	}
	return fmt.Sprintf("req_%d", time.Now().UnixNano())
}

func writeSuccess[T any](w http.ResponseWriter, statusCode int, requestID string, data T) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	if err := json.NewEncoder(w).Encode(contracts.NewSuccess(requestID, data)); err != nil {
		zap.L().Error("Failed to encode success response", zap.Error(err))
	}
}

func writeError(w http.ResponseWriter, statusCode int, requestID, code, message string, details any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	if err := json.NewEncoder(w).Encode(contracts.NewError(requestID, code, message, details)); err != nil {
		zap.L().Error("Failed to encode error response", zap.Error(err))
	}
}

func dependencyStatusPostgres(dbClient *db.DBClient) string {
	if dbClient == nil || dbClient.DB == nil {
		return "not_configured"
	}
	sqlDB, err := dbClient.DB.DB()
	if err != nil {
		return "error"
	}
	if err := sqlDB.Ping(); err != nil {
		return "error"
	}
	return "ok"
}

func dependencyStatusRedis(redisClient *cache.RedisClient) string {
	if redisClient == nil || redisClient.Client == nil {
		return "not_configured"
	}
	if err := redisClient.Client.Ping(context.Background()).Err(); err != nil {
		return "error"
	}
	return "ok"
}

func dependencyStatusZerodha(zc api.ZerodhaClientInterface) string {
	client, ok := zc.(*api.ZerodhaClient)
	if !ok || client == nil || client.Kite == nil {
		return "not_configured"
	}
	type result struct{ status string }
	ch := make(chan result, 1)
	go func() {
		if _, err := client.Kite.GetUserProfile(); err != nil {
			ch <- result{"error"}
		} else {
			ch <- result{"ok"}
		}
	}()
	select {
	case r := <-ch:
		return r.status
	case <-time.After(3 * time.Second):
		return "timeout"
	}
}

func parseSymbols(raw string) []string {
	parts := strings.Split(raw, ",")
	result := make([]string, 0, len(parts))
	seen := map[string]struct{}{}
	for _, part := range parts {
		normalized := normalizeToAPISymbol(part)
		if normalized == "" {
			continue
		}
		if _, exists := seen[normalized]; exists {
			continue
		}
		seen[normalized] = struct{}{}
		result = append(result, normalized)
	}
	return result
}

func normalizeToAPISymbol(raw string) string {
	value := strings.TrimSpace(strings.ToUpper(raw))
	if value == "" {
		return ""
	}
	if strings.Contains(value, "(") && strings.Contains(value, ")") {
		open := strings.LastIndex(value, "(")
		close := strings.LastIndex(value, ")")
		if open > 0 && close > open {
			symbol := strings.TrimSpace(value[:open])
			exchange := strings.TrimSpace(value[open+1 : close])
			return exchange + ":" + symbol
		}
	}
	if strings.Contains(value, ":") {
		return value
	}
	return "NSE:" + value
}

func fetchQuotes(symbols []string, deps HTTPDeps) []quoteItem {
	itemsBySymbol := map[string]quoteItem{}
	now := time.Now()

	client, ok := deps.ZerodhaClient.(*api.ZerodhaClient)
	if ok && client != nil && client.Kite != nil && len(symbols) > 0 {
		quoteMap, err := client.GetQuote(symbols...)
		if err != nil {
			zap.L().Warn("Failed to fetch quotes from Zerodha, falling back to heatmap snapshot", zap.Error(err))
		} else {
			// Access the inner map of quotes
			for requestedSymbol, quote := range quoteMap {
				updatedAt := now
				if !quote.Timestamp.Time.IsZero() {
					updatedAt = quote.Timestamp.Time
				}
				percentChange := percentChangeFromNet(quote.LastPrice, quote.NetChange)
				itemsBySymbol[normalizeToAPISymbol(requestedSymbol)] = quoteItem{
					Symbol:          normalizeToAPISymbol(requestedSymbol),
					InstrumentToken: quote.InstrumentToken,
					LastPrice:       quote.LastPrice,
					NetChange:       quote.NetChange,
					PercentChange:   percentChange,
					VolumeTraded:    int64(quote.Volume),
					OHLC: quoteOHLC{
						Open:  quote.OHLC.Open,
						High:  quote.OHLC.High,
						Low:   quote.OHLC.Low,
						Close: quote.OHLC.Close,
					},
					UpdatedAt: updatedAt,
				}
			}
		}
	}

	if len(itemsBySymbol) < len(symbols) {
		snapshotBySymbol := map[string]*data.HeatmapStock{}
		for _, stock := range data.GetMarketHeatmap().Snapshot() {
			snapshotBySymbol[normalizeToAPISymbol(stock.Symbol)] = stock
		}
		for _, symbol := range symbols {
			if _, exists := itemsBySymbol[symbol]; exists {
				continue
			}
			if stock, exists := snapshotBySymbol[symbol]; exists {
				netChange := netChangeFromPercent(float64(stock.LastPrice)/candles.PriceScale, stock.PriceChangePct)
				itemsBySymbol[symbol] = quoteItem{
					Symbol:          symbol,
					InstrumentToken: lookupInstrumentToken(symbol, deps),
					LastPrice:       float64(stock.LastPrice) / candles.PriceScale,
					NetChange:       netChange,
					PercentChange:   stock.PriceChangePct,
					VolumeTraded:    stock.Volume,
					OHLC:            quoteOHLC{},
					UpdatedAt:       stock.LastUpdated,
				}
			}
		}
	}

	result := make([]quoteItem, 0, len(symbols))
	for _, symbol := range symbols {
		if item, exists := itemsBySymbol[symbol]; exists {
			result = append(result, item)
		} else {
			result = append(result, quoteItem{
				Symbol:    symbol,
				UpdatedAt: now,
			})
		}
	}
	return result
}

func lookupInstrumentToken(symbol string, deps HTTPDeps) int {
	if deps.DBClient == nil || deps.DBClient.DB == nil {
		return 0
	}
	parts := strings.SplitN(symbol, ":", 2)
	if len(parts) != 2 {
		return 0
	}
	var instrument db.Instrument
	if err := deps.DBClient.DB.Where("exchange = ? AND tradingsymbol = ?", parts[0], parts[1]).First(&instrument).Error; err != nil {
		return 0
	}
	return int(instrument.InstrumentToken)
}

func percentChangeFromNet(lastPrice, netChange float64) float64 {
	previousClose := lastPrice - netChange
	if previousClose == 0 {
		return 0
	}
	return (netChange / previousClose) * 100
}

func netChangeFromPercent(lastPrice, percent float64) float64 {
	if percent == -100 {
		return 0
	}
	previousClose := lastPrice / (1 + percent/100)
	return lastPrice - previousClose
}

func limitMarketItems(items []marketOverviewItem, size int) []marketOverviewItem {
	if len(items) <= size {
		return items
	}
	return items[:size]
}
