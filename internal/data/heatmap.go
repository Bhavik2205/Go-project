package data

import (
	"sync"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/marketdata/candles"
)

type OrderBookLevel struct {
	Price  int64
	Volume int64
}

type HeatmapStock struct {
	Symbol         string
	LastPrice      int64
	BidPrice       int64
	AskPrice       int64
	BidAskSpread   int64
	BidDepth       int64
	AskDepth       int64
	Volume         int64
	VolumeAtPrice  map[int64]int64 // scaled_price -> volume
	LastUpdated    time.Time
	PriceChangePct float64 // ratio kept as float64; computed once on update
}

type MarketHeatmap struct {
	Stocks map[string]*HeatmapStock
	mu     sync.RWMutex
}

func NewMarketHeatmap() *MarketHeatmap {
	return &MarketHeatmap{
		Stocks: make(map[string]*HeatmapStock),
	}
}

var globalMarketHeatmap = NewMarketHeatmap()

func GetMarketHeatmap() *MarketHeatmap {
	return globalMarketHeatmap
}

// Update updates the heatmap with new tick/order book data.
// All prices must already be unscaled float64 (as received from NormalizedTick);
// they are scaled to int64 here before storage.
func (hm *MarketHeatmap) Update(symbol string, lastPrice, bidPrice, askPrice float64, bidDepth, askDepth, volume int64, priceLevel float64, prevClose float64) {
	scaledLast := int64(lastPrice * candles.PriceScale)
	scaledBid := int64(bidPrice * candles.PriceScale)
	scaledAsk := int64(askPrice * candles.PriceScale)
	scaledLevel := int64(priceLevel * candles.PriceScale)

	hm.mu.Lock()
	defer hm.mu.Unlock()
	stock, ok := hm.Stocks[symbol]
	if !ok {
		stock = &HeatmapStock{
			Symbol:        symbol,
			VolumeAtPrice: make(map[int64]int64),
		}
		hm.Stocks[symbol] = stock
	}
	stock.LastPrice = scaledLast
	stock.BidPrice = scaledBid
	stock.AskPrice = scaledAsk
	stock.BidAskSpread = scaledAsk - scaledBid
	stock.BidDepth = bidDepth
	stock.AskDepth = askDepth
	stock.Volume = volume
	stock.VolumeAtPrice[scaledLevel]++
	// Prune to prevent unbounded memory growth — keep only the 100 most recent price levels.
	if len(stock.VolumeAtPrice) > 100 {
		stock.VolumeAtPrice = make(map[int64]int64)
	}
	stock.LastUpdated = time.Now()
	if prevClose > 0 {
		stock.PriceChangePct = ((lastPrice - prevClose) / prevClose) * 100
	} else {
		stock.PriceChangePct = 0
	}
}

// Snapshot returns a copy for safe concurrent read.
// Price fields are unscaled back to float64 so callers in the JSON/WS layer
// can use them directly without knowing about PriceScale.
func (hm *MarketHeatmap) Snapshot() []*HeatmapStock {
	hm.mu.RLock()
	defer hm.mu.RUnlock()
	snapshot := make([]*HeatmapStock, 0, len(hm.Stocks))
	for _, s := range hm.Stocks {
		cp := *s
		cp.VolumeAtPrice = make(map[int64]int64, len(s.VolumeAtPrice))
		for k, v := range s.VolumeAtPrice {
			cp.VolumeAtPrice[k] = v
		}
		snapshot = append(snapshot, &cp)
	}
	return snapshot
}
