package data

import (
	"fmt"
	"sync"
	"time"
)

type OrderBookLevel struct {
	Price  float64
	Volume int64
}

type HeatmapStock struct {
	Symbol         string
	LastPrice      float64
	BidPrice       float64
	AskPrice       float64
	BidAskSpread   float64
	BidDepth       int64
	AskDepth       int64
	Volume         int64
	VolumeAtPrice  map[string]int64 // price -> volume
	LastUpdated    time.Time
	PriceChangePct float64 // <-- Add this line
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

// UpdateHeatmap updates the heatmap with new tick/order book data
func (hm *MarketHeatmap) Update(symbol string, lastPrice, bidPrice, askPrice float64, bidDepth, askDepth, volume int64, priceLevel float64, prevClose float64) {
	hm.mu.Lock()
	defer hm.mu.Unlock()
	stock, ok := hm.Stocks[symbol]
	if !ok {
		stock = &HeatmapStock{
			Symbol:        symbol,
			VolumeAtPrice: make(map[string]int64),
		}
		hm.Stocks[symbol] = stock
	}
	stock.LastPrice = lastPrice
	stock.BidPrice = bidPrice
	stock.AskPrice = askPrice
	stock.BidAskSpread = askPrice - bidPrice
	stock.BidDepth = bidDepth
	stock.AskDepth = askDepth
	stock.Volume = volume
	priceKey := fmt.Sprintf("%.2f", priceLevel)
	stock.VolumeAtPrice[priceKey] += 1
	stock.LastUpdated = time.Now()
	if prevClose > 0 {
		stock.PriceChangePct = ((lastPrice - prevClose) / prevClose) * 100
	} else {
		stock.PriceChangePct = 0
	}
}

// Snapshot returns a copy for safe concurrent read
func (hm *MarketHeatmap) Snapshot() []*HeatmapStock {
	hm.mu.RLock()
	defer hm.mu.RUnlock()
	snapshot := make([]*HeatmapStock, 0, len(hm.Stocks))
	for _, s := range hm.Stocks {
		// Deep copy if needed
		cp := *s
		cp.VolumeAtPrice = make(map[string]int64)
		for k, v := range s.VolumeAtPrice {
			cp.VolumeAtPrice[k] = v
		}
		snapshot = append(snapshot, &cp)
	}
	return snapshot
}
