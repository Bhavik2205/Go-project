package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"sort"
	"time"

	"github.com/gorilla/websocket"
)

type HeatmapStock struct {
	Symbol         string
	LastPrice      float64
	BidPrice       float64
	AskPrice       float64
	BidAskSpread   float64
	BidDepth       int64
	AskDepth       int64
	Volume         int64
	VolumeAtPrice  map[string]int64
	LastUpdated    time.Time
	PriceChangePct float64
}

func getBgColorForChange(pct float64) string {
	switch {
	case pct >= 3.0:
		return "\033[1;42m" // Bright Green
	case pct >= 1.5:
		return "\033[42m" // Green
	case pct >= 0.5:
		return "\033[102m" // Light Green
	case pct > -0.5:
		return "\033[47m" // White (neutral)
	case pct > -1.5:
		return "\033[103m" // Light Yellow
	case pct > -3.0:
		return "\033[43m" // Yellow/Orange
	case pct > -5.0:
		return "\033[41m" // Red
	default:
		return "\033[1;41m" // Bright Red
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func inList(s *HeatmapStock, list []*HeatmapStock) bool {
	for _, x := range list {
		if x.Symbol == s.Symbol {
			return true
		}
	}
	return false
}

func RenderHeatmapWS(wsURL string) {
	c, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		fmt.Println("WebSocket dial error:", err)
		return
	}
	defer c.Close()

	interrupt := make(chan os.Signal, 1)
	signal.Notify(interrupt, os.Interrupt)

	for {
		select {
		case <-interrupt:
			fmt.Println("Interrupted, exiting.")
			return
		default:
			_, msg, err := c.ReadMessage()
			if err != nil {
				fmt.Println("WebSocket read error:", err)
				return
			}
			var stocks []*HeatmapStock
			if err := json.Unmarshal(msg, &stocks); err != nil {
				fmt.Println("JSON unmarshal error:", err)
				continue
			}

			// Sort by % change descending
			sort.Slice(stocks, func(i, j int) bool {
				return stocks[i].PriceChangePct > stocks[j].PriceChangePct
			})

			// Top 12 gainers
			gainers := []*HeatmapStock{}
			for _, s := range stocks {
				if s.PriceChangePct > 0 {
					gainers = append(gainers, s)
				}
				if len(gainers) == 12 {
					break
				}
			}

			// Top 12 losers
			losers := []*HeatmapStock{}
			for i := len(stocks) - 1; i >= 0 && len(losers) < 12; i-- {
				s := stocks[i]
				if s.PriceChangePct < 0 {
					losers = append(losers, s)
				}
			}
			// Reverse losers to show biggest loser first
			for i, j := 0, len(losers)-1; i < j; i, j = i+1, j-1 {
				losers[i], losers[j] = losers[j], losers[i]
			}

			// Most active by volume (excluding already picked)
			used := map[string]bool{}
			for _, s := range gainers {
				used[s.Symbol] = true
			}
			for _, s := range losers {
				used[s.Symbol] = true
			}
			sort.Slice(stocks, func(i, j int) bool {
				return stocks[i].Volume > stocks[j].Volume
			})
			mostActive := []*HeatmapStock{}
			for _, s := range stocks {
				if !used[s.Symbol] {
					mostActive = append(mostActive, s)
					used[s.Symbol] = true
				}
				if len(mostActive) == 8 {
					break
				}
			}

			// 4 most neutral (closest to zero, not already picked)
			neutral := make([]*HeatmapStock, len(stocks))
			copy(neutral, stocks)
			sort.Slice(neutral, func(i, j int) bool {
				return abs(neutral[i].PriceChangePct) < abs(neutral[j].PriceChangePct)
			})
			neutrals := []*HeatmapStock{}
			for _, s := range neutral {
				if !used[s.Symbol] {
					neutrals = append(neutrals, s)
					used[s.Symbol] = true
				}
				if len(neutrals) == 4 {
					break
				}
			}

			// Combine all for display (total 36, but if overlap is less, fill up to 40)
			display := append(gainers, neutrals...)
			display = append(display, mostActive...)
			display = append(display, losers...)

			// Fill up to 40 with remaining stocks if needed
			if len(display) < 40 {
				for _, s := range stocks {
					if !inList(s, display) {
						display = append(display, s)
					}
					if len(display) == 40 {
						break
					}
				}
			} else if len(display) > 40 {
				display = display[:40]
			}

			fmt.Print("\033[2J\033[H")
			fmt.Printf("%-18s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-8s\n",
				"Symbol", "LTP", "Bid", "Ask", "Spread", "BidQty", "AskQty", "Volume", "%Change")

			for _, s := range display {
				bg := getBgColorForChange(s.PriceChangePct)
				if inList(s, mostActive) {
					bg = "\033[44m" // Blue for most active by volume
				}
				// Always black text
				blackText := "\033[30m"
				reset := "\033[0m"
				fmt.Printf("%s%s%-18.18s %-10.2f %-10.2f %-10.2f %-10.4f %-10d %-10d %-10d %-8.2f%%%s\n",
					bg, blackText, s.Symbol, s.LastPrice, s.BidPrice, s.AskPrice, s.BidAskSpread, s.BidDepth, s.AskDepth, s.Volume, s.PriceChangePct, reset)
			}
			time.Sleep(120 * time.Millisecond)
		}
	}
}

func main() {
	wsURL := "ws://localhost:8080/ws/heatmap"
	go RenderHeatmapWS(wsURL)
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)
	<-c
}
