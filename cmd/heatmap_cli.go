package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/signal"
	"runtime"
	"sort"
	"sync"
	"syscall"
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
	case pct >= 10.0:
		return "\033[1;45m" // Magenta for >10%
	case pct >= 5.0:
		return "\033[1;42m" // Bright Green
	case pct >= 3.0:
		return "\033[42m" // Green
	case pct >= 1.5:
		return "\033[102m" // Light Green
	case pct >= 0.5:
		return "\033[47m" // White (neutral)
	case pct > -0.5:
		return "\033[103m" // Light Yellow
	case pct > -1.5:
		return "\033[43m" // Yellow/Orange
	case pct > -3.0:
		return "\033[41m" // Red
	case pct > -5.0:
		return "\033[1;41m" // Bright Red
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

func momentumScore(s *HeatmapStock) float64 {
	return s.PriceChangePct * (1 + 0.1*math.Log10(float64(s.Volume)+1))
}

func bidAskImbalance(bid, ask int64) float64 {
	if bid+ask == 0 {
		return 0
	}
	return float64(bid-ask) / float64(bid+ask)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Print list outside of main loop for speed
func printList(list []*StockAnalysis, title, color string) {
	fmt.Printf("\n%s%s%s\n", color, title, "\033[0m")
	for _, s := range list {
		bg := getBgColorForChange(s.PriceChangePct)
		blackText := "\033[30m"
		reset := "\033[0m"
		flag := s.Flag
		if flag == "🔥" {
			flag = "\033[1;33m🔥\033[0m"
		}
		fmt.Printf("%s%s%-3s %-18.18s %-10.2f %-10.2f %-10.2f %-10.4f %-10d %-10d %-10d %-8.2f %-8.2f %-8.2f%s\n",
			bg, blackText, flag, s.Symbol, s.LastPrice, s.BidPrice, s.AskPrice, s.BidAskSpread, s.BidDepth, s.AskDepth, s.Volume, s.PriceChangePct, s.Momentum, s.Imbalance, reset)
	}
}

type StockAnalysis struct {
	*HeatmapStock
	Momentum  float64
	Imbalance float64
	Flag      string
}

// Parallel analysis for large stock lists
func analyzeStocksParallel(stocks []*HeatmapStock) []*StockAnalysis {
	n := len(stocks)
	if n == 0 {
		return nil
	}
	numCPU := runtime.NumCPU()
	chunk := (n + numCPU - 1) / numCPU

	out := make([]*StockAnalysis, n)
	var wg sync.WaitGroup

	for i := 0; i < numCPU; i++ {
		start := i * chunk
		end := min((i+1)*chunk, n)
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for j := start; j < end; j++ {
				s := stocks[j]
				mom := momentumScore(s)
				imb := bidAskImbalance(s.BidDepth, s.AskDepth)
				flag := ""
				if s.PriceChangePct > 2 && s.Volume > 100000 && abs(imb) > 0.5 {
					flag = "🔥"
				} else if s.PriceChangePct > 2 && s.Volume > 100000 {
					flag = "🚀"
				} else if s.PriceChangePct < -2 && s.BidDepth > s.AskDepth*2 {
					flag = "🔄"
				} else if abs(imb) > 0.5 {
					flag = "⚡"
				}
				out[j] = &StockAnalysis{
					HeatmapStock: s,
					Momentum:     mom,
					Imbalance:    imb,
					Flag:         flag,
				}
			}
		}(start, end)
	}
	wg.Wait()
	return out
}

// Fault-tolerant, reconnecting WebSocket reader with fast UI
func RenderHeatmapWS(wsURL string, stop <-chan struct{}, wg *sync.WaitGroup) {
	defer wg.Done()
	var lastUpdate time.Time
	const refreshInterval = 30 * time.Millisecond // 8 FPS, adjust as needed

	for {
		select {
		case <-stop:
			return
		default:
		}

		c, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		if err != nil {
			fmt.Println("WebSocket dial error:", err)
			time.Sleep(2 * time.Second)
			continue
		}
		fmt.Println("Connected to WebSocket server.")
		c.SetReadLimit(1 << 20) // 1MB

		dataCh := make(chan []*HeatmapStock, 2)
		errCh := make(chan error, 1)

		// Reader goroutine
		go func() {
			defer close(dataCh)
			for {
				_, msg, err := c.ReadMessage()
				if err != nil {
					errCh <- err
					return
				}
				var stocks []*HeatmapStock
				if err := json.Unmarshal(msg, &stocks); err != nil {
					fmt.Println("JSON unmarshal error:", err)
					continue
				}
				select {
				case dataCh <- stocks:
				default:
					// Drop if UI is lagging
				}
			}
		}()

	loop:
		for {
			select {
			case <-stop:
				c.Close()
				break loop
			case err := <-errCh:
				fmt.Println("WebSocket read error:", err)
				c.Close()
				time.Sleep(2 * time.Second)
				break loop
			case stocks, ok := <-dataCh:
				if !ok {
					break loop
				}
				now := time.Now()
				if now.Sub(lastUpdate) < refreshInterval {
					continue // throttle UI
				}
				lastUpdate = now

				// Parallel Analysis
				analysis := analyzeStocksParallel(stocks)

				// Intraday Suggest Buy and Short Sell logic
				suggestBuy := make([]*StockAnalysis, 0, 8)
				suggestSell := make([]*StockAnalysis, 0, 8)
				for _, s := range analysis {
					if s.PriceChangePct > 1.2 && s.PriceChangePct < 5.0 && s.Volume > 150000 && s.Imbalance > 0.25 {
						suggestBuy = append(suggestBuy, s)
					}
					if s.PriceChangePct < -1.2 && s.PriceChangePct > -5.0 && s.Volume > 150000 && s.Imbalance < -0.25 {
						suggestSell = append(suggestSell, s)
					}
				}
				sort.Slice(suggestBuy, func(i, j int) bool { return suggestBuy[i].Momentum > suggestBuy[j].Momentum })
				sort.Slice(suggestSell, func(i, j int) bool { return suggestSell[i].Momentum < suggestSell[j].Momentum })

				sort.Slice(analysis, func(i, j int) bool { return analysis[i].Momentum > analysis[j].Momentum })
				topMomentum := analysis[:min(10, len(analysis))]

				sort.Slice(analysis, func(i, j int) bool { return analysis[i].PriceChangePct < analysis[j].PriceChangePct })
				topReversal := analysis[:min(5, len(analysis))]

				sort.Slice(analysis, func(i, j int) bool { return abs(analysis[i].Imbalance) > abs(analysis[j].Imbalance) })
				topImbalance := analysis[:min(5, len(analysis))]

				// UI
				fmt.Print("\033[2J\033[H")
				fmt.Printf("\033[1;37mHeatmap Updated: %s\033[0m\n", time.Now().Format("15:04:05"))
				fmt.Printf("\033[1;36m%-3s %-18s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-8s %-8s %-8s\033[0m\n",
					"F", "Symbol", "LTP", "Bid", "Ask", "Spread", "BidQty", "AskQty", "Volume", "%Chg", "Mom", "Imb")

				printList(suggestBuy[:min(5, len(suggestBuy))], "Suggest Buy (Intraday) 🟢", "\033[1;32m")
				printList(suggestSell[:min(5, len(suggestSell))], "Suggest Short Sell (Intraday) 🔴", "\033[1;31m")
				printList(topMomentum, "Top 10 Momentum 🚀", "\033[1;34m")
				printList(topReversal, "Top 5 Reversal 🔄", "\033[1;35m")
				printList(topImbalance, "Top 5 Imbalance ⚡", "\033[1;36m")

				fmt.Println("\n\033[1;37mLegend: 🚀 Momentum  🔄 Reversal  ⚡ Imbalance  🔥 Sureshot\033[0m")
			}
		}
	}
}

func main() {
	wsURL := "ws://localhost:8000/ws/heatmap"
	stop := make(chan struct{})
	var wg sync.WaitGroup

	wg.Add(1)
	go RenderHeatmapWS(wsURL, stop, &wg)

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	<-sig
	close(stop)
	wg.Wait()
	fmt.Println("Heatmap CLI exited cleanly.")
}
