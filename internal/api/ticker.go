// api/subscribe.go
package api

import (
	"fmt"
	"log"

	kitemodels "github.com/zerodha/gokiteconnect/v4/models"
	kiteticker "github.com/zerodha/gokiteconnect/v4/ticker"
)

func (z *ZerodhaClient) SubscribeToTicks(infos []*InstrumentInfo) error {
	tokens := make([]uint32, 0, len(infos))
	for _, info := range infos {
		tokens = append(tokens, info.Token)
	}
	tokenToLabel := make(map[uint32]string)
	for _, instr := range infos {
		tokens = append(tokens, instr.Token)
		tokenToLabel[instr.Token] = fmt.Sprintf("%s (%s)", instr.Symbol, instr.Exchange)
	}

	z.Ticker = kiteticker.New(z.APIKey, z.AccessToken)

	z.Ticker.OnConnect(func() {
		fmt.Println("✅ Connected to Zerodha WebSocket.")
		for _, info := range infos {
			fmt.Printf("🔗 Subscribed to %s on %s (Token: %d)\n", info.Symbol, info.Exchange, info.Token)
		}
		if err := z.Ticker.Subscribe(tokens); err != nil {
			log.Printf("❌ Subscribe error: %v", err)
		}
		if err := z.Ticker.SetMode(kiteticker.ModeFull, tokens); err != nil {
			log.Printf("❌ SetMode error: %v", err)
		}
	})

	// Use a map to track last prices and volumes per token
	lastPrices := make(map[uint32]float32)
	lastVolumes := make(map[uint32]int)

	z.Ticker.OnTick(func(tick kitemodels.Tick) {
		prevPrice := lastPrices[tick.InstrumentToken]
		currentPrice := float32(tick.LastPrice)

		// Only print if price or volume changed
		if currentPrice != prevPrice || int(tick.VolumeTraded) != lastVolumes[tick.InstrumentToken] {
			// Update stored values
			lastPrices[tick.InstrumentToken] = currentPrice
			lastVolumes[tick.InstrumentToken] = int(tick.VolumeTraded)

			// Decide color for LTP: green if price up, red if down, reset if same or first tick
			colorReset := "\033[0m"
			colorRed := "\033[31m"
			colorGreen := "\033[32m"
			color := colorReset

			if prevPrice != 0 {
				if currentPrice > prevPrice {
					color = colorGreen
				} else if currentPrice < prevPrice {
					color = colorRed
				}
			}

			label := tokenToLabel[tick.InstrumentToken]
			fmt.Printf(
				"📈 %s [Token: %d] - LTP: %s%.2f%s Vol: %d O: %.2f H: %.2f L: %.2f C: %.2f\n",
				label, tick.InstrumentToken,
				color, currentPrice, colorReset,
				tick.VolumeTraded,
				tick.OHLC.Open,
				tick.OHLC.High,
				tick.OHLC.Low,
				tick.OHLC.Close,
			)
		}
	})

	z.Ticker.OnError(func(err error) {
		log.Printf("❌ WebSocket error: %v", err)
	})

	z.Ticker.OnClose(func(code int, reason string) {
		log.Printf("🔌 WebSocket closed: %s (code %d)", reason, code)
	})

	go z.Ticker.Serve()
	return nil
}
