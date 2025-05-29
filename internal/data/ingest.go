package data

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	kiteconnect "github.com/zerodha/gokiteconnect/v4"
)

type HistoricalFetcher struct {
	Kite *kiteconnect.Client
}

func NewHistoricalFetcher(kc *kiteconnect.Client) *HistoricalFetcher {
	return &HistoricalFetcher{Kite: kc}
}

func (hf *HistoricalFetcher) FetchAndSave(symbol string, token uint32, from, to time.Time, interval string, filePath string) error {
	log.Printf("📥 Fetching historical data for %s", symbol)

	records, err := hf.Kite.GetHistoricalData(
		int(token),
		interval,
		from,
		to,
		false, // continuous
		false, // oi (open interest)
	)
	if err != nil {
		return fmt.Errorf("❌ error fetching historical data: %w", err)
	}

	// Create or overwrite file
	f, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("❌ error creating file: %w", err)
	}
	defer f.Close()

	writer := csv.NewWriter(f)
	defer writer.Flush()

	// Header
	writer.Write([]string{"Date", "Open", "High", "Low", "Close", "Volume"})

	for _, r := range records {
		row := []string{
			r.Date.Format("2006-01-02 15:04:05"),
			fmt.Sprintf("%.2f", r.Open),
			fmt.Sprintf("%.2f", r.High),
			fmt.Sprintf("%.2f", r.Low),
			fmt.Sprintf("%.2f", r.Close),
			strconv.Itoa(r.Volume),
		}
		writer.Write(row)
	}

	log.Printf("✅ Saved to %s", filePath)
	return nil
}
