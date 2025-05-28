// api/zerodha_client.go
package api

import (
	"encoding/csv"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

type InstrumentInfo struct {
	Token    uint32
	Symbol   string
	Exchange string
}

func (z *ZerodhaClient) EnsureInstrumentsCSV(path string) error {
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		// File doesn't exist, download it
		return z.DownloadInstrumentsCSV(path)
	}
	if err != nil {
		return err
	}

	// Check file age
	if time.Since(info.ModTime()) > 24*time.Hour {
		fmt.Println("Instruments file is older than 24 hours, downloading new copy...")
		return z.DownloadInstrumentsCSV(path)
	}

	// File exists and is recent enough
	return nil
}

func (z *ZerodhaClient) DownloadInstrumentsCSV(path string) error {
	url := "https://api.kite.trade/instruments"
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to download instruments: %v", err)
	}
	defer resp.Body.Close()

	out, err := os.Create(path)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = out.ReadFrom(resp.Body)
	return err
}

func (z *ZerodhaClient) FindInstrumentToken(symbol string, preferredExchanges []string) (*InstrumentInfo, error) {
	const instrumentsFile = "instruments.csv"

	if err := z.EnsureInstrumentsCSV(instrumentsFile); err != nil {
		return nil, fmt.Errorf("failed to ensure instruments.csv: %v", err)
	}

	file, err := os.Open(instrumentsFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	r := csv.NewReader(file)
	_, _ = r.Read() // skip header

	foundTokens := make(map[string]*InstrumentInfo)

	for {
		record, err := r.Read()
		if err != nil {
			break
		}
		tradingSymbol := strings.TrimSpace(record[2])
		exchange := record[11]

		if strings.EqualFold(tradingSymbol, symbol) {
			token64, err := strconv.ParseUint(record[0], 10, 32)
			if err != nil {
				return nil, fmt.Errorf("invalid token format: %v", err)
			}
			foundTokens[exchange] = &InstrumentInfo{
				Token:    uint32(token64),
				Symbol:   tradingSymbol,
				Exchange: exchange,
			}
		}
	}

	for _, exch := range preferredExchanges {
		if info, ok := foundTokens[exch]; ok {
			return info, nil
		}
	}

	return nil, fmt.Errorf("symbol %s not found in preferred exchanges %v", symbol, preferredExchanges)
}
