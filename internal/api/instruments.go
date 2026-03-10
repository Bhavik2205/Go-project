// api/instruments.go
package api

import (
	"encoding/csv"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"go.uber.org/zap"
)

type InstrumentInfo struct {
	Token          uint32
	ExchangeToken  uint32     // Added based on CSV header
	Symbol         string     // This is 'tradingsymbol' from the CSV
	Name           string     // This is 'name' from the CSV
	LastPrice      float64    // Added based on CSV header, using float64 for precision
	Expiry         *time.Time // Pointer to allow nil for non-F&O instruments
	Strike         *float64   // Pointer to allow nil for non-F&O instruments
	TickSize       float64    // Changed to float64 for better precision and common usage
	LotSize        uint32     // Changed to uint32 for clarity and common usage
	InstrumentType string
	Segment        string
	Exchange       string // This is the 'exchange' from the CSV
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
		zap.L().Info("Instruments file is older than 24 hours, downloading new copy")
		return z.DownloadInstrumentsCSV(path)
	}

	// File exists and is recent enough
	return nil
}

func (z *ZerodhaClient) DownloadInstrumentsCSV(path string) error {
	url := "https://api.kite.trade/instruments"
	resp, err := http.Get(url)
	if err != nil {
		zap.L().Error("Failed to download instruments", zap.String("url", url), zap.Error(err))
		// Return a formatted error message
		return fmt.Errorf("failed to download instruments from %s: %v", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		zap.L().Error("Failed to download instruments", zap.String("url", url), zap.Error(err))
		return fmt.Errorf("failed to download instruments: received non-OK HTTP status %d", resp.StatusCode)
	}

	out, err := os.Create(path)
	if err != nil {
		zap.L().Error("Failed to create instruments file", zap.String("path", path), zap.Error(err))
		return fmt.Errorf("failed to create instruments file '%s': %v", path, err)
	}
	defer out.Close()

	_, err = out.ReadFrom(resp.Body)
	if err != nil {
		zap.L().Error("Failed to write downloaded instruments to file", zap.String("path", path), zap.Error(err))
		return fmt.Errorf("failed to write downloaded instruments to file '%s': %v", path, err)
	}
	zap.L().Info("Successfully downloaded instruments CSV", zap.String("path", path))
	return nil
}

func (z *ZerodhaClient) FindInstrumentToken(symbol string, preferredExchanges []string) (*InstrumentInfo, error) {
	const instrumentsFile = "instruments.csv"

	if err := z.EnsureInstrumentsCSV(instrumentsFile); err != nil {
		zap.L().Error("Failed to ensure instruments.csv", zap.Error(err))
		return nil, fmt.Errorf("failed to ensure instruments.csv: %v", err)
	}

	file, err := os.Open(instrumentsFile)
	if err != nil {
		zap.L().Error("Failed to open instruments file", zap.String("file", instrumentsFile), zap.Error(err))
		return nil, fmt.Errorf("failed to open instruments file '%s': %v", instrumentsFile, err)
	}
	defer file.Close()

	r := csv.NewReader(file)
	header, err := r.Read() // Read the header row
	if err != nil {
		zap.L().Error("Failed to read CSV header", zap.String("file", instrumentsFile), zap.Error(err))
		return nil, fmt.Errorf("failed to read CSV header from '%s': %v", instrumentsFile, err)
	}

	// Map column names to their indices for robust parsing
	// (using a map for O(1) lookup after initial setup)
	colMap := make(map[string]int)
	for i, colName := range header {
		colMap[strings.ToLower(strings.TrimSpace(colName))] = i
	}

	// Helper to get column index safely
	getColIndex := func(name string) (int, error) {
		idx, ok := colMap[strings.ToLower(name)]
		if !ok {
			zap.L().Error("Column not found in CSV header", zap.String("column", name), zap.String("file", instrumentsFile))
			return -1, fmt.Errorf("column '%s' not found in CSV header", name)
		}
		return idx, nil
	}

	// Get all required column indices from the header
	idxInstrumentToken, err := getColIndex("instrument_token")
	if err != nil {
		return nil, err
	}
	idxExchangeToken, err := getColIndex("exchange_token")
	if err != nil {
		return nil, err
	}
	idxTradingSymbol, err := getColIndex("tradingsymbol")
	if err != nil {
		return nil, err
	}
	idxName, err := getColIndex("name")
	if err != nil {
		return nil, err
	}
	idxLastPrice, err := getColIndex("last_price")
	if err != nil {
		return nil, err
	}
	idxExpiry, err := getColIndex("expiry")
	if err != nil {
		return nil, err
	}
	idxStrike, err := getColIndex("strike")
	if err != nil {
		return nil, err
	}
	idxTickSize, err := getColIndex("tick_size")
	if err != nil {
		return nil, err
	}
	idxLotSize, err := getColIndex("lot_size")
	if err != nil {
		return nil, err
	}
	idxInstrumentType, err := getColIndex("instrument_type")
	if err != nil {
		return nil, err
	}
	idxSegment, err := getColIndex("segment")
	if err != nil {
		return nil, err
	}
	idxExchange, err := getColIndex("exchange")
	if err != nil {
		return nil, err
	}

	foundInstruments := make(map[string]*InstrumentInfo) // Map by exchange for easy lookup

	for {
		record, err := r.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break // End of file
			}
			zap.L().Error("Failed to read CSV record", zap.String("file", instrumentsFile), zap.Error(err))
			return nil, fmt.Errorf("failed to read CSV record: %v", err)
		}

		// Basic check: ensure the record has enough columns to avoid out-of-bounds access
		// Using the largest index you'll access (e.g., idxExchange)
		if len(record) <= idxExchange {
			zap.L().Warn("Skipping malformed row with insufficient columns", zap.Strings("record", record))
			continue
		}

		tradingSymbol := strings.TrimSpace(record[idxTradingSymbol])
		exchange := record[idxExchange]

		if strings.EqualFold(tradingSymbol, symbol) {
			// Parse Token
			token64, parseErr := strconv.ParseUint(record[idxInstrumentToken], 10, 32)
			if parseErr != nil {
				zap.L().Warn("Invalid instrument_token format", zap.String("token", record[idxInstrumentToken]), zap.String("symbol", tradingSymbol), zap.Error(parseErr))
				continue
			}
			exchangeToken64, parseErr := strconv.ParseUint(record[idxExchangeToken], 10, 32)
			if parseErr != nil {
				zap.L().Warn("Invalid exchange_token format", zap.String("token", record[idxExchangeToken]), zap.String("symbol", tradingSymbol), zap.Error(parseErr))
				exchangeToken64 = 0
			}

			// Parse numerical fields
			lastPrice, parseErr := strconv.ParseFloat(record[idxLastPrice], 64)
			if parseErr != nil {
				zap.L().Warn("Invalid last_price format", zap.String("last_price", record[idxLastPrice]), zap.String("symbol", tradingSymbol), zap.Error(parseErr))
				lastPrice = 0.0
			}

			tickSize, parseErr := strconv.ParseFloat(record[idxTickSize], 64)
			if parseErr != nil {
				zap.L().Warn("Invalid tick_size format", zap.String("tick_size", record[idxTickSize]), zap.String("symbol", tradingSymbol), zap.Error(parseErr))
				tickSize = 0.0
			}

			lotSize, parseErr := strconv.ParseUint(record[idxLotSize], 10, 32)
			if parseErr != nil {
				zap.L().Warn("Invalid lot_size format", zap.String("lot_size", record[idxLotSize]), zap.String("symbol", tradingSymbol), zap.Error(parseErr))
				lotSize = 0
			}

			// Parse optional fields (Expiry, Strike)
			var expiry *time.Time
			if record[idxExpiry] != "" {
				t, dateParseErr := time.Parse("2006-01-02", record[idxExpiry]) // Assuming YYYY-MM-DD format
				if dateParseErr == nil {
					expiry = &t
				} else {
					zap.L().Warn("Invalid expiry format", zap.String("expiry", record[idxExpiry]), zap.String("symbol", tradingSymbol), zap.Error(dateParseErr))
				}
			}

			var strike *float64
			if record[idxStrike] != "" {
				s, strikeParseErr := strconv.ParseFloat(record[idxStrike], 64)
				if strikeParseErr == nil {
					strike = &s
				} else {
					zap.L().Warn("Invalid strike price format", zap.String("strike", record[idxStrike]), zap.String("symbol", tradingSymbol), zap.Error(strikeParseErr))
				}
			}

			foundInstruments[exchange] = &InstrumentInfo{
				Token:          uint32(token64),
				ExchangeToken:  uint32(exchangeToken64),
				Symbol:         tradingSymbol,
				Name:           record[idxName],
				LastPrice:      lastPrice,
				Expiry:         expiry,
				Strike:         strike,
				TickSize:       tickSize,
				LotSize:        uint32(lotSize),
				InstrumentType: record[idxInstrumentType],
				Segment:        record[idxSegment],
				Exchange:       exchange,
			}
		}
	}

	// Prioritize instruments based on preferred exchanges
	for _, exch := range preferredExchanges {
		if info, ok := foundInstruments[exch]; ok {
			return info, nil
		}
	}

	zap.L().Warn("Symbol not found in preferred exchanges", zap.String("symbol", symbol), zap.Strings("preferredExchanges", preferredExchanges))
	return nil, fmt.Errorf("symbol '%s' not found in preferred exchanges: %v", symbol, preferredExchanges)
}
