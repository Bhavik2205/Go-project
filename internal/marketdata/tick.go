package marketdata

import (
	"time"

	kitemodels "github.com/zerodha/gokiteconnect/v4/models"
)

// NormalizedTick is the canonical tick event model used throughout the system.
// It separates exchange event time from system ingest time.
type NormalizedTick struct {
	// Instrument identification
	InstrumentToken uint32 `json:"instrument_token"`
	Symbol          string `json:"symbol"`   // e.g., "NSE:RELIANCE"
	Exchange        string `json:"exchange"` // e.g., "NSE"

	// Timestamps
	EventTime  time.Time `json:"event_time"`  // Exchange timestamp (from tick)
	IngestTime time.Time `json:"ingest_time"` // When this system received the tick

	// Core market data
	LastPrice          float64 `json:"last_price"`
	LastTradedQuantity uint32  `json:"last_traded_quantity"`
	Volume             uint32  `json:"volume"`
	AverageTradePrice  float64 `json:"average_trade_price"`
	NetChange          float64 `json:"net_change"`
	PercentChange      float64 `json:"percent_change"`
	PrevClose          float64 `json:"prev_close"`

	// OHLC (daily values, may be repeated in each tick)
	OHLC kitemodels.OHLC `json:"ohlc"`

	// Market depth (top 5 bids and asks)
	Depth kitemodels.Depth `json:"depth"`

	// Aggregated quantities
	TotalBuyQuantity  uint32 `json:"total_buy_quantity"`
	TotalSellQuantity uint32 `json:"total_sell_quantity"`

	// Open interest (for derivatives)
	OpenInterest uint32 `json:"open_interest"`

	Mode string `json:"mode"` // "simulation", "live", or "kite"
}
