// Package market serves instrument and OHLCV candle data for charting.
package market

import (
	"net/http"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api/handlers/httpx"
	"github.com/Bhavik2205/ML-Bot/internal/db"
)

// priceScale matches candles.PriceScale. Candles are stored as scaled int64
// and converted back to float only at the API boundary.
const priceScale = 10000

type candleDTO struct {
	Timestamp time.Time `json:"timestamp"`
	Open      float64   `json:"open"`
	High      float64   `json:"high"`
	Low       float64   `json:"low"`
	Close     float64   `json:"close"`
	Volume    int64     `json:"volume"`
}

type candlesResponse struct {
	InstrumentToken uint32      `json:"instrument_token"`
	Interval        string      `json:"interval"`
	Count           int         `json:"count"`
	Candles         []candleDTO `json:"candles"`
}

type instrumentDTO struct {
	InstrumentToken uint32  `json:"instrument_token"`
	Exchange        string  `json:"exchange"`
	TradingSymbol   string  `json:"trading_symbol"`
	Name            string  `json:"name"`
	InstrumentType  string  `json:"instrument_type"`
	TickSize        float64 `json:"tick_size"`
	LotSize         int     `json:"lot_size"`
}

// HandleCandles returns OHLCV candles for one instrument and interval,
// oldest first so a chart can plot them directly.
//
// Query: ?instrument_token=<n>&interval=<1m|5m|...>&limit=<1..5000>
func HandleCandles(dbc *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if dbc == nil {
			httpx.WriteError(w, http.StatusServiceUnavailable, r, httpx.CodeUnavailable, "database unavailable", nil)
			return
		}

		token, ok := httpx.QueryUint32(r, "instrument_token")
		if !ok {
			httpx.WriteError(w, http.StatusBadRequest, r, httpx.CodeBadRequest,
				"instrument_token is required and must be a positive integer", nil)
			return
		}
		interval := r.URL.Query().Get("interval")
		if interval == "" {
			interval = "1m"
		}
		limit := httpx.QueryInt(r, "limit", 200, 1, 5000)

		// Take the newest `limit` rows, then reverse so the response is
		// chronological. Ordering ascending with a limit would return the
		// oldest candles in the table, which is never what a chart wants.
		var rows []db.OHLCVCandle
		err := dbc.DB.
			Where("instrument_token = ? AND interval = ?", token, interval).
			Order("timestamp DESC").
			Limit(limit).
			Find(&rows).Error
		if err != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal, "failed to query candles", err.Error())
			return
		}

		out := make([]candleDTO, len(rows))
		for i, c := range rows {
			out[len(rows)-1-i] = candleDTO{
				Timestamp: c.Timestamp,
				Open:      float64(c.Open) / priceScale,
				High:      float64(c.High) / priceScale,
				Low:       float64(c.Low) / priceScale,
				Close:     float64(c.Close) / priceScale,
				Volume:    c.Volume,
			}
		}

		httpx.WriteSuccess(w, http.StatusOK, r, candlesResponse{
			InstrumentToken: token,
			Interval:        interval,
			Count:           len(out),
			Candles:         out,
		})
	}
}

// HandleInstruments lists tradable instruments for symbol pickers.
//
// Query: ?limit=<1..1000>&exchange=<NSE|BSE>
func HandleInstruments(dbc *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if dbc == nil {
			httpx.WriteError(w, http.StatusServiceUnavailable, r, httpx.CodeUnavailable, "database unavailable", nil)
			return
		}
		limit := httpx.QueryInt(r, "limit", 200, 1, 1000)

		q := dbc.DB.Model(&db.Instrument{}).Order("tradingsymbol ASC").Limit(limit)
		if v := r.URL.Query().Get("exchange"); v != "" {
			q = q.Where("exchange = ?", v)
		}

		var rows []db.Instrument
		if err := q.Find(&rows).Error; err != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal, "failed to query instruments", err.Error())
			return
		}

		out := make([]instrumentDTO, 0, len(rows))
		for _, in := range rows {
			out = append(out, instrumentDTO{
				InstrumentToken: in.InstrumentToken,
				Exchange:        in.Exchange,
				TradingSymbol:   in.Tradingsymbol,
				Name:            in.Name,
				InstrumentType:  in.InstrumentType,
				TickSize:        in.TickSize,
				LotSize:         in.LotSize,
			})
		}
		httpx.WriteSuccess(w, http.StatusOK, r, out)
	}
}
