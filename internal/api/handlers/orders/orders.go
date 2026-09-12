// Package orders serves the persisted order and trade history.
package orders

import (
	"net/http"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api/handlers/httpx"
	"github.com/Bhavik2205/ML-Bot/internal/db"
)

type orderDTO struct {
	ID              uint      `json:"id"`
	BrokerOrderID   string    `json:"broker_order_id"`
	InstrumentToken uint32    `json:"instrument_token"`
	StrategyName    string    `json:"strategy_name"`
	TransactionType string    `json:"transaction_type"`
	OrderType       string    `json:"order_type"`
	Quantity        int       `json:"quantity"`
	Price           float64   `json:"price"`
	FilledQuantity  int       `json:"filled_quantity"`
	FilledPrice     float64   `json:"filled_price"`
	Status          string    `json:"status"`
	Product         string    `json:"product"`
	TradeType       string    `json:"trade_type"`
	PlacedAt        time.Time `json:"placed_at"`
}

type tradeDTO struct {
	ID              uint      `json:"id"`
	OrderID         uint      `json:"order_id"`
	TradeID         string    `json:"trade_id"`
	InstrumentToken uint32    `json:"instrument_token"`
	TransactionType string    `json:"transaction_type"`
	Quantity        int       `json:"quantity"`
	Price           float64   `json:"price"`
	TradeType       string    `json:"trade_type"`
	TradeTime       time.Time `json:"trade_time"`
}

// HandleListOrders returns recent orders, newest first.
//
// Filters: ?trade_type=PAPER|LIVE, ?strategy=<name>, ?limit=<1..500>.
func HandleListOrders(dbc *db.DBClient, t *httpx.Trading) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if dbc == nil {
			httpx.WriteError(w, http.StatusServiceUnavailable, r, httpx.CodeUnavailable, "database unavailable", nil)
			return
		}
		limit := httpx.QueryInt(r, "limit", 50, 1, 500)

		q := dbc.DB.Model(&db.Order{}).Order("placed_at DESC").Limit(limit)
		if v := r.URL.Query().Get("trade_type"); v != "" {
			q = q.Where("trade_type = ?", v)
		}
		if v := r.URL.Query().Get("strategy"); v != "" {
			q = q.Where("strategy_name = ?", v)
		}
		if tok, ok := httpx.QueryUint32(r, "instrument_token"); ok {
			q = q.Where("instrument_token = ?", tok)
		}

		var rows []db.Order
		if err := q.Find(&rows).Error; err != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal, "failed to query orders", err.Error())
			return
		}

		out := make([]orderDTO, 0, len(rows))
		for _, o := range rows {
			out = append(out, orderDTO{
				ID: o.ID, BrokerOrderID: o.BrokerOrderID, InstrumentToken: o.InstrumentToken,
				StrategyName: o.StrategyName, TransactionType: o.TransactionType, OrderType: o.OrderType,
				Quantity: o.Quantity, Price: o.Price, FilledQuantity: o.FilledQuantity,
				FilledPrice: o.FilledPrice, Status: o.Status, Product: o.Product,
				TradeType: o.TradeType, PlacedAt: o.PlacedAt,
			})
		}
		httpx.WriteSuccess(w, http.StatusOK, r, out)
	}
}

// HandleListTrades returns recent executions, newest first.
func HandleListTrades(dbc *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if dbc == nil {
			httpx.WriteError(w, http.StatusServiceUnavailable, r, httpx.CodeUnavailable, "database unavailable", nil)
			return
		}
		limit := httpx.QueryInt(r, "limit", 50, 1, 500)

		q := dbc.DB.Model(&db.Trade{}).Order("trade_time DESC").Limit(limit)
		if v := r.URL.Query().Get("trade_type"); v != "" {
			q = q.Where("trade_type = ?", v)
		}
		if tok, ok := httpx.QueryUint32(r, "instrument_token"); ok {
			q = q.Where("instrument_token = ?", tok)
		}

		var rows []db.Trade
		if err := q.Find(&rows).Error; err != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal, "failed to query trades", err.Error())
			return
		}

		out := make([]tradeDTO, 0, len(rows))
		for _, tr := range rows {
			out = append(out, tradeDTO{
				ID: tr.ID, OrderID: tr.OrderID, TradeID: tr.TradeID,
				InstrumentToken: tr.InstrumentToken, TransactionType: tr.TransactionType,
				Quantity: tr.Quantity, Price: tr.Price, TradeType: tr.TradeType, TradeTime: tr.TradeTime,
			})
		}
		httpx.WriteSuccess(w, http.StatusOK, r, out)
	}
}
