package stockHandler

import (
	"encoding/json"
	"net/http"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"go.uber.org/zap"
)

// Handler function that takes ZerodhaClient as argument
func HandleInstrumentLookup(z *api.ZerodhaClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if z == nil || z.Kite == nil {
			zap.L().Warn("instrument lookup called in simulate mode")
			http.Error(w, "Zerodha disabled in simulate mode", http.StatusServiceUnavailable)
			return
		}

		symbol := r.URL.Query().Get("symbol")
		if symbol == "" {
			zap.L().Warn("Missing symbol parameter in instrument lookup", zap.String("url", r.URL.String()))
			http.Error(w, "Missing 'symbol'", http.StatusBadRequest)
			return
		}

		info, err := z.Kite.GetQuote("NSE:" + symbol)
		if err != nil {
			zap.L().Error("Failed to get quote from Zerodha", zap.String("symbol", symbol), zap.Error(err))
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		if err := json.NewEncoder(w).Encode(info); err != nil {
			zap.L().Error("Failed to encode instrument info as JSON", zap.String("symbol", symbol), zap.Error(err))
			http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		}
	}
}
