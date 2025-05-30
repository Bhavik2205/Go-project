package stockHandler

import (
	"encoding/json"
	"net/http"

	"github.com/Bhavik2205/ML-Bot/internal/api"
)

// Handler function that takes ZerodhaClient as argument
func HandleInstrumentLookup(z *api.ZerodhaClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		symbol := r.URL.Query().Get("symbol")
		if symbol == "" {
			http.Error(w, "Missing 'symbol'", http.StatusBadRequest)
			return
		}

		info, err := z.Kite.GetQuote("NSE:" + symbol)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		json.NewEncoder(w).Encode(info)
	}
}
