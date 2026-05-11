package auth

import (
	"encoding/json"
	"net/http"

	"github.com/Bhavik2205/ML-Bot/internal/contracts"
)

func writeSuccess(w http.ResponseWriter, status int, r *http.Request, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(contracts.NewSuccess(r.Header.Get("X-Request-ID"), data))
}

func writeError(w http.ResponseWriter, status int, r *http.Request, code, message string, details any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(contracts.NewError(r.Header.Get("X-Request-ID"), code, message, details))
}
