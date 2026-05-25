package validation

import (
	"encoding/json"
	"net/http"

	"github.com/go-playground/validator/v10"
)

var validate = validator.New()

// BindAndValidate decodes the JSON request body into dst and validates it.
// Returns false and writes a 400 response if decoding or validation fails.
func BindAndValidate(w http.ResponseWriter, r *http.Request, dst any) bool {
	if err := json.NewDecoder(r.Body).Decode(dst); err != nil {
		http.Error(w, `{"error":{"code":"BAD_REQUEST","message":"Invalid JSON body"}}`, http.StatusBadRequest)
		return false
	}
	if err := validate.Struct(dst); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"code":    "VALIDATION_ERROR",
				"message": "One or more fields are invalid.",
				"details": err.Error(),
			},
		})
		return false
	}
	return true
}
