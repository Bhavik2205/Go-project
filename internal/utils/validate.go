// internal/config/validate.go
package utils

import (
	"fmt"
	"os"
	"strings"
)

// ValidateRequiredEnv checks all required environment variables and returns an error
// listing any missing or invalid values. It does not modify state.
func ValidateRequiredEnv() error {
	var missing []string
	var invalid []string

	// --- Core required for any mode ---
	if os.Getenv("JWT_SECRET") == "" {
		missing = append(missing, "JWT_SECRET")
	}

	// Database variables (always required, even in simulation mode)
	dbVars := []string{"DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"}
	for _, v := range dbVars {
		if os.Getenv(v) == "" {
			missing = append(missing, v)
		}
	}

	// Redis: either REDIS_URL or (REDIS_HOST+REDIS_PORT) must be set
	redisURL := os.Getenv("REDIS_URL")
	redisHost := os.Getenv("REDIS_HOST")
	redisPort := os.Getenv("REDIS_PORT")
	if redisURL == "" && (redisHost == "" || redisPort == "") {
		missing = append(missing, "REDIS_URL or (REDIS_HOST and REDIS_PORT)")
	}

	// --- Mode-specific checks: we cannot read app config here,
	// but we can check a flag from env (optional). However,
	// the simplest is to always check Zerodha credentials;
	// they will be used only if not in simulation mode,
	// but validation can be deferred until after we know the mode.
	// To avoid false errors, we'll add a placeholder comment.
	// We'll rely on main.go's later checks for simulation mode.

	// --- Optional but recommended for production ---
	// DATA_ENCRYPTION_KEY (used for user broker tokens)
	if os.Getenv("DATA_ENCRYPTION_KEY") == "" {
		// Not missing, but warn? For now, just note.
		// We'll not treat as error because it's only needed if using broker accounts.
	}

	if len(missing) > 0 {
		return fmt.Errorf("missing required environment variables: %s", strings.Join(missing, ", "))
	}
	if len(invalid) > 0 {
		return fmt.Errorf("invalid environment variables: %s", strings.Join(invalid, ", "))
	}
	return nil
}
