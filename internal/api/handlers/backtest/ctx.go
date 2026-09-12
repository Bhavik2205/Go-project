package backtest

import (
	"context"
	"net/http"
	"time"
)

// contextWithTimeout derives a bounded context from the request, so a client
// disconnect cancels the replay instead of letting it run to completion.
func contextWithTimeout(r *http.Request, d time.Duration) (context.Context, context.CancelFunc) {
	return context.WithTimeout(r.Context(), d)
}
