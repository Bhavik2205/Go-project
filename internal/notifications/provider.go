// Package notifications contains notification channel providers and alert
// delivery services.
package notifications

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"
)

// Channel type identifiers, matching the channel_type column.
const (
	ChannelTelegram = "telegram"
	ChannelWhatsApp = "whatsapp"
)

// Event types written to notification_history.event_type.
const (
	EventTradeExecution = "TRADE_EXECUTION"
	EventPnLThreshold   = "PNL_THRESHOLD"
	EventErrorAlert     = "ERROR_ALERT"
	EventTest           = "TEST"
)

// Delivery status values.
const (
	StatusPending = "PENDING"
	StatusSent    = "SENT"
	StatusFailed  = "FAILED"
)

// ErrPermanent wraps a failure that retrying cannot fix: a bad token, an
// unknown chat, a malformed recipient. The worker marks these FAILED at once
// instead of burning the retry budget on something that will never succeed.
type ErrPermanent struct{ Err error }

func (e *ErrPermanent) Error() string { return e.Err.Error() }
func (e *ErrPermanent) Unwrap() error { return e.Err }

func permanent(format string, args ...any) error {
	return &ErrPermanent{Err: fmt.Errorf(format, args...)}
}

// IsPermanent reports whether an error should not be retried.
func IsPermanent(err error) bool {
	var p *ErrPermanent
	return errors.As(err, &p)
}

// Provider delivers one message over one channel.
//
// Send must respect ctx and return a provider message id on success. A
// transient failure returns a plain error; a permanent one returns ErrPermanent.
type Provider interface {
	// Name is the channel_type this provider handles.
	Name() string
	// Send delivers the message and returns the provider's message id.
	Send(ctx context.Context, cfg json.RawMessage, message string) (string, error)
	// Validate checks a config without sending anything, so the API can
	// reject bad credentials at save time rather than at first alert.
	Validate(cfg json.RawMessage) error
}

// httpClient is shared by the providers. The timeout is deliberately short:
// an alert that takes longer than this to send has already lost most of its
// value, and the worker will retry.
var httpClient = &http.Client{Timeout: 10 * time.Second}

// Registry maps channel type to provider.
type Registry struct {
	providers map[string]Provider
}

// NewRegistry builds a registry with the built-in providers.
func NewRegistry() *Registry {
	r := &Registry{providers: make(map[string]Provider, 2)}
	r.Register(&TelegramProvider{})
	r.Register(&WhatsAppProvider{})
	return r
}

// Register adds a provider, replacing any existing one for that channel.
func (r *Registry) Register(p Provider) { r.providers[p.Name()] = p }

// Get returns the provider for a channel type.
func (r *Registry) Get(channelType string) (Provider, bool) {
	p, ok := r.providers[channelType]
	return p, ok
}

// Names lists the supported channel types.
func (r *Registry) Names() []string {
	out := make([]string, 0, len(r.providers))
	for n := range r.providers {
		out = append(out, n)
	}
	return out
}

// retryableStatus reports whether an HTTP status is worth retrying.
//
// 429 and 5xx are transient. Everything else in the 4xx range means the
// request itself is wrong, and sending it again will fail identically.
func retryableStatus(code int) bool {
	return code == http.StatusTooManyRequests || code >= 500
}
