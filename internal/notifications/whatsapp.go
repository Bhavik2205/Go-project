package notifications

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// metaGraphAPI is the WhatsApp Cloud API base. A variable so tests can
// redirect it.
var metaGraphAPI = "https://graph.facebook.com/v21.0"

// WhatsApp delivery modes.
const (
	// WhatsAppMeta uses the official Cloud API.
	WhatsAppMeta = "meta"
	// WhatsAppWebhook posts to any HTTP endpoint, which covers self-hosted
	// gateways and the various unofficial bridges people run.
	WhatsAppWebhook = "webhook"
)

// WhatsAppConfig is what the user enters in the dashboard.
//
// Two shapes are supported because there is no single WhatsApp API: the
// official Cloud API needs a token and a phone number id, while self-hosted
// gateways are just an HTTP endpoint.
type WhatsAppConfig struct {
	// Provider is "meta" (default when accessToken is present) or "webhook".
	Provider string `json:"provider,omitempty"`

	// --- Meta Cloud API ---
	AccessToken   string `json:"accessToken,omitempty"`
	PhoneNumberID string `json:"phoneNumberId,omitempty"`

	// --- Generic webhook ---
	APIURL string `json:"apiUrl,omitempty"`
	// AuthHeader is sent verbatim as the Authorization header when set.
	AuthHeader string `json:"authHeader,omitempty"`

	// Recipient is the destination in international format, e.g. +919876543210.
	// PhoneNumber is accepted as an alias for backwards compatibility with the
	// original config shape.
	Recipient   string `json:"recipient,omitempty"`
	PhoneNumber string `json:"phoneNumber,omitempty"`
}

// to returns the destination number, preferring the newer field name.
func (c *WhatsAppConfig) to() string {
	if c.Recipient != "" {
		return c.Recipient
	}
	return c.PhoneNumber
}

// mode resolves which delivery path to use.
func (c *WhatsAppConfig) mode() string {
	if c.Provider != "" {
		return c.Provider
	}
	if c.AccessToken != "" && c.PhoneNumberID != "" {
		return WhatsAppMeta
	}
	if c.APIURL != "" {
		return WhatsAppWebhook
	}
	return ""
}

// WhatsAppProvider sends via the Meta Cloud API or a generic webhook.
type WhatsAppProvider struct{}

// Name identifies the channel.
func (p *WhatsAppProvider) Name() string { return ChannelWhatsApp }

// Validate checks the config shape without contacting anything.
func (p *WhatsAppProvider) Validate(raw json.RawMessage) error {
	_, err := parseWhatsApp(raw)
	return err
}

func parseWhatsApp(raw json.RawMessage) (*WhatsAppConfig, error) {
	var cfg WhatsAppConfig
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &cfg); err != nil {
			return nil, fmt.Errorf("whatsapp config is not valid JSON: %w", err)
		}
	}

	cfg.AccessToken = strings.TrimSpace(cfg.AccessToken)
	cfg.PhoneNumberID = strings.TrimSpace(cfg.PhoneNumberID)
	cfg.APIURL = strings.TrimSpace(cfg.APIURL)
	cfg.Recipient = strings.TrimSpace(cfg.Recipient)
	cfg.PhoneNumber = strings.TrimSpace(cfg.PhoneNumber)

	switch cfg.mode() {
	case WhatsAppMeta:
		if cfg.AccessToken == "" {
			return nil, fmt.Errorf("accessToken is required for the Meta Cloud API")
		}
		if cfg.PhoneNumberID == "" {
			return nil, fmt.Errorf("phoneNumberId is required for the Meta Cloud API")
		}
	case WhatsAppWebhook:
		if cfg.APIURL == "" {
			return nil, fmt.Errorf("apiUrl is required for webhook delivery")
		}
		if !strings.HasPrefix(cfg.APIURL, "https://") && !strings.HasPrefix(cfg.APIURL, "http://") {
			return nil, fmt.Errorf("apiUrl must start with http:// or https://")
		}
	default:
		return nil, fmt.Errorf(
			"supply either accessToken and phoneNumberId (Meta Cloud API) or apiUrl (webhook)")
	}

	if cfg.to() == "" {
		return nil, fmt.Errorf("recipient phone number is required")
	}
	// WhatsApp addresses are international numbers; catching this here avoids
	// a confusing provider-side rejection later.
	if !strings.HasPrefix(cfg.to(), "+") {
		return nil, fmt.Errorf("recipient must be in international format, e.g. +919876543210")
	}
	return &cfg, nil
}

// Send delivers one message over the configured path.
func (p *WhatsAppProvider) Send(ctx context.Context, raw json.RawMessage, message string) (string, error) {
	cfg, err := parseWhatsApp(raw)
	if err != nil {
		return "", &ErrPermanent{Err: err}
	}

	// The Cloud API caps a text body at 4096 characters.
	const maxLen = 4096
	if len(message) > maxLen {
		message = message[:maxLen-3] + "..."
	}

	if cfg.mode() == WhatsAppMeta {
		return sendMeta(ctx, cfg, message)
	}
	return sendWebhook(ctx, cfg, message)
}

type metaResponse struct {
	Messages []struct {
		ID string `json:"id"`
	} `json:"messages"`
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    int    `json:"code"`
	} `json:"error"`
}

func sendMeta(ctx context.Context, cfg *WhatsAppConfig, message string) (string, error) {
	// The Cloud API wants the number without the leading plus.
	to := strings.TrimPrefix(cfg.to(), "+")

	payload, err := json.Marshal(map[string]any{
		"messaging_product": "whatsapp",
		"to":                to,
		"type":              "text",
		"text":              map[string]any{"body": message},
	})
	if err != nil {
		return "", &ErrPermanent{Err: fmt.Errorf("encode whatsapp request: %w", err)}
	}

	url := fmt.Sprintf("%s/%s/messages", metaGraphAPI, cfg.PhoneNumberID)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return "", fmt.Errorf("build whatsapp request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cfg.AccessToken)

	resp, err := httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("whatsapp request failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))

	var mr metaResponse
	_ = json.Unmarshal(body, &mr)

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		if len(mr.Messages) > 0 {
			return mr.Messages[0].ID, nil
		}
		return "", nil // accepted, but no id returned
	}

	desc := mr.Error.Message
	if desc == "" {
		desc = strings.TrimSpace(string(body))
	}

	if retryableStatus(resp.StatusCode) {
		return "", fmt.Errorf("whatsapp %d: %s", resp.StatusCode, desc)
	}
	return "", permanent("whatsapp %d: %s", resp.StatusCode, desc)
}

type webhookResponse struct {
	ID        string `json:"id"`
	MessageID string `json:"message_id"`
	Error     string `json:"error"`
}

func sendWebhook(ctx context.Context, cfg *WhatsAppConfig, message string) (string, error) {
	payload, err := json.Marshal(map[string]any{
		"to":      cfg.to(),
		"message": message,
	})
	if err != nil {
		return "", &ErrPermanent{Err: fmt.Errorf("encode whatsapp request: %w", err)}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.APIURL, bytes.NewReader(payload))
	if err != nil {
		return "", fmt.Errorf("build whatsapp request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.AuthHeader != "" {
		req.Header.Set("Authorization", cfg.AuthHeader)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("whatsapp webhook failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		var wr webhookResponse
		_ = json.Unmarshal(body, &wr)
		if wr.MessageID != "" {
			return wr.MessageID, nil
		}
		return wr.ID, nil
	}

	desc := strings.TrimSpace(string(body))
	if retryableStatus(resp.StatusCode) {
		return "", fmt.Errorf("whatsapp webhook %d: %s", resp.StatusCode, desc)
	}
	return "", permanent("whatsapp webhook %d: %s", resp.StatusCode, desc)
}
