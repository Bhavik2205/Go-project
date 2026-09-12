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

// telegramAPI is the Bot API base. A variable rather than a constant so tests
// can point it at a local server.
var telegramAPI = "https://api.telegram.org"

// TelegramConfig is what the user enters in the dashboard.
type TelegramConfig struct {
	// BotToken comes from @BotFather, in the form "123456:ABC-DEF...".
	BotToken string `json:"botToken"`
	// ChatID is the destination. Groups and channels are negative numbers.
	ChatID string `json:"chatId"`
	// ParseMode is optional: "" (plain), "MarkdownV2" or "HTML".
	ParseMode string `json:"parseMode,omitempty"`
}

// TelegramProvider sends via the Telegram Bot API.
type TelegramProvider struct{}

// Name identifies the channel.
func (p *TelegramProvider) Name() string { return ChannelTelegram }

// Validate checks the config shape without contacting Telegram.
func (p *TelegramProvider) Validate(raw json.RawMessage) error {
	cfg, err := parseTelegram(raw)
	if err != nil {
		return err
	}
	if cfg.ParseMode != "" &&
		cfg.ParseMode != "MarkdownV2" &&
		cfg.ParseMode != "HTML" &&
		cfg.ParseMode != "Markdown" {
		return fmt.Errorf("parseMode must be empty, Markdown, MarkdownV2 or HTML")
	}
	return nil
}

func parseTelegram(raw json.RawMessage) (*TelegramConfig, error) {
	var cfg TelegramConfig
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &cfg); err != nil {
			return nil, fmt.Errorf("telegram config is not valid JSON: %w", err)
		}
	}
	cfg.BotToken = strings.TrimSpace(cfg.BotToken)
	cfg.ChatID = strings.TrimSpace(cfg.ChatID)

	if cfg.BotToken == "" {
		return nil, fmt.Errorf("botToken is required")
	}
	// A token is "<numeric id>:<secret>". Catching the shape here turns a
	// silent 404 from Telegram into a clear message at save time.
	if !strings.Contains(cfg.BotToken, ":") {
		return nil, fmt.Errorf("botToken looks malformed; expected the form 123456:ABC-DEF from @BotFather")
	}
	if cfg.ChatID == "" {
		return nil, fmt.Errorf("chatId is required")
	}
	return &cfg, nil
}

type telegramResponse struct {
	OK          bool   `json:"ok"`
	Description string `json:"description"`
	ErrorCode   int    `json:"error_code"`
	Result      struct {
		MessageID int64 `json:"message_id"`
	} `json:"result"`
}

// Send posts one message and returns the Telegram message id.
func (p *TelegramProvider) Send(ctx context.Context, raw json.RawMessage, message string) (string, error) {
	cfg, err := parseTelegram(raw)
	if err != nil {
		// A config that cannot be parsed will not parse next time either.
		return "", &ErrPermanent{Err: err}
	}

	// Telegram rejects messages over 4096 characters outright, so truncate
	// rather than have the whole alert fail.
	const maxLen = 4096
	if len(message) > maxLen {
		message = message[:maxLen-3] + "..."
	}

	body := map[string]any{
		"chat_id": cfg.ChatID,
		"text":    message,
	}
	if cfg.ParseMode != "" {
		body["parse_mode"] = cfg.ParseMode
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return "", &ErrPermanent{Err: fmt.Errorf("encode telegram request: %w", err)}
	}

	url := fmt.Sprintf("%s/bot%s/sendMessage", telegramAPI, cfg.BotToken)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return "", fmt.Errorf("build telegram request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		// Network problems are transient by definition.
		return "", fmt.Errorf("telegram request failed: %w", err)
	}
	defer resp.Body.Close()

	raw2, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))

	var tr telegramResponse
	_ = json.Unmarshal(raw2, &tr)

	if resp.StatusCode == http.StatusOK && tr.OK {
		return fmt.Sprintf("%d", tr.Result.MessageID), nil
	}

	desc := tr.Description
	if desc == "" {
		desc = strings.TrimSpace(string(raw2))
	}

	if retryableStatus(resp.StatusCode) {
		return "", fmt.Errorf("telegram %d: %s", resp.StatusCode, desc)
	}
	// 401 bad token, 400 bad chat_id, 403 bot blocked: all unfixable by retry.
	return "", permanent("telegram %d: %s", resp.StatusCode, desc)
}
