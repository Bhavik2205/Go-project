package notifications

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Bhavik2205/ML-Bot/internal/security"
)

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

func TestTelegramValidate(t *testing.T) {
	p := &TelegramProvider{}

	cases := []struct {
		name    string
		cfg     string
		wantErr bool
	}{
		{"valid", `{"botToken":"123456:ABC-DEF","chatId":"-1001234"}`, false},
		{"valid with parse mode", `{"botToken":"1:a","chatId":"5","parseMode":"HTML"}`, false},
		{"missing token", `{"chatId":"-1001234"}`, true},
		{"missing chat", `{"botToken":"123456:ABC"}`, true},
		{"token without colon", `{"botToken":"notatoken","chatId":"5"}`, true},
		{"bad parse mode", `{"botToken":"1:a","chatId":"5","parseMode":"rtf"}`, true},
		{"empty", `{}`, true},
	}

	for _, c := range cases {
		err := p.Validate(json.RawMessage(c.cfg))
		if (err != nil) != c.wantErr {
			t.Errorf("%s: err = %v, wantErr = %v", c.name, err, c.wantErr)
		}
	}
}

func TestWhatsAppValidate(t *testing.T) {
	p := &WhatsAppProvider{}

	cases := []struct {
		name    string
		cfg     string
		wantErr bool
	}{
		{"meta", `{"accessToken":"tok","phoneNumberId":"123","recipient":"+919876543210"}`, false},
		{"webhook", `{"apiUrl":"https://gw.example.com/send","recipient":"+919876543210"}`, false},
		{"legacy phoneNumber alias", `{"apiUrl":"https://x.com/s","phoneNumber":"+911234567890"}`, false},
		{"meta without token", `{"phoneNumberId":"123","recipient":"+91987"}`, true},
		{"no delivery path", `{"recipient":"+919876543210"}`, true},
		{"missing recipient", `{"apiUrl":"https://x.com/s"}`, true},
		{"recipient not international", `{"apiUrl":"https://x.com/s","recipient":"9876543210"}`, true},
		{"apiUrl not a url", `{"apiUrl":"gw.example.com","recipient":"+91987"}`, true},
	}

	for _, c := range cases {
		err := p.Validate(json.RawMessage(c.cfg))
		if (err != nil) != c.wantErr {
			t.Errorf("%s: err = %v, wantErr = %v", c.name, err, c.wantErr)
		}
	}
}

// ---------------------------------------------------------------------------
// Telegram delivery
// ---------------------------------------------------------------------------

func TestTelegramSendSuccess(t *testing.T) {
	var gotPath, gotBody string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true,"result":{"message_id":4242}}`))
	}))
	defer srv.Close()

	old := telegramAPI
	telegramAPI = srv.URL
	defer func() { telegramAPI = old }()

	p := &TelegramProvider{}
	id, err := p.Send(context.Background(),
		json.RawMessage(`{"botToken":"123:abc","chatId":"-100"}`), "hello")
	if err != nil {
		t.Fatalf("send: %v", err)
	}
	if id != "4242" {
		t.Errorf("message id = %q, want 4242", id)
	}
	// The token belongs in the path, never the body.
	if gotPath != "/bot123:abc/sendMessage" {
		t.Errorf("path = %q", gotPath)
	}
	if !strings.Contains(gotBody, `"chat_id":"-100"`) || !strings.Contains(gotBody, `"text":"hello"`) {
		t.Errorf("body = %s", gotBody)
	}
}

func TestTelegramBadTokenIsPermanent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"ok":false,"error_code":401,"description":"Unauthorized"}`))
	}))
	defer srv.Close()

	old := telegramAPI
	telegramAPI = srv.URL
	defer func() { telegramAPI = old }()

	p := &TelegramProvider{}
	_, err := p.Send(context.Background(),
		json.RawMessage(`{"botToken":"123:bad","chatId":"-100"}`), "hi")
	if err == nil {
		t.Fatal("expected an error")
	}
	// Retrying a bad token forever would burn the budget for nothing.
	if !IsPermanent(err) {
		t.Errorf("401 should be permanent, got retryable: %v", err)
	}
}

func TestTelegramRateLimitIsRetryable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"ok":false,"error_code":429,"description":"Too Many Requests"}`))
	}))
	defer srv.Close()

	old := telegramAPI
	telegramAPI = srv.URL
	defer func() { telegramAPI = old }()

	p := &TelegramProvider{}
	_, err := p.Send(context.Background(),
		json.RawMessage(`{"botToken":"1:a","chatId":"5"}`), "hi")
	if err == nil {
		t.Fatal("expected an error")
	}
	if IsPermanent(err) {
		t.Errorf("429 should be retryable, got permanent: %v", err)
	}
}

func TestTelegramServerErrorIsRetryable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer srv.Close()

	old := telegramAPI
	telegramAPI = srv.URL
	defer func() { telegramAPI = old }()

	p := &TelegramProvider{}
	if _, err := p.Send(context.Background(),
		json.RawMessage(`{"botToken":"1:a","chatId":"5"}`), "hi"); IsPermanent(err) {
		t.Errorf("502 should be retryable: %v", err)
	}
}

func TestTelegramTruncatesLongMessage(t *testing.T) {
	var gotLen int

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Text string `json:"text"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotLen = len(body.Text)
		_, _ = w.Write([]byte(`{"ok":true,"result":{"message_id":1}}`))
	}))
	defer srv.Close()

	old := telegramAPI
	telegramAPI = srv.URL
	defer func() { telegramAPI = old }()

	p := &TelegramProvider{}
	if _, err := p.Send(context.Background(),
		json.RawMessage(`{"botToken":"1:a","chatId":"5"}`), strings.Repeat("x", 9000)); err != nil {
		t.Fatalf("send: %v", err)
	}
	// Telegram rejects over 4096 outright, so truncating beats failing.
	if gotLen > 4096 {
		t.Errorf("text length = %d, want <= 4096", gotLen)
	}
}

func TestTelegramBadConfigIsPermanent(t *testing.T) {
	p := &TelegramProvider{}
	_, err := p.Send(context.Background(), json.RawMessage(`{}`), "hi")
	if !IsPermanent(err) {
		t.Errorf("an unusable config should be permanent, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// WhatsApp delivery
// ---------------------------------------------------------------------------

func TestWhatsAppMetaSend(t *testing.T) {
	var auth, body string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization")
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		_, _ = w.Write([]byte(`{"messages":[{"id":"wamid.XYZ"}]}`))
	}))
	defer srv.Close()

	old := metaGraphAPI
	metaGraphAPI = srv.URL
	defer func() { metaGraphAPI = old }()

	p := &WhatsAppProvider{}
	id, err := p.Send(context.Background(),
		json.RawMessage(`{"accessToken":"tok","phoneNumberId":"999","recipient":"+919876543210"}`),
		"filled")
	if err != nil {
		t.Fatalf("send: %v", err)
	}
	if id != "wamid.XYZ" {
		t.Errorf("id = %q", id)
	}
	if auth != "Bearer tok" {
		t.Errorf("auth header = %q", auth)
	}
	// The Cloud API wants the number without the leading plus.
	if !strings.Contains(body, `"to":"919876543210"`) {
		t.Errorf("body = %s", body)
	}
}

func TestWhatsAppWebhookSend(t *testing.T) {
	var body string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		_, _ = w.Write([]byte(`{"message_id":"hook-1"}`))
	}))
	defer srv.Close()

	p := &WhatsAppProvider{}
	cfg := `{"apiUrl":"` + srv.URL + `","recipient":"+919876543210"}`
	id, err := p.Send(context.Background(), json.RawMessage(cfg), "ping")
	if err != nil {
		t.Fatalf("send: %v", err)
	}
	if id != "hook-1" {
		t.Errorf("id = %q", id)
	}
	// The webhook path keeps the plus, unlike the Cloud API.
	if !strings.Contains(body, `"to":"+919876543210"`) {
		t.Errorf("body = %s", body)
	}
}

func TestWhatsAppAuthFailureIsPermanent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"Invalid OAuth token"}}`))
	}))
	defer srv.Close()

	old := metaGraphAPI
	metaGraphAPI = srv.URL
	defer func() { metaGraphAPI = old }()

	p := &WhatsAppProvider{}
	_, err := p.Send(context.Background(),
		json.RawMessage(`{"accessToken":"bad","phoneNumberId":"1","recipient":"+91987"}`), "hi")
	if !IsPermanent(err) {
		t.Errorf("401 should be permanent: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

func TestRegistryHasBuiltins(t *testing.T) {
	r := NewRegistry()
	for _, name := range []string{ChannelTelegram, ChannelWhatsApp} {
		if _, ok := r.Get(name); !ok {
			t.Errorf("%s provider is not registered", name)
		}
	}
	if _, ok := r.Get("carrier-pigeon"); ok {
		t.Error("unknown channel should not resolve")
	}
}

// ---------------------------------------------------------------------------
// Config encryption
// ---------------------------------------------------------------------------

func testEncryptor(t *testing.T) *security.Encryptor {
	t.Helper()
	key := make([]byte, 32)
	for i := range key {
		key[i] = byte(i)
	}
	e, err := security.NewEncryptor(key)
	if err != nil {
		t.Fatalf("build encryptor: %v", err)
	}
	return e
}

func TestEncryptDecryptRoundTrip(t *testing.T) {
	e := testEncryptor(t)
	original := json.RawMessage(`{"botToken":"123:secret","chatId":"-100"}`)

	sealed, err := EncryptConfig(e, original)
	if err != nil {
		t.Fatalf("encrypt: %v", err)
	}
	// The token must not be readable in the stored column.
	if strings.Contains(string(sealed), "123:secret") {
		t.Fatalf("plaintext token leaked into the stored config: %s", sealed)
	}
	if !IsEncrypted(sealed) {
		t.Error("sealed config should be reported as encrypted")
	}

	plain, err := DecryptConfig(e, sealed)
	if err != nil {
		t.Fatalf("decrypt: %v", err)
	}
	if !strings.Contains(string(plain), "123:secret") {
		t.Errorf("round trip lost the token: %s", plain)
	}
}

func TestDecryptPlaintextPassesThrough(t *testing.T) {
	e := testEncryptor(t)
	// Rows written before encryption was switched on must keep working.
	plain := json.RawMessage(`{"botToken":"1:a"}`)

	out, err := DecryptConfig(e, plain)
	if err != nil {
		t.Fatalf("decrypt: %v", err)
	}
	if string(out) != string(plain) {
		t.Errorf("got %s, want the original plaintext", out)
	}
}

func TestDecryptEncryptedWithoutKeyFails(t *testing.T) {
	e := testEncryptor(t)
	sealed, _ := EncryptConfig(e, json.RawMessage(`{"botToken":"1:a"}`))

	// Losing the key must be a loud error, not a silently empty config.
	if _, err := DecryptConfig(nil, sealed); err == nil {
		t.Error("expected an error decrypting without a key")
	}
}

func TestEncryptWithoutEncryptorPassesThrough(t *testing.T) {
	cfg := json.RawMessage(`{"botToken":"1:a"}`)
	out, err := EncryptConfig(nil, cfg)
	if err != nil {
		t.Fatalf("encrypt: %v", err)
	}
	if string(out) != string(cfg) {
		t.Errorf("got %s, want passthrough when no key is configured", out)
	}
}

func TestHasCredentials(t *testing.T) {
	e := testEncryptor(t)

	sealed, _ := EncryptConfig(e, json.RawMessage(`{"botToken":"1:a"}`))
	if !HasCredentials(e, sealed) {
		t.Error("sealed config with a token should report configured")
	}

	empty, _ := EncryptConfig(e, json.RawMessage(`{}`))
	if HasCredentials(e, empty) {
		t.Error("empty config should not report configured")
	}

	blank, _ := EncryptConfig(e, json.RawMessage(`{"botToken":""}`))
	if HasCredentials(e, blank) {
		t.Error("blank values should not count as configured")
	}
}
