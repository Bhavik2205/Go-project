package security

import "testing"

func TestIsSensitiveKey(t *testing.T) {
	cases := map[string]bool{
		"apiSecret":        true,
		"access_token":     true,
		"Authorization":    true,
		"telegramBotToken": true,
		"userName":         false,
		"broker":           false,
	}

	for key, expected := range cases {
		if got := IsSensitiveKey(key); got != expected {
			t.Fatalf("IsSensitiveKey(%q) = %v, want %v", key, got, expected)
		}
	}
}

func TestRedactMap(t *testing.T) {
	input := map[string]any{
		"apiSecret": "secret-value",
		"profile": map[string]any{
			"userName":    "Bhavik",
			"accessToken": "token-value",
		},
	}

	redacted := RedactMap(input)
	if redacted["apiSecret"] != RedactedValue {
		t.Fatalf("expected apiSecret to be redacted")
	}

	profile, ok := redacted["profile"].(map[string]any)
	if !ok {
		t.Fatalf("expected nested profile map")
	}
	if profile["userName"] != "Bhavik" {
		t.Fatalf("expected userName to be preserved")
	}
	if profile["accessToken"] != RedactedValue {
		t.Fatalf("expected nested accessToken to be redacted")
	}
}
