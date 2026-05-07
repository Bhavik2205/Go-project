package security

import "strings"

const RedactedValue = "[REDACTED]"

var sensitiveKeyFragments = []string{
	"access_token",
	"accesstoken",
	"api_secret",
	"apisecret",
	"authorization",
	"bot_token",
	"bottoken",
	"jwt",
	"password",
	"refresh_token",
	"refreshtoken",
	"secret",
	"token",
}

func IsSensitiveKey(key string) bool {
	normalized := strings.ToLower(strings.ReplaceAll(strings.TrimSpace(key), "-", "_"))
	for _, fragment := range sensitiveKeyFragments {
		if strings.Contains(normalized, fragment) {
			return true
		}
	}
	return false
}

func RedactString(value string) string {
	if value == "" {
		return ""
	}
	if len(value) <= 4 {
		return RedactedValue
	}
	return value[:2] + strings.Repeat("*", min(len(value)-4, 8)) + value[len(value)-2:]
}

func RedactMap(input map[string]any) map[string]any {
	if input == nil {
		return nil
	}

	output := make(map[string]any, len(input))
	for key, value := range input {
		if IsSensitiveKey(key) {
			output[key] = RedactedValue
			continue
		}

		switch typed := value.(type) {
		case map[string]any:
			output[key] = RedactMap(typed)
		case map[string]string:
			output[key] = RedactStringMap(typed)
		default:
			output[key] = value
		}
	}
	return output
}

func RedactStringMap(input map[string]string) map[string]string {
	if input == nil {
		return nil
	}

	output := make(map[string]string, len(input))
	for key, value := range input {
		if IsSensitiveKey(key) {
			output[key] = RedactedValue
			continue
		}
		output[key] = value
	}
	return output
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
