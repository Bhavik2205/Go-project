package notifications

import (
	"encoding/json"
	"fmt"

	"github.com/Bhavik2205/ML-Bot/internal/security"
)

// envelopeField holds the encrypted config blob. Wrapping it in JSON keeps the
// column valid JSONB either way, so an encrypted and a plaintext row can
// coexist and older rows keep working.
type cryptoEnvelope struct {
	Encrypted string `json:"__enc,omitempty"`
}

// EncryptConfig protects channel credentials at rest.
//
// Bot tokens and API keys are bearer credentials: anyone holding one can send
// as that account, so they must not sit in the database as plaintext. With no
// encryptor configured the config is stored as-is, which is a deliberate
// degradation rather than a silent failure — the caller logs it.
func EncryptConfig(e *security.Encryptor, cfg json.RawMessage) (json.RawMessage, error) {
	if e == nil {
		return cfg, nil
	}
	if len(cfg) == 0 {
		cfg = json.RawMessage(`{}`)
	}

	sealed, err := e.EncryptString(string(cfg))
	if err != nil {
		return nil, fmt.Errorf("encrypt channel config: %w", err)
	}

	out, err := json.Marshal(cryptoEnvelope{Encrypted: sealed})
	if err != nil {
		return nil, fmt.Errorf("wrap encrypted config: %w", err)
	}
	return out, nil
}

// DecryptConfig reverses EncryptConfig.
//
// A config without the envelope marker is returned unchanged, so rows written
// before encryption was switched on still work.
func DecryptConfig(e *security.Encryptor, stored json.RawMessage) (json.RawMessage, error) {
	if len(stored) == 0 {
		return json.RawMessage(`{}`), nil
	}

	var env cryptoEnvelope
	if err := json.Unmarshal(stored, &env); err != nil || env.Encrypted == "" {
		// Not an envelope: treat it as plaintext config.
		return stored, nil
	}

	if e == nil {
		return nil, fmt.Errorf(
			"channel config is encrypted but no encryption key is configured; set %s",
			security.EncryptionKeyEnv)
	}

	plain, err := e.DecryptString(env.Encrypted)
	if err != nil {
		return nil, fmt.Errorf("decrypt channel config: %w", err)
	}
	return json.RawMessage(plain), nil
}

// IsEncrypted reports whether a stored config is sealed.
func IsEncrypted(stored json.RawMessage) bool {
	var env cryptoEnvelope
	return json.Unmarshal(stored, &env) == nil && env.Encrypted != ""
}

// HasCredentials reports whether a decrypted config holds anything, without
// returning the values themselves.
func HasCredentials(e *security.Encryptor, stored json.RawMessage) bool {
	plain, err := DecryptConfig(e, stored)
	if err != nil {
		// Undecryptable but present still counts as configured; reporting
		// "not configured" would invite the user to overwrite it blindly.
		return IsEncrypted(stored)
	}
	var m map[string]any
	if err := json.Unmarshal(plain, &m); err != nil {
		return false
	}
	for _, v := range m {
		if s, ok := v.(string); ok && s != "" {
			return true
		}
	}
	return false
}
