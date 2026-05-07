package security

import (
	"encoding/base64"
	"errors"
	"testing"
)

func TestEncryptorRoundTrip(t *testing.T) {
	encodedKey, err := GenerateKey()
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}

	key, err := base64.StdEncoding.DecodeString(encodedKey)
	if err != nil {
		t.Fatalf("DecodeString() error = %v", err)
	}

	encryptor, err := NewEncryptor(key)
	if err != nil {
		t.Fatalf("NewEncryptor() error = %v", err)
	}

	encrypted, err := encryptor.EncryptString("kite-access-token")
	if err != nil {
		t.Fatalf("EncryptString() error = %v", err)
	}
	if encrypted == "kite-access-token" {
		t.Fatalf("ciphertext should not equal plaintext")
	}

	decrypted, err := encryptor.DecryptString(encrypted)
	if err != nil {
		t.Fatalf("DecryptString() error = %v", err)
	}
	if decrypted != "kite-access-token" {
		t.Fatalf("decrypted = %q, want %q", decrypted, "kite-access-token")
	}
}

func TestNewEncryptorRejectsBadKeys(t *testing.T) {
	if _, err := NewEncryptor(nil); !errors.Is(err, ErrMissingEncryptionKey) {
		t.Fatalf("nil key error = %v, want ErrMissingEncryptionKey", err)
	}

	if _, err := NewEncryptor([]byte("short")); !errors.Is(err, ErrInvalidEncryptionKey) {
		t.Fatalf("short key error = %v, want ErrInvalidEncryptionKey", err)
	}
}

func TestEncryptorEmptyString(t *testing.T) {
	key := make([]byte, 32)
	encryptor, err := NewEncryptor(key)
	if err != nil {
		t.Fatalf("NewEncryptor() error = %v", err)
	}

	encrypted, err := encryptor.EncryptString("")
	if err != nil {
		t.Fatalf("EncryptString(empty) error = %v", err)
	}
	if encrypted != "" {
		t.Fatalf("empty encrypt = %q, want empty", encrypted)
	}

	decrypted, err := encryptor.DecryptString("")
	if err != nil {
		t.Fatalf("DecryptString(empty) error = %v", err)
	}
	if decrypted != "" {
		t.Fatalf("empty decrypt = %q, want empty", decrypted)
	}
}
