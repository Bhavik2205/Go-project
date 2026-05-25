package security

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"os"
)

const (
	EncryptionKeyEnv = "TRADINGBOT_ENCRYPTION_KEY"
	encryptedPrefix  = "v1:"
)

var (
	ErrMissingEncryptionKey = errors.New("missing encryption key")
	ErrInvalidEncryptionKey = errors.New("encryption key must decode to 32 bytes")
)

type Encryptor struct {
	aead cipher.AEAD
}

func NewEncryptor(rawKey []byte) (*Encryptor, error) {
	if len(rawKey) == 0 {
		return nil, ErrMissingEncryptionKey
	}
	if len(rawKey) != 32 {
		return nil, ErrInvalidEncryptionKey
	}

	block, err := aes.NewCipher(rawKey)
	if err != nil {
		return nil, fmt.Errorf("create aes cipher: %w", err)
	}

	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("create gcm cipher: %w", err)
	}

	return &Encryptor{aead: aead}, nil
}

func NewEncryptorFromEnv() (*Encryptor, error) {
	encodedKey := os.Getenv(EncryptionKeyEnv)
	if encodedKey == "" {
		return nil, ErrMissingEncryptionKey
	}

	key, err := base64.StdEncoding.DecodeString(encodedKey)
	if err != nil {
		return nil, fmt.Errorf("decode encryption key: %w", err)
	}

	return NewEncryptor(key)
}

func GenerateKey() (string, error) {
	key := make([]byte, 32)
	if _, err := io.ReadFull(rand.Reader, key); err != nil {
		return "", fmt.Errorf("generate encryption key: %w", err)
	}
	return base64.StdEncoding.EncodeToString(key), nil
}

func (e *Encryptor) EncryptString(plaintext string) (string, error) {
	if plaintext == "" {
		return "", nil
	}

	nonce := make([]byte, e.aead.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return "", fmt.Errorf("generate nonce: %w", err)
	}

	ciphertext := e.aead.Seal(nil, nonce, []byte(plaintext), nil)
	payload := append(nonce, ciphertext...)
	return encryptedPrefix + base64.StdEncoding.EncodeToString(payload), nil
}

func (e *Encryptor) DecryptString(encrypted string) (string, error) {
	if encrypted == "" {
		return "", nil
	}
	if len(encrypted) <= len(encryptedPrefix) || encrypted[:len(encryptedPrefix)] != encryptedPrefix {
		return "", errors.New("invalid encrypted payload version")
	}

	payload, err := base64.StdEncoding.DecodeString(encrypted[len(encryptedPrefix):])
	if err != nil {
		return "", fmt.Errorf("decode encrypted payload: %w", err)
	}

	nonceSize := e.aead.NonceSize()
	if len(payload) <= nonceSize {
		return "", errors.New("encrypted payload too short")
	}

	nonce := payload[:nonceSize]
	ciphertext := payload[nonceSize:]
	plaintext, err := e.aead.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return "", fmt.Errorf("decrypt payload: %w", err)
	}

	return string(plaintext), nil
}
