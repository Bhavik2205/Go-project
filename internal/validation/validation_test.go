package validation_test

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Bhavik2205/ML-Bot/internal/validation"
)

type sampleRequest struct {
	Email    string `json:"email"    validate:"required,email"`
	Password string `json:"password" validate:"required,min=8"`
	Age      int    `json:"age"      validate:"omitempty,min=0,max=150"`
}

func bindRequest(t *testing.T, body string) (*httptest.ResponseRecorder, bool) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	var dst sampleRequest
	ok := validation.BindAndValidate(rr, req, &dst)
	return rr, ok
}

func TestBindAndValidate_Valid(t *testing.T) {
	_, ok := bindRequest(t, `{"email":"user@example.com","password":"strongpass"}`)
	if !ok {
		t.Error("expected BindAndValidate to return true for valid input")
	}
}

func TestBindAndValidate_InvalidEmail(t *testing.T) {
	rr, ok := bindRequest(t, `{"email":"not-an-email","password":"strongpass"}`)
	if ok {
		t.Error("expected false for invalid email")
	}
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

func TestBindAndValidate_MissingRequired(t *testing.T) {
	rr, ok := bindRequest(t, `{"email":"user@example.com"}`)
	if ok {
		t.Error("expected false for missing required password")
	}
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

func TestBindAndValidate_PasswordTooShort(t *testing.T) {
	rr, ok := bindRequest(t, `{"email":"user@example.com","password":"short"}`)
	if ok {
		t.Error("expected false for password < 8 chars")
	}
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

func TestBindAndValidate_MalformedJSON(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader([]byte(`{not valid json`)))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	var dst sampleRequest
	ok := validation.BindAndValidate(rr, req, &dst)
	if ok {
		t.Error("expected false for malformed JSON")
	}
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

func TestBindAndValidate_EmptyBody(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader([]byte{}))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	var dst sampleRequest
	ok := validation.BindAndValidate(rr, req, &dst)
	if ok {
		t.Error("expected false for empty body")
	}
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

func TestBindAndValidate_ErrorResponseIsJSON(t *testing.T) {
	rr, _ := bindRequest(t, `{"email":"bad","password":"x"}`)
	ct := rr.Header().Get("Content-Type")
	if !strings.Contains(ct, "application/json") {
		t.Errorf("expected JSON Content-Type for error, got %q", ct)
	}
}

func TestBindAndValidate_ValidOptionalField(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/",
		strings.NewReader(`{"email":"u@example.com","password":"strongpass","age":25}`))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	var dst sampleRequest
	ok := validation.BindAndValidate(rr, req, &dst)
	if !ok {
		t.Errorf("expected true for valid optional field, got 400: %s", rr.Body.String())
	}
}

func TestBindAndValidate_InvalidOptionalField(t *testing.T) {
	rr, ok := bindRequest(t, `{"email":"u@example.com","password":"strongpass","age":200}`)
	if ok {
		t.Error("expected false for age > 150")
	}
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}
