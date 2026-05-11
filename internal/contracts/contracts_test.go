package contracts_test

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/contracts"
)

func TestNewSuccess_Structure(t *testing.T) {
	type payload struct {
		Name string `json:"name"`
	}
	resp := contracts.NewSuccess("req_123", payload{Name: "test"})
	if resp.Meta.RequestID != "req_123" {
		t.Errorf("expected requestId req_123, got %s", resp.Meta.RequestID)
	}
	if resp.Meta.Version != contracts.APIVersionV1 {
		t.Errorf("expected version v1, got %s", resp.Meta.Version)
	}
	if resp.Data.Name != "test" {
		t.Errorf("expected name test, got %s", resp.Data.Name)
	}
	if resp.Meta.ServerTime.IsZero() {
		t.Error("expected non-zero ServerTime")
	}
}

func TestNewError_Structure(t *testing.T) {
	resp := contracts.NewError("req_456", "VALIDATION_ERROR", "field is required", map[string]string{"field": "email"})
	if resp.Error.Code != "VALIDATION_ERROR" {
		t.Errorf("expected VALIDATION_ERROR, got %s", resp.Error.Code)
	}
	if resp.Error.Message != "field is required" {
		t.Errorf("unexpected message: %s", resp.Error.Message)
	}
	if resp.Meta.RequestID != "req_456" {
		t.Errorf("expected req_456, got %s", resp.Meta.RequestID)
	}
}

func TestNewSuccess_JSONRoundtrip(t *testing.T) {
	resp := contracts.NewSuccess("req_789", map[string]int{"count": 42})
	b, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := decoded["data"]; !ok {
		t.Error("expected 'data' key in JSON")
	}
	if _, ok := decoded["meta"]; !ok {
		t.Error("expected 'meta' key in JSON")
	}
}

func TestNewError_JSONRoundtrip(t *testing.T) {
	resp := contracts.NewError("req_000", "NOT_FOUND", "resource not found", nil)
	b, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := decoded["error"]; !ok {
		t.Error("expected 'error' key in JSON")
	}
}

func TestNewMeta_ServerTimeIsRecent(t *testing.T) {
	before := time.Now()
	meta := contracts.NewMeta("req_test", time.Now())
	after := time.Now()
	if meta.ServerTime.Before(before) || meta.ServerTime.After(after) {
		t.Errorf("ServerTime %v not in expected range", meta.ServerTime)
	}
}

func TestPagination_OmittedWhenNil(t *testing.T) {
	resp := contracts.NewSuccess("req_page", "data")
	b, _ := json.Marshal(resp)
	var decoded map[string]any
	json.Unmarshal(b, &decoded)
	meta := decoded["meta"].(map[string]any)
	if _, ok := meta["pagination"]; ok {
		t.Error("pagination should be omitted when nil")
	}
}

func TestWSEvent_Structure(t *testing.T) {
	type tickData struct {
		Symbol    string  `json:"symbol"`
		LastPrice float64 `json:"lastPrice"`
	}
	evt := contracts.NewWSEvent("MARKET_TICK", contracts.WSTopicMarketTicks, "evt_001", tickData{
		Symbol:    "NSE:RELIANCE",
		LastPrice: 3000.5,
	})
	if evt.Type != "MARKET_TICK" {
		t.Errorf("expected MARKET_TICK, got %s", evt.Type)
	}
	if evt.Topic != contracts.WSTopicMarketTicks {
		t.Errorf("expected %s, got %s", contracts.WSTopicMarketTicks, evt.Topic)
	}
	if evt.Data.Symbol != "NSE:RELIANCE" {
		t.Errorf("expected NSE:RELIANCE, got %s", evt.Data.Symbol)
	}
	if evt.ServerTime.IsZero() {
		t.Error("expected non-zero ServerTime")
	}
}

func TestWSError_Structure(t *testing.T) {
	evt := contracts.NewWSError("evt_err", "system", "SUBSCRIPTION_DENIED", "not allowed", nil)
	if evt.Type != contracts.WSMessageError {
		t.Errorf("expected ERROR type, got %s", evt.Type)
	}
	if evt.Error.Code != "SUBSCRIPTION_DENIED" {
		t.Errorf("expected SUBSCRIPTION_DENIED, got %s", evt.Error.Code)
	}
}
