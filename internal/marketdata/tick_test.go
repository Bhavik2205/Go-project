package marketdata

import (
	"encoding/json"
	"testing"
	"time"
)

func TestNormalizedTickJSON(t *testing.T) {
	tick := NormalizedTick{
		InstrumentToken: 12345,
		Symbol:          "NSE:RELIANCE",
		EventTime:       time.Now(),
		IngestTime:      time.Now(),
		LastPrice:       2500.50,
	}
	data, err := json.Marshal(tick)
	if err != nil {
		t.Fatal(err)
	}
	var decoded NormalizedTick
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.InstrumentToken != tick.InstrumentToken {
		t.Errorf("expected %d, got %d", tick.InstrumentToken, decoded.InstrumentToken)
	}
}
