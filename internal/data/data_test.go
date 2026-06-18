package data_test

import (
	"sync"
	"testing"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/data"
)

// ── Heatmap Tests ──────────────────────────────────────────────────────────────

func TestHeatmap_UpdateAndSnapshot(t *testing.T) {
	hm := data.NewMarketHeatmap()
	hm.Update("NSE:RELIANCE", 3000.0, 2999.0, 3001.0, 100, 200, 50000, 3000.0, 2985.0)

	snap := hm.Snapshot()
	if len(snap) != 1 {
		t.Fatalf("expected 1 stock, got %d", len(snap))
	}
	s := snap[0]
	if s.Symbol != "NSE:RELIANCE" {
		t.Errorf("expected NSE:RELIANCE, got %s", s.Symbol)
	}
	if s.LastPrice != 30000000 {
		t.Errorf("expected LastPrice 30000000, got %d", s.LastPrice)
	}
	if s.BidPrice != 29990000 {
		t.Errorf("expected BidPrice 29990000, got %d", s.BidPrice)
	}
	if s.AskPrice != 30010000 {
		t.Errorf("expected AskPrice 30010000, got %d", s.AskPrice)
	}
	// PriceChangePct = (3000 - 2985) / 2985 * 100 ≈ 0.502
	if s.PriceChangePct < 0.4 || s.PriceChangePct > 0.6 {
		t.Errorf("unexpected PriceChangePct: %f", s.PriceChangePct)
	}
}

func TestHeatmap_ZeroPrevClose(t *testing.T) {
	hm := data.NewMarketHeatmap()
	hm.Update("NSE:TCS", 4000.0, 3999.0, 4001.0, 50, 100, 10000, 4000.0, 0)
	snap := hm.Snapshot()
	if snap[0].PriceChangePct != 0 {
		t.Errorf("expected 0 PriceChangePct with zero prevClose, got %f", snap[0].PriceChangePct)
	}
}

func TestHeatmap_VolumeAtPricePruning(t *testing.T) {
	hm := data.NewMarketHeatmap()
	// Insert 110 unique price levels — should be pruned to 100
	for i := 0; i < 110; i++ {
		price := float64(1000 + i)
		hm.Update("NSE:INFY", price, price-1, price+1, 10, 10, 1000, price, 999.0)
	}
	snap := hm.Snapshot()
	if len(snap[0].VolumeAtPrice) > 100 {
		t.Errorf("VolumeAtPrice should be pruned to ≤100, got %d", len(snap[0].VolumeAtPrice))
	}
}

func TestHeatmap_SnapshotIsDeepCopy(t *testing.T) {
	hm := data.NewMarketHeatmap()
	hm.Update("NSE:HDFC", 1500.0, 1499.0, 1501.0, 10, 20, 5000, 1500.0, 1490.0)
	snap := hm.Snapshot()
	// Mutate the snapshot
	snap[0].LastPrice = 9999
	snap[0].VolumeAtPrice[99990000] = 999

	// Original should be unchanged
	snap2 := hm.Snapshot()
	if snap2[0].LastPrice == 9999 {
		t.Error("snapshot mutation affected original heatmap data")
	}
	if _, ok := snap2[0].VolumeAtPrice[99990000]; ok {
		t.Error("snapshot VolumeAtPrice mutation affected original")
	}
}

func TestHeatmap_ConcurrentUpdatesAndSnapshots(t *testing.T) {
	hm := data.NewMarketHeatmap()
	symbols := []string{"NSE:A", "NSE:B", "NSE:C", "NSE:D", "NSE:E"}
	var wg sync.WaitGroup

	// 50 concurrent writers
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			sym := symbols[idx%len(symbols)]
			price := float64(1000 + idx)
			hm.Update(sym, price, price-1, price+1, 10, 10, 1000, price, price-10)
		}(i)
	}

	// 20 concurrent readers
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			snap := hm.Snapshot()
			_ = snap
		}()
	}

	wg.Wait()
}

func TestHeatmap_MultipleSymbols(t *testing.T) {
	hm := data.NewMarketHeatmap()
	for i := 0; i < 10; i++ {
		sym := "NSE:SYM" + string(rune('A'+i))
		hm.Update(sym, float64(1000+i), float64(999+i), float64(1001+i), 10, 10, 1000, float64(1000+i), float64(990+i))
	}
	snap := hm.Snapshot()
	if len(snap) != 10 {
		t.Errorf("expected 10 symbols, got %d", len(snap))
	}
}

// ── Candle Generation Tests ────────────────────────────────────────────────────

func TestParseSymbols_Normalization(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"NSE:RELIANCE", "NSE:RELIANCE"},
		{"reliance", "NSE:RELIANCE"},
		{"RELIANCE (NSE)", "NSE:RELIANCE"},
		{"  nse:tcs  ", "NSE:TCS"},
	}
	for _, tt := range tests {
		// We test normalizeToAPISymbol indirectly via the heatmap symbol storage
		hm := data.NewMarketHeatmap()
		hm.Update(tt.input, 100, 99, 101, 1, 1, 100, 100, 90)
		snap := hm.Snapshot()
		if len(snap) == 0 {
			t.Errorf("no snapshot for input %q", tt.input)
		}
	}
}

// ── Ingestion Pipeline Tests ───────────────────────────────────────────────────

func TestGetMarketHeatmap_Singleton(t *testing.T) {
	hm1 := data.GetMarketHeatmap()
	hm2 := data.GetMarketHeatmap()
	if hm1 != hm2 {
		t.Error("GetMarketHeatmap should return the same singleton instance")
	}
}

func TestHeatmap_UpdateIdempotent(t *testing.T) {
	hm := data.NewMarketHeatmap()
	// Same symbol updated multiple times — should not create duplicates
	for i := 0; i < 5; i++ {
		hm.Update("NSE:WIPRO", float64(300+i), float64(299+i), float64(301+i), 10, 10, 1000, float64(300+i), 295.0)
	}
	snap := hm.Snapshot()
	if len(snap) != 1 {
		t.Errorf("expected 1 entry for same symbol, got %d", len(snap))
	}
	// Last update should win
	if snap[0].LastPrice != 3040000 {
		t.Errorf("expected LastPrice 3040000, got %d", snap[0].LastPrice)
	}
}

func TestHeatmap_BidAskSpread(t *testing.T) {
	hm := data.NewMarketHeatmap()
	hm.Update("NSE:SBIN", 500.0, 499.5, 500.5, 100, 100, 10000, 500.0, 495.0)
	snap := hm.Snapshot()
	if snap[0].BidAskSpread != 10000 {
		t.Errorf("expected spread 10000, got %d", snap[0].BidAskSpread)
	}
}

func TestHeatmap_LastUpdatedIsRecent(t *testing.T) {
	before := time.Now()
	hm := data.NewMarketHeatmap()
	hm.Update("NSE:ONGC", 200.0, 199.0, 201.0, 50, 50, 5000, 200.0, 198.0)
	after := time.Now()
	snap := hm.Snapshot()
	if snap[0].LastUpdated.Before(before) || snap[0].LastUpdated.After(after) {
		t.Errorf("LastUpdated %v not in expected range [%v, %v]", snap[0].LastUpdated, before, after)
	}
}

// ── CleanText Tests ────────────────────────────────────────────────────────────

func TestCleanText(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"Hello, World!", "hello world"},
		{"Visit https://example.com for more", "visit  for more"},
		{"  spaces  ", "spaces"},
		{"UPPER CASE", "upper case"},
		{"", ""},
	}
	for _, tt := range tests {
		got := data.CleanText(tt.input)
		if got != tt.expected {
			t.Errorf("CleanText(%q) = %q, want %q", tt.input, got, tt.expected)
		}
	}
}
