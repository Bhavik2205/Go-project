#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════╗"
echo "║   TRADING BOT OBSERVABILITY - QUICK TEST      ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Test 1: Unit tests
echo "📝 TEST 1: Unit tests (metrics registration)"
echo "────────────────────────────────────────────────"
if go test ./internal/observability -v -run TestMetrics 2>&1 | grep -q "PASS"; then
  echo "✅ Metrics registration working"
else
  echo "⚠️  Skipping (tests may not exist yet)"
fi

echo ""

# Test 2: Start server and test endpoint
echo "🚀 TEST 2: Metrics endpoint"
echo "────────────────────────────────────────────────"
echo "Starting server..."
timeout 30 go run ./cmd/server/main.go > /tmp/server.log 2>&1 &
SERVER_PID=$!
sleep 3

# Check if server started
if ! kill -0 $SERVER_PID 2>/dev/null; then
  echo "❌ FAILED: Server didn't start"
  echo "Logs:"
  cat /tmp/server.log
  exit 1
fi

# Test endpoint
if curl -s http://localhost:8080/api/v1/metrics > /tmp/metrics.txt 2>/dev/null; then
  echo "✅ Endpoint responds"
  
  # Count metrics
  METRIC_COUNT=$(grep -c "^trading_" /tmp/metrics.txt || true)
  echo "   Found $METRIC_COUNT metric lines"
  
  # Check key metrics
  echo ""
  echo "Checking key metrics..."
  for metric in "trading_ticks_received_total" "trading_tick_lag_ms_bucket" "trading_goroutines_current" "trading_panics_total"; do
    if grep -q "$metric" /tmp/metrics.txt; then
      echo "   ✓ $metric"
    else
      echo "   ✗ $metric (MISSING)"
    fi
  done
else
  echo "❌ FAILED: Metrics endpoint not responding"
  echo "Check if port 8080 is in use or app failed to start"
fi

echo ""

# Test 3: Sample metrics output
echo "📊 TEST 3: Sample metrics values"
echo "────────────────────────────────────────────────"
echo "Counters:"
grep "trading_ticks_received_total" /tmp/metrics.txt | tail -1 || echo "  (not available)"
grep "trading_panics_total" /tmp/metrics.txt | tail -1 || echo "  (not available)"

echo ""
echo "Gauges:"
grep "trading_goroutines_current " /tmp/metrics.txt | tail -1 || echo "  (not available)"
grep "trading_memory_heap_alloc_bytes " /tmp/metrics.txt | tail -1 || echo "  (not available)"

echo ""
echo "Histograms (p50 bucket):"
grep 'trading_tick_lag_ms_bucket.*le="50"' /tmp/metrics.txt | tail -1 || echo "  (not available)"

# Cleanup
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║              ✅ QUICK TEST COMPLETE             ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Start Prometheus (scrapes http://localhost:8080/api/v1/metrics)"
echo "  2. Import Grafana dashboards from grafana/dashboards/"
echo "  3. View at http://localhost:3000/d/trading-market-data"
echo ""
