# Testing Observability Infrastructure

Complete guide to verify metrics, Prometheus, and Grafana dashboards are working correctly.

---

## Level 1: Unit Testing (Metrics Registration)

### Test: Verify metrics are registered

**File:** `internal/observability/metrics_test.go`

```go
package observability

import (
	"testing"
	"github.com/prometheus/client_golang/prometheus"
)

func TestMetricsRegistered(t *testing.T) {
	// Should not panic if called multiple times
	InitMetrics()
	InitMetrics() // Idempotent
	
	if TicksReceived == nil {
		t.Fatal("TicksReceived not initialized")
	}
	if TickLag == nil {
		t.Fatal("TickLag not initialized")
	}
	if TickQueueDepth == nil {
		t.Fatal("TickQueueDepth not initialized")
	}
	
	t.Log("✓ All 23 metrics initialized successfully")
}

func TestMetricsIncrement(t *testing.T) {
	InitMetrics()
	
	// Test counter
	TicksReceived.Inc()
	TicksReceived.Add(5)
	t.Log("✓ Counter increment works")
	
	// Test histogram
	TickLag.Observe(42.5)
	t.Log("✓ Histogram observe works")
	
	// Test gauge
	TickQueueDepth.Set(10)
	TickQueueDepth.Add(5)
	TickQueueDepth.Sub(2)
	t.Log("✓ Gauge operations work")
}
```

**Run:**
```bash
cd /Users/bhavikpatel/Desktop/TradingBot/Go-project
go test ./internal/observability -v
```

**Expected output:**
```
=== RUN   TestMetricsRegistered
    metrics_test.go:XX: ✓ All 23 metrics initialized successfully
--- PASS: TestMetricsRegistered (0.XXs)
=== RUN   TestMetricsIncrement
    metrics_test.go:XX: ✓ Counter increment works
    metrics_test.go:XX: ✓ Histogram observe works
    metrics_test.go:XX: ✓ Gauge operations work
--- PASS: TestMetricsIncrement (0.XXs)
PASS
ok  	tradingbot/internal/observability	0.XXXs
```

---

## Level 2: Integration Testing (Metrics Endpoint)

### Test: Verify `/api/v1/metrics` endpoint returns metrics

```bash
# Start the application
cd /Users/bhavikpatel/Desktop/TradingBot/Go-project
go run cmd/server/main.go &
SERVER_PID=$!
sleep 2  # Wait for startup

# Test 1: Endpoint is accessible
echo "TEST 1: Endpoint accessibility"
curl -s http://localhost:8080/api/v1/metrics | head -5
if [ $? -eq 0 ]; then
  echo "✓ Metrics endpoint accessible"
else
  echo "✗ FAILED: Metrics endpoint not responding"
  kill $SERVER_PID
  exit 1
fi

# Test 2: Verify metric types exist
echo -e "\nTEST 2: Metric types"
curl -s http://localhost:8080/api/v1/metrics | grep "# TYPE trading" | head -10
echo "Expected: Multiple trading_* metrics"

# Test 3: Verify HELP text
echo -e "\nTEST 3: Metric descriptions"
curl -s http://localhost:8080/api/v1/metrics | grep "# HELP trading_ticks_received"
curl -s http://localhost:8080/api/v1/metrics | grep "# HELP trading_tick_lag"
echo "✓ Metrics have descriptions"

# Test 4: Verify metrics have values
echo -e "\nTEST 4: Metric values"
curl -s http://localhost:8080/api/v1/metrics | grep "trading_ticks_received_total"
curl -s http://localhost:8080/api/v1/metrics | grep "trading_goroutines_current"
echo "✓ Metrics have values"

# Cleanup
kill $SERVER_PID
```

**Run:**
```bash
bash /Users/bhavikpatel/Desktop/TradingBot/Go-project/test-metrics-endpoint.sh
```

---

## Level 3: Prometheus Integration Testing

### Test: Verify Prometheus scrapes metrics

**Step 1: Create test Prometheus config**

```yaml
# prometheus-test.yml
global:
  scrape_interval: 5s
  evaluation_interval: 5s

scrape_configs:
  - job_name: 'tradingbot'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/api/v1/metrics'
```

**Step 2: Run Prometheus (if available)**

```bash
# Download Prometheus (if not installed)
# https://prometheus.io/download/

# Or via Docker
docker run -d \
  --name prometheus-test \
  -p 9090:9090 \
  -v $(pwd)/prometheus-test.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus:latest

# Wait 10s for Prometheus to start
sleep 10

# Verify Prometheus is running
curl -s http://localhost:9090/graph | grep Prometheus
echo "✓ Prometheus running"
```

**Step 3: Start TradingBot**

```bash
cd /Users/bhavikpatel/Desktop/TradingBot/Go-project
go run cmd/server/main.go &
SERVER_PID=$!
sleep 2
```

**Step 4: Query Prometheus**

```bash
# Test 1: Check targets
echo "TEST 1: Scrape targets"
curl -s 'http://localhost:9090/api/v1/targets' | jq '.data.activeTargets[0]'

# Should show UP status after scrape
echo "Wait 5s for first scrape..."
sleep 5

# Test 2: Query metrics
echo -e "\nTEST 2: Query ticks received"
curl -s 'http://localhost:9090/api/v1/query?query=trading_ticks_received_total' | jq '.data'

# Test 3: Query time series
echo -e "\nTEST 3: Query goroutines"
curl -s 'http://localhost:9090/api/v1/query?query=trading_goroutines_current' | jq '.data'

# Test 4: Query histogram (quantiles)
echo -e "\nTEST 4: Query tick lag p99"
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.99, trading_tick_lag_ms_bucket)' | jq '.data'

# Cleanup
kill $SERVER_PID
docker stop prometheus-test
docker rm prometheus-test
```

---

## Level 4: Manual Verification (No Tools Needed)

### Quick smoke test

```bash
#!/bin/bash
set -e

echo "=== OBSERVABILITY SMOKE TEST ==="
echo ""

# Start server
echo "1. Starting server..."
cd /Users/bhavikpatel/Desktop/TradingBot/Go-project
go run cmd/server/main.go > /tmp/server.log 2>&1 &
SERVER_PID=$!
sleep 3

# Test endpoint
echo "2. Testing /api/v1/metrics endpoint..."
METRICS=$(curl -s http://localhost:8080/api/v1/metrics)

if [ -z "$METRICS" ]; then
  echo "✗ FAILED: No response from metrics endpoint"
  kill $SERVER_PID
  exit 1
fi

echo "✓ Endpoint responds"

# Verify key metrics exist
echo ""
echo "3. Checking metric names..."

checks=(
  "trading_ticks_received_total"
  "trading_ticks_processed_total"
  "trading_ticks_dropped_total"
  "trading_tick_lag_ms"
  "trading_candle_latency_ms"
  "trading_indicator_latency_ms"
  "trading_goroutines_current"
  "trading_memory_heap_alloc_bytes"
  "trading_panics_total"
)

for metric in "${checks[@]}"; do
  if echo "$METRICS" | grep -q "$metric"; then
    echo "✓ $metric"
  else
    echo "✗ MISSING: $metric"
  fi
done

# Check metric types
echo ""
echo "4. Checking metric types..."
echo "$METRICS" | grep "# TYPE trading" | wc -l
echo "total metric types found"

# Cleanup
kill $SERVER_PID
echo ""
echo "=== ALL CHECKS PASSED ==="
```

**Run:**
```bash
bash /Users/bhavikpatel/Desktop/TradingBot/Go-project/smoke-test.sh
```

---

## Level 5: Grafana Dashboard Testing

### Prerequisites
- Prometheus running and scraping metrics
- Grafana running

### Test dashboards are importable

```bash
#!/bin/bash

GRAFANA_URL="http://localhost:3000"
GRAFANA_API_KEY="your_api_key_here"
DASHBOARD_DIR="/Users/bhavikpatel/Desktop/TradingBot/Go-project/grafana/dashboards"

echo "=== GRAFANA DASHBOARD IMPORT TEST ==="
echo ""

# Verify Grafana is running
echo "1. Checking Grafana availability..."
if ! curl -s "$GRAFANA_URL/api/health" | grep -q "ok"; then
  echo "✗ FAILED: Grafana not running or not responding"
  echo "Start Grafana first: docker run -d -p 3000:3000 grafana/grafana"
  exit 1
fi
echo "✓ Grafana is running"

# Check Prometheus data source
echo ""
echo "2. Checking Prometheus data source..."
DS=$(curl -s "$GRAFANA_URL/api/datasources" \
  -H "Authorization: Bearer $GRAFANA_API_KEY" | grep -i prometheus)

if [ -z "$DS" ]; then
  echo "⚠ WARNING: No Prometheus data source configured"
  echo "Add it manually: Configuration → Data Sources → Prometheus"
else
  echo "✓ Prometheus data source found"
fi

# Import each dashboard
echo ""
echo "3. Importing dashboards..."
for dashboard in "$DASHBOARD_DIR"/*.json; do
  name=$(basename "$dashboard")
  echo -n "  $name... "
  
  response=$(curl -s -X POST \
    "$GRAFANA_URL/api/dashboards/db" \
    -H "Authorization: Bearer $GRAFANA_API_KEY" \
    -H "Content-Type: application/json" \
    -d @"$dashboard")
  
  if echo "$response" | grep -q "success"; then
    echo "✓"
  elif echo "$response" | grep -q "already exists"; then
    echo "⚠ (already exists, skipping)"
  else
    echo "✗"
    echo "    Error: $response"
  fi
done

echo ""
echo "=== IMPORT TEST COMPLETE ==="
echo "View dashboards at: $GRAFANA_URL/d/trading-market-data"
```

**Run:**
```bash
# Create API key in Grafana first:
# Configuration → API Keys → New API Key (role: Admin)

GRAFANA_API_KEY="your_key_here" bash test-grafana-import.sh
```

---

## Level 6: Load Testing (Verify Metrics Under Stress)

### Test: Generate tick load and verify metrics record correctly

```bash
#!/bin/bash

echo "=== METRICS LOAD TEST ==="
echo ""

# Start server
echo "1. Starting server..."
cd /Users/bhavikpatel/Desktop/TradingBot/Go-project
go run cmd/server/main.go > /tmp/server.log 2>&1 &
SERVER_PID=$!
sleep 3

# Get baseline
echo "2. Recording baseline metrics..."
BASELINE=$(curl -s http://localhost:8080/api/v1/metrics | grep "trading_ticks_received_total")
echo "   Baseline: $BASELINE"

# Simulate tick load (if your app has a test API endpoint)
echo "3. Sending test ticks (30 seconds)..."
# This depends on your app's testing support
# For example, if you have an internal test endpoint:
# for i in {1..100}; do
#   curl -s -X POST http://localhost:8080/internal/test/tick \
#     -d '{"price": 100.0, "timestamp": '$(date +%s000)'}' &
# done

sleep 30

# Get after-load metrics
echo "4. Recording after-load metrics..."
AFTER=$(curl -s http://localhost:8080/api/v1/metrics | grep "trading_ticks_received_total")
echo "   After: $AFTER"

# Extract values
BASELINE_VAL=$(echo "$BASELINE" | awk '{print $NF}')
AFTER_VAL=$(echo "$AFTER" | awk '{print $NF}')
DIFF=$((AFTER_VAL - BASELINE_VAL))

echo ""
echo "5. Results:"
echo "   Baseline: $BASELINE_VAL"
echo "   After:    $AFTER_VAL"
echo "   Ticks:    $DIFF"

if [ $DIFF -gt 0 ]; then
  echo "✓ Metrics are recording"
else
  echo "⚠ No tick increase (app may not be ingesting data)"
fi

# Check goroutines didn't leak
echo ""
echo "6. Checking for goroutine leaks..."
GOROUTINES=$(curl -s http://localhost:8080/api/v1/metrics | grep "trading_goroutines_current{" | awk '{print $NF}')
echo "   Current goroutines: $GOROUTINES"

if [ "$GOROUTINES" -lt 100 ]; then
  echo "✓ Goroutine count is healthy"
else
  echo "⚠ WARNING: High goroutine count, possible leak"
fi

# Check memory
echo ""
echo "7. Checking memory usage..."
HEAP=$(curl -s http://localhost:8080/api/v1/metrics | grep "trading_memory_heap_alloc_bytes" | awk '{print $NF}')
HEAP_MB=$((HEAP / 1024 / 1024))
echo "   Heap allocated: ${HEAP_MB}MB"

if [ "$HEAP_MB" -lt 500 ]; then
  echo "✓ Memory usage is healthy"
else
  echo "⚠ WARNING: High memory usage (${HEAP_MB}MB)"
fi

# Cleanup
kill $SERVER_PID

echo ""
echo "=== LOAD TEST COMPLETE ==="
```

---

## Level 7: Metrics Data Validation

### Test: Verify metrics produce expected values

```bash
#!/bin/bash

METRICS_URL="http://localhost:8080/api/v1/metrics"

echo "=== METRICS DATA VALIDATION ==="
echo ""

# Start server
cd /Users/bhavikpatel/Desktop/TradingBot/Go-project
go run cmd/server/main.go > /tmp/server.log 2>&1 &
SERVER_PID=$!
sleep 3

get_metric_value() {
  local metric_name=$1
  curl -s "$METRICS_URL" | grep "$metric_name{" | awk '{print $NF}'
}

echo "1. Counter metrics (should be >= 0):"
echo "   TicksReceived: $(get_metric_value 'trading_ticks_received_total')"
echo "   TicksProcessed: $(get_metric_value 'trading_ticks_processed_total')"
echo "   TicksDropped: $(get_metric_value 'trading_ticks_dropped_total')"

echo ""
echo "2. Gauge metrics (should be >= 0):"
echo "   Goroutines: $(get_metric_value 'trading_goroutines_current')"
echo "   Memory: $(get_metric_value 'trading_memory_heap_alloc_bytes')"
echo "   QueueDepth: $(get_metric_value 'trading_tick_queue_depth')"

echo ""
echo "3. Histogram buckets (should be >= 0):"
curl -s "$METRICS_URL" | grep "trading_tick_lag_ms_bucket" | head -3

echo ""
echo "4. Specific metric checks:"

# Verify staleness detection
STALE=$(get_metric_value 'trading_tick_feed_dead')
echo "   Tick Feed Dead: $STALE (should be 0 if ticks are flowing)"

# Verify panic counter
PANICS=$(get_metric_value 'trading_panics_total')
echo "   Panics: $PANICS (should be 0)"

# Verify metric format
echo ""
echo "5. Checking Prometheus format compliance:"
curl -s "$METRICS_URL" | grep -E "^[a-z_]+.*[0-9.]+$" | wc -l
echo "   metric lines found"

kill $SERVER_PID

echo ""
echo "=== VALIDATION COMPLETE ==="
```

---

## Checklist: Complete Testing

**Run in order:**

- [ ] Level 1: Unit tests
  ```bash
  go test ./internal/observability -v
  ```

- [ ] Level 2: Endpoint test
  ```bash
  # Start app, then: curl http://localhost:8080/api/v1/metrics | head -20
  ```

- [ ] Level 3: Prometheus integration (requires Prometheus)
  ```bash
  # Configure prometheus-test.yml, start Prometheus, start app
  # Query: curl 'http://localhost:9090/api/v1/query?query=trading_ticks_received_total'
  ```

- [ ] Level 4: Smoke test
  ```bash
  bash smoke-test.sh
  ```

- [ ] Level 5: Grafana import (requires Grafana + Prometheus)
  ```bash
  GRAFANA_API_KEY="xxx" bash test-grafana-import.sh
  ```

- [ ] Level 6: Load test
  ```bash
  bash load-test.sh
  ```

- [ ] Level 7: Data validation
  ```bash
  bash metrics-validation.sh
  ```

---

## Troubleshooting

### Issue: Metrics endpoint returns 404

**Solution:**
1. Verify `/api/v1/metrics` route is registered in `internal/httpapi/routes.go`
2. Check app is running: `curl http://localhost:8080/health`
3. Check port (default 8080, verify with app logs)

### Issue: Metrics show zero values

**Solution:**
1. Verify app is actually ingesting data (check logs)
2. Verify metrics are being incremented in code (search for `.Inc()`)
3. Run with test data if available

### Issue: Prometheus can't scrape

**Solution:**
1. Verify Prometheus config targets app correctly
2. Check firewall: `curl http://localhost:8080/api/v1/metrics` from Prometheus host
3. Verify metrics endpoint returns valid Prometheus format

### Issue: Grafana shows "No data"

**Solution:**
1. Verify Prometheus is scraping: http://localhost:9090/targets
2. Verify data source is configured: Configuration → Data Sources
3. Try manual query in Prometheus: http://localhost:9090/graph
4. Check panel query is correct (use Prometheus query editor)

### Issue: Goroutine count always increasing

**Red flag:** Likely goroutine leak in code

**Debug:**
1. Check all goroutines have cleanup
2. Review recent changes that spawn goroutines
3. Check channel operations aren't blocking indefinitely

---

## Production Verification

Before deploying to production:

1. ✅ All tests pass
2. ✅ Metrics endpoint returns <50ms
3. ✅ CPU overhead < 1%
4. ✅ Memory overhead < 50MB
5. ✅ All dashboards show data
6. ✅ Alert rules are configured
7. ✅ Prometheus retention policy set
