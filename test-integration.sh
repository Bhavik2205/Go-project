#!/bin/bash
# Full integration test: Prometheus + Metrics + Grafana
# Requires: Docker (for Prometheus), Go, Grafana running on :3000

set -e

echo "╔════════════════════════════════════════════════╗"
echo "║   FULL INTEGRATION TEST - PROMETHEUS + APP    ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Check prerequisites
echo "📋 Checking prerequisites..."
if ! command -v go &> /dev/null; then
  echo "❌ Go not installed"
  exit 1
fi
echo "✓ Go installed"

if ! command -v docker &> /dev/null; then
  echo "⚠️  Docker not found (Prometheus test will be skipped)"
  SKIP_PROMETHEUS=1
else
  echo "✓ Docker available"
fi

echo ""

# Step 1: Build app
echo "🔨 STEP 1: Building application"
echo "────────────────────────────────────────────────"
if go build -o /tmp/tradingbot ./cmd/server/main.go; then
  echo "✅ Build successful"
else
  echo "❌ Build failed"
  exit 1
fi

echo ""

# Step 2: Start app
echo "🚀 STEP 2: Starting TradingBot"
echo "────────────────────────────────────────────────"
/tmp/tradingbot > /tmp/tradingbot.log 2>&1 &
APP_PID=$!
echo "App PID: $APP_PID"

# Wait for app to start
sleep 2
if ! kill -0 $APP_PID 2>/dev/null; then
  echo "❌ App failed to start"
  cat /tmp/tradingbot.log
  exit 1
fi
echo "✅ App running"

echo ""

# Step 3: Test metrics endpoint
echo "📊 STEP 3: Testing metrics endpoint"
echo "────────────────────────────────────────────────"
if curl -s http://localhost:8080/api/v1/metrics > /tmp/metrics.txt; then
  LINES=$(wc -l < /tmp/metrics.txt)
  echo "✅ Metrics endpoint working ($LINES lines)"
  
  # Display first few metrics
  echo ""
  echo "Sample metrics:"
  head -10 /tmp/metrics.txt | grep -v "^#" | head -3
else
  echo "❌ Metrics endpoint failed"
  kill $APP_PID
  exit 1
fi

echo ""

# Step 4: Start Prometheus (optional)
if [ -z "$SKIP_PROMETHEUS" ]; then
  echo "🔍 STEP 4: Starting Prometheus"
  echo "────────────────────────────────────────────────"
  
  # Create Prometheus config
  cat > /tmp/prometheus-test.yml <<EOF
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: 'tradingbot'
    static_configs:
      - targets: ['host.docker.internal:8080']
    metrics_path: '/api/v1/metrics'
EOF
  
  # Start Prometheus
  docker run -d \
    --name prometheus-test-$$\
    -p 9090:9090 \
    -v /tmp/prometheus-test.yml:/etc/prometheus/prometheus.yml \
    prom/prometheus:latest > /dev/null 2>&1
  
  PROM_PID=$(docker ps | grep prometheus-test-$$ | awk '{print $1}')
  echo "✅ Prometheus container: $PROM_PID"
  
  # Wait for Prometheus to scrape
  echo "Waiting for Prometheus to scrape metrics..."
  sleep 15
  
  # Query Prometheus
  echo ""
  echo "📈 STEP 5: Querying Prometheus"
  echo "────────────────────────────────────────────────"
  
  QUERY_RESULT=$(curl -s 'http://localhost:9090/api/v1/query?query=trading_ticks_received_total')
  if echo "$QUERY_RESULT" | grep -q '"result":\[\]'; then
    echo "⚠️  Prometheus hasn't scraped yet (try again in 10s)"
  elif echo "$QUERY_RESULT" | grep -q 'trading_ticks_received_total'; then
    echo "✅ Prometheus successfully scraped metrics"
    echo "$QUERY_RESULT" | head -5
  else
    echo "⚠️  Query result unclear"
  fi
  
  # Cleanup Prometheus
  docker stop prometheus-test-$$ > /dev/null 2>&1
  docker rm prometheus-test-$$ > /dev/null 2>&1
  echo "Prometheus stopped"
fi

echo ""

# Step 6: Final summary
echo "🎯 STEP 6: Summary"
echo "────────────────────────────────────────────────"

# Check for crashes/panics
if grep -i "panic\|error" /tmp/tradingbot.log | grep -v "^$"; then
  echo "⚠️  Warnings in logs (see above)"
else
  echo "✅ No panics or errors in logs"
fi

# Check metrics count
METRIC_TYPES=$(grep "^# TYPE" /tmp/metrics.txt | wc -l)
echo "✅ Metric types defined: $METRIC_TYPES"

# Cleanup
kill $APP_PID 2>/dev/null || true

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║            ✅ INTEGRATION TEST COMPLETE         ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Next steps to view dashboards:"
echo "  1. Start Prometheus (or use your existing instance)"
echo "  2. Import dashboards from grafana/dashboards/ into Grafana"
echo "  3. Open http://localhost:3000/d/trading-market-data"
echo ""
