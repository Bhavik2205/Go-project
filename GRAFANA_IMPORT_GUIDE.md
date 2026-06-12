# Grafana Dashboard Import Guide

## Quick Start

All 6 dashboards are ready to import into Grafana. They are located in `grafana/dashboards/`.

### Dashboard Files

| # | Dashboard | File | Focus Area |
|---|-----------|------|-----------|
| 1 | Market Data | `01-market-data.json` | Tick flow, lag, loss, feed health |
| 2 | Queues | `02-queues.json` | Queue depths, backpressure |
| 3 | Candles | `03-candles.json` | Candle finalization latency |
| 4 | Indicators | `04-indicators.json` | Indicator latency, errors, queue depth |
| 5 | Runtime | `05-runtime.json` | Goroutines, memory, GC, panics |
| 6 | Recovery | `06-recovery.json` | Placeholder for future WAL/replay metrics |

---

## Step-by-Step Import Instructions

### Prerequisites

1. **Prometheus configured as data source** in Grafana
   - Go to: Configuration → Data Sources
   - Add Prometheus with URL `http://localhost:9090` (or your Prometheus server)
   - Test connection & save
   - **Make note of the data source name** (default: "Prometheus")

2. **TradingBot running** with `/api/v1/metrics` endpoint exposed
   - Verify: `curl http://localhost:8080/api/v1/metrics`
   - Should return Prometheus metrics (text/plain)

3. **Prometheus scraping TradingBot**
   - Add scrape config to `prometheus.yml`:
     ```yaml
     global:
       scrape_interval: 5s
     scrape_configs:
       - job_name: 'tradingbot'
         static_configs:
           - targets: ['localhost:8080']
         metrics_path: '/api/v1/metrics'
     ```

### Import via UI (Recommended)

1. **Open Grafana home page**
   - Navigate to: http://localhost:3000

2. **Import dashboard**
   - Click: **+** (plus icon) in left sidebar → **Import**
   - Or: Dashboards → New → Import

3. **Load JSON file**
   - Click: "Upload JSON file"
   - Select: `grafana/dashboards/01-market-data.json`
   - Review import dialog
   - **Select Prometheus data source** from dropdown
   - Click: **Import**

4. **Repeat for all 6 dashboards**
   - `02-queues.json`
   - `03-candles.json`
   - `04-indicators.json`
   - `05-runtime.json`
   - `06-recovery.json`

### Import via API (Automated)

**Script to import all dashboards:**

```bash
#!/bin/bash

GRAFANA_URL="http://localhost:3000"
GRAFANA_API_KEY="your_api_key_here"  # Create in Grafana: Configuration → API Keys
DASHBOARD_DIR="grafana/dashboards"

for dashboard in "$DASHBOARD_DIR"/*.json; do
  echo "Importing $(basename $dashboard)..."
  
  curl -X POST \
    "$GRAFANA_URL/api/dashboards/db" \
    -H "Authorization: Bearer $GRAFANA_API_KEY" \
    -H "Content-Type: application/json" \
    -d @"$dashboard"
    
  echo "✓ $(basename $dashboard) imported"
done
```

**Steps:**

1. Create API key in Grafana:
   - Configuration → API Keys → New API Key
   - Role: Admin (for imports)
   - Copy token

2. Save script as `import-dashboards.sh`

3. Run:
   ```bash
   chmod +x import-dashboards.sh
   GRAFANA_API_KEY="your_token" ./import-dashboards.sh
   ```

---

## Dashboard Descriptions

### Dashboard 1: Market Data

**Purpose:** Monitor tick ingestion and feed health

**Key Panels:**
- **Ticks/sec** - Throughput indicator
- **Tick Lag** - Time from Zerodha timestamp to processing (p50/p95/p99)
- **Tick Loss** - % of dropped ticks
- **Tick Feed Dead** - Alert (1 = no ticks for >5s)

**Alerts to Set:**
- `Tick Lag p99 > 500ms` → Warning
- `Tick Lag p99 > 1000ms` → Critical
- `Tick Feed Dead == 1` → Critical

### Dashboard 2: Queues

**Purpose:** Detect backpressure and processing bottlenecks

**Key Panels:**
- **DB Flush Queue Depth** - Database write queue
- **Broadcast Queue Depth** - WebSocket message queue
- **Replay Queue Depth** (future) - WAL replay queue

**Interpretation:**
- Sustained growth → Bottleneck detected
- Spikes → Temporary burst (OK if short-lived)
- Consistently empty → Healthy flow

### Dashboard 3: Candles

**Purpose:** Monitor candle aggregation and finalization

**Key Panels:**
- **Open Candles** - Count of in-flight candle aggregations
- **Finalize Latency** - Time from candle close to finalization
- **Late Ticks** (future) - Ticks arriving after candle closure

**Alerts to Set:**
- `Finalize Latency p95 > 100ms` → Investigate
- `Open Candles > 10,000` → Bottleneck warning

### Dashboard 4: Indicators

**Purpose:** Monitor indicator computation performance

**Key Panels:**
- **Indicator Latency** - Time to compute all indicators (p50/p95/p99)
- **Error Rate** - Errors per second
- **Queue Depth** - Pending indicator computations

**Alerts to Set:**
- `Error Rate > 0` → Warning (investigate error type)
- `Latency p99 > 500ms` → Investigate (indicators blocking tick flow)

### Dashboard 5: Runtime

**Purpose:** Monitor process health and resource usage

**Key Panels:**
- **Goroutine Count** - Active goroutines (detect leaks)
- **Memory Usage** - Heap allocation, in-use, system (MB)
- **GC Frequency** - Garbage collection runs per 5min
- **Panic Rate** - **Critical alert if > 0**

**Alerts to Set:**
- `Goroutine Count > 5000` → Warning (possible leak)
- `Goroutine Count > 10000` → Critical
- `Memory Heap > 500MB` → Warning
- `Panic Rate > 0` → Critical

### Dashboard 6: Recovery (Placeholder)

**Purpose:** Future dashboard for WAL & replay observability

**To be implemented when WAL feature is added:**
- Replay duration
- Replay lag
- Recovered ticks
- Recovery status

---

## Verification Steps

After importing all dashboards:

1. **Check data appears**
   - Navigate to each dashboard
   - Verify panels show data (not "No data")
   - If no data: verify Prometheus is scraping (`http://localhost:9090/targets`)

2. **Test metrics endpoint**
   ```bash
   curl -s http://localhost:8080/api/v1/metrics | head -20
   ```
   Should show lines like:
   ```
   # HELP trading_ticks_received_total Total ticks received from Zerodha
   # TYPE trading_ticks_received_total counter
   trading_ticks_received_total 12345
   ```

3. **Verify tick lag is reasonable**
   - Should be < 100ms for modern internet
   - > 500ms indicates network/processing issues

4. **Monitor goroutines**
   - Should be stable (not growing over hours)
   - Spike indicates goroutine leak

---

## Troubleshooting

### Issue: "No data" or "Datasource not found"

**Solution:**
1. Verify data source exists in Grafana
2. Click dashboard panel → Edit → Query inspector
3. Check data source dropdown shows "Prometheus"
4. Verify Prometheus scrape targets are healthy

### Issue: Tick Lag is very high (>1000ms)

**Possible causes:**
1. Network latency between Zerodha and server
2. Processing bottleneck (queue depth growing)
3. Zerodha feed is slow (market hours issues)

**Debug:**
- Check Queues dashboard for backpressure
- Check Runtime dashboard for goroutine/memory issues
- Check Prometheus itself is not overloaded

### Issue: Goroutine count constantly increasing

**Cause:** Goroutine leak

**Debug:**
1. Check which component is leaking (code review)
2. Look for missing `context.Done()` channels or cancel cleanup
3. Check for blocked sends on channels without timeout

### Issue: Panic Rate > 0

**Immediate action:** CRITICAL

1. Check application logs for panic stack trace
2. Identify component (metric labels if available)
3. Fix and redeploy immediately
4. Review all modified code for bugs

---

## Setting Up Alerts (Optional)

**Grafana Alerting (Recommended):**

For each critical metric, create an alert rule:

1. Click gear icon → Alerting → Alert Rules
2. New alert rule:
   - **Name:** `High Tick Lag`
   - **Condition:** `histogram_quantile(0.99, rate(trading_tick_lag_ms_bucket[5m])) > 1000`
   - **Evaluate every:** 10s
   - **For:** 1m (wait 1 min before alerting)
   - **Notification channel:** Slack/Email/etc.

**Critical alerts to set:**
- Tick Lag p99 > 1s
- Tick Feed Dead == 1
- Panic Rate > 0
- Goroutines > 10,000
- Memory > 1GB

**Prometheus Alerting (Alternative):**

Create `prometheus-alerts.yaml`:
```yaml
groups:
  - name: trading_bot
    rules:
      - alert: HighTickLag
        expr: histogram_quantile(0.99, rate(trading_tick_lag_ms_bucket[5m])) > 1000
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High tick lag detected (p99 > 1000ms)"
```

---

## Production Recommendations

1. **Retention:** Keep metrics for at least 30 days
   - Set Prometheus `retention` in `prometheus.yml`:
   ```yaml
   global:
     scrape_interval: 5s
   storage:
     tsdb:
       retention: 30d
   ```

2. **Backup:** Export dashboards regularly
   ```bash
   curl -s http://localhost:3000/api/dashboards/uid/trading-market-data \
     -H "Authorization: Bearer $API_KEY" | jq > market-data-backup.json
   ```

3. **SLOs:** Establish Service Level Objectives
   - Tick Lag p99 < 100ms
   - Feed uptime > 99.9%
   - Panic rate = 0

4. **On-call:** Configure escalation policies for critical alerts

---

## Related Documentation

- **Implementation Details:** See `OBSERVABILITY_IMPLEMENTATION.md`
- **Metrics Reference:** See `grafana/README.md`
- **Grafana Docs:** https://grafana.com/docs/
- **Prometheus Docs:** https://prometheus.io/docs/
