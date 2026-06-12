# Grafana Dashboards Setup

## How to Create the Dashboards

### Option 1: Import JSON Dashboards
1. Open Grafana UI
2. Click **+** (Create) → **Import dashboard**
3. Paste the JSON content from the files in `dashboards/` directory
4. Select your Prometheus data source
5. Click **Import**

### Option 2: Manual Setup
Create the dashboards using the Grafana UI by following the metric names and panel configurations below.

---

## Dashboard 1: Market Data
**Purpose:** Monitor tick flow, latency, and data quality

### Panels:
1. **Ticks/sec (Rate)**
   - Metric: `rate(trading_ticks_received_total[1m])`
   - Type: Graph
   - Unit: ops

2. **Tick Lag (Histogram)**
   - Metric: `trading_tick_lag_ms`
   - Type: Heatmap or Graph
   - Unit: ms
   - Alert if p99 > 500ms

3. **Tick Loss Rate**
   - Metric: `rate(trading_ticks_dropped_total[1m]) / (rate(trading_ticks_received_total[1m]) + 0.001)`
   - Type: Graph
   - Unit: percent

4. **Feed Dead Alert**
   - Metric: `trading_tick_feed_dead`
   - Type: Stat (red if 1)
   - Unit: bool
   - Alert: Critical if feed dead > 5s

5. **Last Tick Age**
   - Metric: `time() - trading_last_tick_timestamp_seconds`
   - Type: Gauge
   - Unit: s
   - Threshold: Warning at 3s, Critical at 5s

---

## Dashboard 2: Queues
**Purpose:** Monitor backpressure and processing bottlenecks

### Panels:
1. **Broadcast Queue Depth**
   - Metric: `trading_tick_queue_depth`
   - Type: Graph
   - Alert if > 1000 (threshold)

2. **DB Flush Queue Depth**
   - Metric: `trading_db_flush_queue_depth`
   - Type: Graph
   - Alert if > 100

3. **Candle Queue Depth**
   - Metric: `trading_candle_queue_depth`
   - Type: Graph

4. **Indicator Queue Depth**
   - Metric: `trading_indicator_queue_depth`
   - Type: Graph

5. **Queue Depth Trend**
   - Metric: `max(trading_tick_queue_depth, trading_db_flush_queue_depth, trading_candle_queue_depth)`
   - Type: Area

---

## Dashboard 3: Candles
**Purpose:** Monitor candle generation and finalization performance

### Panels:
1. **Open Candles Count**
   - Metric: `trading_open_candles_count`
   - Type: Graph
   - Unit: short

2. **Candle Finalization Latency**
   - Metric: `trading_candle_latency_ms`
   - Type: Heatmap
   - Unit: ms

3. **Candles Finalized (Rate)**
   - Metric: `rate(trading_candles_finalized_total[1m])`
   - Type: Graph
   - Unit: ops

4. **Finalization Latency Percentiles**
   - P50: `histogram_quantile(0.5, rate(trading_candle_latency_ms_bucket[5m]))`
   - P95: `histogram_quantile(0.95, rate(trading_candle_latency_ms_bucket[5m]))`
   - P99: `histogram_quantile(0.99, rate(trading_candle_latency_ms_bucket[5m]))`
   - Type: Graph

---

## Dashboard 4: Indicators
**Purpose:** Monitor indicator computation performance

### Panels:
1. **Indicator Latency (Heatmap)**
   - Metric: `trading_indicator_latency_ms`
   - Type: Heatmap
   - Unit: ms

2. **Indicator Errors (Rate)**
   - Metric: `rate(trading_indicator_errors_total[1m])`
   - Type: Graph
   - Unit: ops
   - Alert if rate > 0

3. **Indicator Queue Depth**
   - Metric: `trading_indicator_queue_depth`
   - Type: Graph

4. **Latency P99**
   - Metric: `histogram_quantile(0.99, rate(trading_indicator_latency_ms_bucket[5m]))`
   - Type: Stat
   - Unit: ms

---

## Dashboard 5: Runtime
**Purpose:** Monitor system resource usage and health

### Panels:
1. **Goroutines**
   - Metric: `trading_goroutines_current`
   - Type: Graph
   - Alert if > 5000 (potential leak)

2. **Memory Heap (Bytes)**
   - Metric: `trading_memory_heap_alloc_bytes / 1024 / 1024`
   - Type: Graph
   - Unit: MB
   - Alert if > 2GB (adjust based on deployment)

3. **Memory System (Bytes)**
   - Metric: `trading_memory_sys_bytes / 1024 / 1024`
   - Type: Graph
   - Unit: MB

4. **GC Runs**
   - Metric: `rate(trading_gc_runs_total[5m])`
   - Type: Graph
   - Unit: ops/5m

5. **Panic Count**
   - Metric: `rate(trading_panics_total[1m])`
   - Type: Stat
   - Alert if > 0

6. **Heap In Use**
   - Metric: `trading_memory_heap_inuse_bytes / 1024 / 1024`
   - Type: Gauge
   - Unit: MB

---

## Dashboard 6: Recovery
**Purpose:** Monitor WAL and replay operations (for future implementation)

### Panels:
1. **Placeholder: Replay Duration**
   - Metric: TBD
   - Type: Graph
   - Unit: s

2. **Placeholder: Replay Lag**
   - Metric: TBD
   - Type: Gauge
   - Unit: ms

3. **Placeholder: Recovered Ticks**
   - Metric: TBD
   - Type: Counter
   - Unit: short

---

## Alert Rules

### Critical Alerts:
1. **Feed Dead**: `trading_tick_feed_dead > 0 for 30s`
   - Action: Page on-call engineer
2. **High Panic Rate**: `rate(trading_panics_total[1m]) > 0`
   - Action: Page on-call engineer
3. **Memory Critical**: `trading_memory_heap_alloc_bytes / 1024 / 1024 > 2000`
   - Action: Investigate OOM risk

### Warning Alerts:
1. **High Queue Depth**: `trading_tick_queue_depth > 1000`
   - Action: Investigate processing bottleneck
2. **Tick Lag High**: `histogram_quantile(0.99, rate(trading_tick_lag_ms_bucket[5m])) > 500`
   - Action: Investigate network/processing delays
3. **Goroutine Leak**: `trading_goroutines_current > 5000`
   - Action: Investigate for goroutine leaks

---

## Testing the Metrics

### Via Prometheus UI:
Navigate to `http://localhost:9090` and query:
- `trading_ticks_received_total` - Should be increasing in live mode
- `trading_tick_lag_ms` - Should have values in range 1-100ms typically
- `trading_goroutines_current` - Should be stable

### Via Grafana:
- Navigate to `http://localhost:3000`
- Query dashboards after importing
- Check for data in each panel

---

## Configuration

### Prometheus Config (`prometheus.yml`):
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'trading-bot'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/api/v1/metrics'
    scrape_interval: 5s
    scrape_timeout: 5s
```

### Grafana Data Source:
- Name: Prometheus
- URL: http://prometheus:9090
- Access: Server

---

## Troubleshooting

### No metrics appearing?
1. Check `/api/v1/metrics` endpoint directly in browser
2. Verify Prometheus is scraping: `http://prometheus:9090/targets`
3. Check logs for `InitMetrics` error messages

### High latency spikes?
1. Check queue depth panels
2. Check memory usage
3. Monitor tick lag histogram

### Goroutine leak?
1. Check goroutine count trend
2. Look for panics in panic counter
3. Check runtime logs for errors
