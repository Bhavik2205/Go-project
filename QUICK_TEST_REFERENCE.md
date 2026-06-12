# Testing Observability - Quick Reference

## TL;DR - Run These Tests

### 1. Quick Test (1 minute) ✨
```bash
cd /Users/bhavikpatel/Desktop/TradingBot/Go-project
bash test-quick.sh
```
**Verifies:** Metrics endpoint is working and returning data

### 2. Full Integration Test (2 minutes)
```bash
bash test-integration.sh
```
**Verifies:** Prometheus can scrape metrics from app

### 3. View Dashboards (Once Prometheus is running)
1. Import dashboards: Follow `GRAFANA_IMPORT_GUIDE.md`
2. Open: http://localhost:3000/d/trading-market-data

---

## What Gets Tested

| Test | Checks | Command |
|------|--------|---------|
| **Quick** | ✓ App builds<br>✓ Metrics endpoint responds<br>✓ All 23 metrics present | `bash test-quick.sh` |
| **Integration** | ✓ Quick tests<br>✓ Prometheus scrapes data<br>✓ No crashes in logs | `bash test-integration.sh` |
| **Unit Tests** | ✓ Metrics register correctly<br>✓ Counters/gauges/histograms work | `go test ./internal/observability -v` |
| **Grafana** | ✓ Dashboards import<br>✓ Panels show data | Manual (see import guide) |

---

## Expected Output

### test-quick.sh ✅
```
✅ Build successful
✅ App running
✅ Endpoint responds (85 lines)
✓ trading_ticks_received_total
✓ trading_tick_lag_ms_bucket
✓ trading_goroutines_current
✓ trading_panics_total
✅ QUICK TEST COMPLETE
```

### test-integration.sh ✅
```
✓ Go installed
✓ Docker available
✅ Build successful
✅ App running
✅ Metrics endpoint working (85 lines)
✅ Prometheus container started
✅ Prometheus successfully scraped metrics
✅ No panics or errors in logs
✅ Metric types defined: 23
✅ INTEGRATION TEST COMPLETE
```

---

## 🚨 Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| `Connection refused` on `localhost:8080` | Make sure app is running: `ps aux \| grep tradingbot` |
| `curl: (7) Failed to connect` | App port may be different (check logs) or port 8080 in use |
| Metrics show `0` values | Normal - metrics start at 0, increase as app runs and processes ticks |
| `Prometheus isn't scraping` | Wait 5-10s after Prometheus starts, or check `http://localhost:9090/targets` |
| `No data in Grafana` | Verify Prometheus data source is working: `http://localhost:9090/graph` |

---

## Full Test Flow (Complete)

```
1. Run quick test (verify endpoint)
   └─> bash test-quick.sh
   
2. Run integration test (verify Prometheus)
   └─> bash test-integration.sh
   
3. Start Prometheus (use existing or Docker)
   └─> prometheus --config.file=prometheus.yml
   
4. Start Grafana
   └─> docker run -p 3000:3000 grafana/grafana
   
5. Add Prometheus data source in Grafana
   └─> Configuration → Data Sources → Add Prometheus
   
6. Import dashboards
   └─> +Import → Upload JSON
   └─> Import all 6 files from grafana/dashboards/
   
7. View dashboards
   └─> http://localhost:3000/d/trading-market-data
   └─> http://localhost:3000/d/trading-queues
   └─> (etc...)
```

---

## Metrics to Check

After running tests, these metrics should be present:

```
Counters (total):
  trading_ticks_received_total ✓
  trading_ticks_processed_total ✓
  trading_ticks_dropped_total ✓
  trading_indicator_errors_total ✓
  trading_panics_total ✓

Histograms (milliseconds):
  trading_tick_lag_ms_bucket ✓
  trading_candle_latency_ms_bucket ✓
  trading_indicator_latency_ms_bucket ✓

Gauges (current value):
  trading_goroutines_current ✓
  trading_memory_heap_alloc_bytes ✓
  trading_memory_heap_inuse_bytes ✓
  trading_memory_sys_bytes ✓
  trading_gc_runs_total ✓
  trading_tick_queue_depth ✓
  trading_db_flush_queue_depth ✓
  trading_indicator_queue_depth ✓
  trading_tick_feed_dead ✓
```

---

## Next: Configure Alerts (Optional but Recommended)

Once dashboards are working, set up alerts:

**Critical Alerts:**
- Panic rate > 0 (immediate action required!)
- Tick lag p99 > 1000ms
- Tick feed dead == 1
- Goroutines > 10,000

**See:** `GRAFANA_IMPORT_GUIDE.md` section "Setting Up Alerts"

---

## Files Reference

| File | Purpose |
|------|---------|
| `test-quick.sh` | Fast endpoint test (1 min) |
| `test-integration.sh` | Full test with Prometheus (2 min) |
| `TESTING_OBSERVABILITY.md` | Comprehensive testing guide (7 levels) |
| `GRAFANA_IMPORT_GUIDE.md` | Dashboard import + setup instructions |
| `OBSERVABILITY_IMPLEMENTATION.md` | Technical architecture details |
| `grafana/README.md` | Metric descriptions + alert rules |
| `grafana/dashboards/*.json` | 6 ready-to-import dashboards |

---

## Performance Expectations

After running with real tick data:

| Metric | Expected | Good Range |
|--------|----------|------------|
| Tick Lag p50 | <50ms | <100ms |
| Tick Lag p99 | <200ms | <500ms |
| Tick Loss % | <0.1% | <1% |
| Goroutines | 20-50 | <1000 |
| Memory (Heap) | 50-200MB | <500MB |
| Panic Rate | 0/sec | 0/sec (always) |

---

## Success Criteria ✅

Test is **PASSED** when:

- [ ] `test-quick.sh` shows "✅ QUICK TEST COMPLETE"
- [ ] `test-integration.sh` shows "✅ INTEGRATION TEST COMPLETE"
- [ ] Metrics endpoint returns all 23 metrics
- [ ] Prometheus scrapes metrics successfully
- [ ] All 6 Grafana dashboards import without errors
- [ ] Dashboards show data (not "No data")
- [ ] No panics in app logs

---

## Still Need Help?

1. **Metrics not appearing?** → Check `TESTING_OBSERVABILITY.md` Level 4 (Metrics missing)
2. **Prometheus not scraping?** → Check `GRAFANA_IMPORT_GUIDE.md` (Troubleshooting)
3. **Grafana shows no data?** → Check `GRAFANA_IMPORT_GUIDE.md` (No data error)
4. **App crashes?** → Check app logs: `tail -f /tmp/tradingbot.log`

---

## Time Estimates

| Activity | Time | Prerequisites |
|----------|------|---|
| Quick test | 1 min | Go, app running |
| Integration test | 2 min | Go, Docker, app running |
| Import Grafana dashboards | 5 min | Grafana running, Prometheus configured |
| Verify dashboards show data | 2 min | Previous steps complete |
| **Total end-to-end** | **15 min** | All tools available |

**Once dashboards are working, you can move on to:** WAL (Write-Ahead Log) implementation for crash recovery.
