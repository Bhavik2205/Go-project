# FULL CODEBASE AUDIT — ML-Bot Trading Platform
**Principal Engineer / Trading Systems Architect Review**
**Date:** 2026-06-09
**Codebase:** `github.com/Bhavik2205/ML-Bot` — Go-based NSE algorithmic trading platform

---

## EXECUTIVE SUMMARY

The system has a **solid structural foundation** but is **not production-ready** and contains multiple **P0/critical bugs that directly explain the simulation crashes**. The TickBus migration is architecturally correct but introduces a **fatal deadlock path** under simulation load. Candle generation contains **volume calculation bugs** and **simulation-mode bucket misalignment**. The most urgent security risk is **live Zerodha credentials committed in `.env` and `.access_token` files tracked in the repo** — these must be rotated immediately.

| Area | Score | Status |
|------|-------|--------|
| Architecture | 6.5/10 | Solid skeleton, needs decoupling |
| TickBus | 5/10 | Deadlock risk, no overflow protection |
| Candle Engine | 5.5/10 | Volume bug, sim-mode misalignment |
| Security | 2/10 | Committed live credentials — P0 |
| Production Readiness | NOT READY | Paper trading only, blocked |

---

## 1. ARCHITECTURE AUDIT — Score: 6.5/10

### Strengths
- Clean separation: `internal/marketdata`, `internal/data`, `internal/api`, `internal/db`
- RuntimeManager provides ordered start/stop with dependency inversion
- TickBus interface correctly decouples producers from consumers

### Critical Issues

#### GOD PACKAGE: `internal/data`
- **File:** `internal/data/`
- **Issue:** Contains `MarketDataIngestor`, `CandleGenerator`, `IndicatorManager`, `MarketHeatmap`, `wsClient` struct, `dataSourceFromConfig()`, and shared `wsPingPeriod`/`wsWriteWait` constants
- **Impact:** Any change to one component risks breaking others; impossible to unit test in isolation
- **Fix:** Split into `internal/ingestor`, `internal/candleengine` (delegate layer), `internal/indicators/manager`

#### GLOBAL STATE SINGLETON
- **File:** `internal/data/heatmap.go:36`
- **Code:** `var globalMarketHeatmap = NewMarketHeatmap()`
- **Issue:** Package-level singleton prevents multi-tenancy and makes testing impossible; `GetMarketHeatmap()` called directly from `ingest.go` creates hidden coupling
- **Fix:** Inject `*MarketHeatmap` through `MarketDataIngestor` constructor

#### CIRCULAR CONCERN: `data` package imports `indicators` package
- The `data` package creates and imports `indicators.Candle` and `indicators.Indicator` — domain types should flow one way; `data` should not depend on `indicators` internals

#### ARCHITECTURE DIAGRAM

```
Zerodha WS ──────┐
                 ▼
SimulatedTicker ─► TickBus (InProcess/Redis/Dual)
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
    MarketDataIngestor  CandleGenerator  (future subscribers)
              │           │
              ▼           ▼
         DB Workers   CandleEngine
              │           │ OnFinalize callback
              ▼           ▼
         TimescaleDB  IndicatorManager ── WS Broadcast
                          │
                          ▼
                       DB Workers ── TimescaleDB
```

---

## 2. TICKBUS AUDIT — Health Score: 5/10

### BUG #1 (P0 — CRASH CAUSE): `InProcessTickBus.Publish` blocks on slow subscribers

- **File:** `internal/marketdata/tickbus/inprocess.go:23-26`
- **Code:**
  ```go
  for _, ch := range b.subscribers {
      select {
      case ch <- tick:          // BLOCKS if subscriber channel is full
      case <-ctx.Done():
          return ctx.Err()
      }
  }
  ```
- **Issue:** There are **2 subscribers** (DataIngestor + CandleGenerator). If **either** subscriber's channel (capacity 1000) fills up, the Publish loop blocks on that subscriber. The *next* tick from the Zerodha WebSocket goroutine or SimulatedTicker tries to Publish, which also blocks. Under `simulationSpeedMultiplier: 8.0` with 9 candle intervals × N instruments, the CandleGenerator processes each tick through 9 interval calculations, each potentially involving lock + finalization + indicator dispatch. This creates a **producer/consumer imbalance that fills the subscriber channel in seconds**, after which the publisher blocks forever — goroutines pile up, memory explodes, crash.
- **Fix:** Use non-blocking send with a default drop (and counter), OR increase buffer to 50000, OR use fan-out goroutines per subscriber:
  ```go
  select {
  case ch <- tick:
  default:
      atomic.AddUint64(&b.droppedTicks, 1) // track drops, never block
  }
  ```

### BUG #2 (P0 — PANIC): `RedisTickBus.Publish` panics on nil type assertion

- **File:** `internal/marketdata/tickbus/redis.go:31`
- **Code:** `ProcessedAtNanos: ctx.Value("processed_at_nanos").(int64)`
- **Issue:** This is an **unchecked type assertion on a context value keyed by a plain string**. If the caller does not inject this value (and none do — `ticker.go` uses `context.Background()`, simulator loop has no such value), this panics at runtime with `interface conversion: interface {} is nil, not int64`
- **Severity:** P0 — kills the process the first time `DualTickBus` or `RedisTickBus` mode is used
- **Fix:**
  ```go
  var processedAt int64
  if v, ok := ctx.Value(processedAtNanosKey).(int64); ok {
      processedAt = v
  } else {
      processedAt = time.Now().UnixNano()
  }
  ```
  Also use a typed context key, never a plain string.

### BUG #3 (HIGH): `InProcessTickBus.Publish` holds RLock while blocking

- **File:** `internal/marketdata/tickbus/inprocess.go:18`
- **Issue:** `mu.RLock()` is held for the **entire publish loop** including blocking `case ch <- tick`. Any concurrent `Subscribe()` call that needs `mu.Lock()` will deadlock if the Publish is blocked.
- **Fix:** Copy the subscribers slice under RLock, then release before the channel loop.

### BUG #4 (HIGH): Double `StartCandleDBWriter` launched

- **File:** `internal/app/app.go:135` AND `internal/data/candles.go:217`
- **Code:** `app.go` registers `candle_db_writer` service which calls `go app.CandleGenerator.StartCandleDBWriter(ctx)`, AND `StartCandleGeneration()` internally also calls `go cg.StartCandleDBWriter(ctx)` at line 217
- **Issue:** Two goroutines drain the same `candleDBFlushCh`. Race condition: both read from the same channel, batches get split. DB upserts will be incomplete and out of order.
- **Fix:** Remove the registration of `candle_db_writer` from `app.go:130-137` — the internal call in `StartCandleGeneration` is sufficient.

### BUG #5 (HIGH): `ticker.go` uses `context.Background()` in live WebSocket callback

- **File:** `internal/api/ticker.go:115`
- **Code:** `tb.Publish(context.Background(), normalized)`
- **Issue:** The `OnTick` callback fires from within the Kite WebSocket goroutine. Using `context.Background()` means there is no cancellation path when the application shuts down — this goroutine will attempt to publish forever and may block if the TickBus channel is full. During reconnect logic, stale goroutines accumulate.
- **Fix:** Pass the application context through `SubscribeToTicks` and capture it in the closure.

### BUG #6 (MEDIUM): `StartMonitoring` goroutine has no context / no stop signal

- **File:** `internal/app/app.go:150-156`
- **Code:** `go monitor.StartSystemMonitor(5*time.Second, alertFunc)` — function ignores context, runs `for {}` forever
- **Issue:** Goroutine leaks on shutdown; the inner goroutine in `StartSystemMonitor` also wraps itself in yet another goroutine
- **Fix:** Pass `ctx context.Context` to `StartSystemMonitor`, break loop on `<-ctx.Done()`

---

## 3. CANDLE GENERATION AUDIT — Score: 5.5/10

### BUG #7 (P0 — CORRECTNESS): Volume uses `LastTradedQuantity` not cumulative delta

- **File:** `internal/marketdata/candles/candle_engine.go:153, 169`
- **Code:**
  ```go
  Volume: float64(tick.LastTradedQuantity),  // open
  candle.Volume += float64(tick.LastTradedQuantity)  // update
  ```
- **Issue:** `LastTradedQuantity` is the size of the **last individual trade** (e.g., 50 shares). The correct candle volume is the **sum of all trade quantities** within the bucket. While `+= LastTradedQuantity` is correct for accumulation, `tick.Volume` (cumulative day volume) should be used to detect and handle duplicate ticks: if `tick.Volume` hasn't changed from the last tick for this instrument, the tick is a duplicate and should be skipped (or the `LastTradedQuantity` is a heartbeat). Currently duplicate ticks inflate candle volume.
- **Also:** `NormalizedTick` carries both `LastTradedQuantity` (trade size) and `Volume` (cumulative day volume). The candle engine ignores `Volume` entirely.
- **Fix:** Track the last known cumulative volume per instrument; candle volume delta = `tick.Volume - lastVolume[token]`; set `lastVolume[token] = tick.Volume` after each tick.

### BUG #8 (P0 — CORRECTNESS): Simulation mode uses `time.Truncate()` — misaligned buckets

- **File:** `internal/marketdata/candles/candle_engine.go:206-207`
- **Code:**
  ```go
  return t.Truncate(interval)
  ```
- **Issue:** `time.Truncate(interval)` truncates relative to the **Unix epoch** (1970-01-01 00:00:00 UTC). For a 5-minute candle, it produces buckets at `:00`, `:05`, `:10`, etc. But the simulated tick `EventTime` is `time.Now()` IST — so 9:16 AM IST truncates to 9:15 AM UTC bucket, not 9:15 AM IST. In simulation, `EventTime` is wall-clock `time.Now()` not a simulated market timestamp, so candle boundaries are misaligned with market session. A 1-minute candle at 11:03:47 IST will be placed in the `11:03:00 IST` bucket (correct for live) but in simulation mode it maps to wrong UTC-relative bucket.
- **More critically:** In simulation mode, all ticks have `EventTime = time.Now()` (current wall clock), so if simulation runs at 8x speed, 6.25 hours of simulated data is generated in ~47 real minutes. The candle finalizer runs on real-time wall clock. At 8x speed, a simulated "9:15 AM" 1-minute candle is generated in 7.5 real seconds — but the finalizer only fires every `FinalizeIntervalMs = 1000ms` and uses `watermark = time.Now() - GracePeriod`. Since `EventTime = time.Now()`, candles are finalized correctly by real-time watermark. **But** `getCandleStartTime` in simulation mode calls `t.Truncate(interval)` where `t` is the current real wall-clock time, not a simulated market time. This means all instruments are in the same real-time bucket and produce correct 1-second candles in real-time — but the "9 candle intervals" each create a subscriber+processing path, magnifying load.
- **Fix for simulation:** Either (a) maintain a simulated clock per instrument and advance it with each tick, or (b) use a fixed simulated market-open anchor and derive bucket from `tickIndex * tickInterval`.

### BUG #9 (HIGH): `OpenCandle` uses `sync.RWMutex` but updated with `mu.Lock()` while engine holds `mu.Lock()`

- **File:** `internal/marketdata/candles/candle_engine.go:162-170` and `open_candle.go:22`
- **Issue:** `CandleEngine.ProcessTick` holds `e.mu.Lock()` (engine-level) and then also acquires `candle.mu.Lock()` (candle-level). This is a **two-level lock order** that must be consistent everywhere. `finalizeByWatermark` also holds `e.mu.Lock()` and calls `finalizeCandleLocked(candle)` — but `OnFinalize` callback (`handleFinalizedCandle`) does `cg.candleDBFlushCh <- ohlcv` (non-blocking, fine) and `cg.indicatorManagerInputCh <- indicatorCandle` (also non-blocking). However, `broadcastCandle` is also called from `finalizeCandleLocked` while `e.mu.Lock()` is held — it calls `cg.candleWsClients.Range(...)` with channel sends. If `client.send` channel is full, the `default` branch drops — OK. But the full `sync.Map.Range` under engine lock means **candle WebSocket client registration/unregistration** contends with every tick. Under high-speed simulation this is a significant throughput bottleneck.
- **Fix:** The `OnFinalize` callback should queue the candle to a channel and return immediately; a separate goroutine handles DB write, indicator dispatch, and WS broadcast outside the engine lock.

### BUG #10 (HIGH): No out-of-order tick protection

- **File:** `internal/marketdata/candles/candle_engine.go`
- **Issue:** If a late tick arrives with `EventTime` before the current open candle's `StartTime`, it is silently treated as a new candle's tick (because `!candle.StartTime.Equal(startTime)` is true — the bucket is different). This forces finalization of the current bucket and opens a new one in the past.
- **Fix:** Add: `if startTime.Before(candle.StartTime) { // drop late tick; log }` before the new candle creation path.

### BUG #11 (MEDIUM): `isMarketOpen` is exclusive of exact open/close times

- **File:** `internal/marketdata/candles/candle_engine.go:176-180`
- **Code:** `marketTime.After(open) && marketTime.Before(close)`
- **Issue:** A tick arriving at exactly 9:15:00.000 is rejected (`After` is strict). The first tick of the day is lost.
- **Fix:** `!marketTime.Before(open) && marketTime.Before(close)` (i.e., `>=` open)

### Per-Timeframe Candle Status

| Interval | Bucket Alignment | OHLC | Volume | Finalization | Rating |
|----------|-----------------|------|--------|-------------|--------|
| 1s | Correct (live), Wrong (sim) | ✓ | WRONG (LTQ not delta) | ✓ | ❌ |
| 5s | Correct (live), Wrong (sim) | ✓ | WRONG | ✓ | ❌ |
| 15s | Correct (live), Wrong (sim) | ✓ | WRONG | ✓ | ❌ |
| 30s | Correct (live), Wrong (sim) | ✓ | WRONG | ✓ | ❌ |
| 1m | Correct (live), Wrong (sim) | ✓ | WRONG | ✓ | ❌ |
| 5m | Correct (live), Wrong (sim) | ✓ | WRONG | ✓ | ❌ |
| 15m | Correct (live), Wrong (sim) | ✓ | WRONG | ✓ | ❌ |
| 30m | Correct (live), Wrong (sim) | ✓ | WRONG | ✓ | ❌ |
| 1h | ✓ (live mode uses market-open alignment) | ✓ | WRONG | ✓ | ❌ |

**Volume is wrong in all 9 timeframes.**

---

## 4. SIMULATION CRASH INVESTIGATION

### Top 10 Most Likely Crash Causes (Ranked)

**#1 — TickBus deadlock / goroutine pile-up (P0)**
`simulationSpeedMultiplier: 8.0` with 9 intervals. At 8x speed, 1 real second = ~8 seconds of simulated market time = ~80 ticks per instrument. The simulator iterates all instruments then sleeps for `realTimeDelay = 100ms / 8 = 12.5ms`. With e.g. 10 instruments, that's 10 Publish calls per 12.5ms. Each Publish blocks on subscriber channels. `CandleGenerator` subscriber processes 9 intervals per tick under engine lock, invoking `OnFinalize` which does channel sends, WS broadcast, and indicator dispatch. Under load, subscriber channels fill and Publish blocks. Goroutines accumulate. OOM follows.

**#2 — Double `StartCandleDBWriter` (P0)**
Two goroutines compete on `candleDBFlushCh`. This creates non-deterministic batch splitting. One goroutine's batch may be empty, causing a GORM `CreateInBatches` call with zero records — this has been known to cause panics in some GORM versions.

**#3 — `ctx.Value` nil panic in RedisTickBus (P0)**
If `tick_bus: "dual"` or `"redis"` is ever tested, first tick panics the process.

**#4 — `OnFinalize` called under engine lock, then calls WS broadcast (HIGH)**
Under high-speed sim, many candles finalize simultaneously. WS broadcast under the engine lock means the engine is stalled for the duration of every broadcast. With 9 intervals × N instruments, this becomes the bottleneck that causes goroutines to accumulate waiting for the engine lock.

**#5 — Indicator channel saturation (HIGH)**
`IndicatorInputCh` has capacity 5000. At 8x speed with 9 intervals × N instruments, finalized candles flow in rapidly. `IndicatorManager` calculates all enabled indicators (SMA, EMA, MACD, ATR, RSI, Stochastic, BB, OBV, VWAP, ADX — 10 indicators) per candle. If `processedIndicatorCh` (also 5000) fills up, the indicator calculation goroutine blocks, which blocks the input channel, which blocks `handleFinalizedCandle`, which is called under the engine lock.

**#6 — DB flush channel saturation (HIGH)**
`candleDBFlushCh` capacity = `DBFlushChannelSize` (10000). At high speed, DB writes can't keep up. Non-blocking send drops candles silently. More critically, `MarketDataIngestor.dbFlushCh` (also 10000) fills when DB workers are behind. The blocking send with 5-second timeout (`dbFlushTimeout`) causes the `processTick` goroutine to block inside `bufferLock.Lock()`, which stalls the TickBus subscriber goroutine, which stalls the entire TickBus.

**#7 — `tickSequenceCounters` unbounded growth during simulation**
Each tick timestamp becomes a key. In simulation, `EventTime = time.Now()`. The cleanup runs every `TickSequenceCleanupInterval = 1s` with expiry `MaxTickSequenceCacheDuration = 2s`. At 8x speed, ticks arrive faster than cleanup. If the cleanup goroutine is behind, `tickSequenceCounters` grows unboundedly for all instruments.

**#8 — WS broadcast goroutines under high fanout**
With 8 WS broadcaster workers, each consuming from `broadcastChannel` (20000) and calling `wsClients.Range(...)` — every iteration of Range under high load causes GC pressure from the closure allocation. Not a crash cause alone but contributes.

**#9 — Heatmap `VolumeAtPrice` map reconstruction**
`internal/data/heatmap.go:60-62`: when `len(stock.VolumeAtPrice) > 100`, the entire map is replaced with `make(map[string]int64)` — all historical data lost. This runs under `hm.mu.Lock()` which is also held during every tick. Under high simulation speed this lock is extremely hot.

**#10 — System monitor goroutine leaks on restart**
`StartSystemMonitor` never stops. If the app is restarted (e.g., in tests or via hot-reload), each restart spawns another monitoring goroutine that runs forever.

---

## 5. PERFORMANCE AUDIT — Top 20 Improvements by Impact

| Rank | Issue | File | Fix |
|------|-------|------|-----|
| 1 | `OnFinalize` called under engine global lock | `candle_engine.go` | Queue to channel, release lock before callback |
| 2 | `InProcessTickBus.Publish` holds RLock while blocking on channel sends | `tickbus/inprocess.go` | Copy slice under RLock, release, then send |
| 3 | `processTick` allocates anonymous struct + JSON marshal on every tick for WS broadcast | `ingest.go` | Pre-allocate struct pool; batch marshal |
| 4 | `handleFinalizedCandle` allocates new `db.OHLCVCandle` and `indicators.Candle` on every finalization | `candles.go` | Use sync.Pool |
| 5 | `safeDepthItem` uses `interface{}` type switch on every tick | `ingest.go` | Use typed array directly; eliminate reflection |
| 6 | `tickSequenceCounters` uses `map[uint]map[time.Time]int` — two-level map with time.Time key | `ingest.go` | Use a single map with composite key or ring buffer |
| 7 | 9 separate TickBus subscriptions (DataIngestor + CandleGenerator) each with full tick copy | architecture | Fan-out via single multiplexer goroutine |
| 8 | `heatmap.VolumeAtPrice` map rebuilt on every `> 100` overflow | `heatmap.go` | Use a ring buffer or LRU with pre-allocated size |
| 9 | `broadcastCandle` marshals JSON on every finalized candle | `candles.go` | Marshal once, cache result |
| 10 | `writeCandleBatch` defers `recoverGoroutine` — defer overhead in hot path | `candles.go` | Remove defer from hot path; use explicit recover only at goroutine boundary |
| 11 | `IndicatorManager.candleHistory` protected by `historyMu sync.RWMutex` — slice trim causes allocation | `indicators_manager.go` | Use circular buffer per instrument/interval |
| 12 | DB workers use GORM `CreateInBatches` which reflects on struct fields every call | `ingest.go` | Use raw SQL batch inserts via pgx |
| 13 | `wsClients.Range` in WS dispatcher called for every broadcast message | `ingest.go` | Cache client list; update only on connect/disconnect |
| 14 | `convertNormalizedToKiteTick` copies entire struct including Depth (5+5 DepthItems) on every tick | `ingest.go` | Keep NormalizedTick as canonical type for frontend too |
| 15 | `realtime/hub.go:ServeHeatmap` marshals full heatmap snapshot every 200ms | `hub.go` | Only broadcast diff; or push-on-change |
| 16 | `sync.Map` for `wsClients` — `Range` under concurrent modification is slow | multiple | Use `sync.RWMutex` + `map[*websocket.Conn]chan []byte` |
| 17 | Logger `zap.L().Debug(...)` with `zap.Float64` allocations on every tick | `ingest.go` | Move all per-tick debug logging behind a rate limiter |
| 18 | `time.After(m.dbFlushTimeout)` in hot path creates a new timer on every tick batch | `ingest.go` | Use `time.NewTimer` + Reset pattern |
| 19 | `ohlcData[token]` is a value-type struct stored in map — written back on every tick | `simulated_ticker.go` | Use pointer map |
| 20 | `indicators_manager.go` copies `candlesCopy` slice on every finalized candle | `indicators_manager.go` | Pass read-only slice view; avoid copy if indicators are called synchronously |

---

## 6. GOROUTINE LEAK AUDIT

| # | File | Function | Leak Reason |
|---|------|----------|-------------|
| 1 | `internal/execution/monitor.go:13` | `StartSystemMonitor` | Inner `go func()` runs forever, no ctx, no stop |
| 2 | `internal/app/app.go:135` | `candle_db_writer` service | Duplicate launch; one goroutine is an orphan |
| 3 | `internal/api/ticker.go:97-151` | `OnClose` reconnect goroutine | Reconnect loop only exits after 5s sleep — if connection immediately fails again, a new goroutine is launched but the outer one never exits; goroutines accumulate |
| 4 | `internal/realtime/hub.go:159` | `ServeHeatmap` inner drain goroutine | `go func() { conn.ReadMessage() }` — no context, no stop signal; leaks when connection drops silently |
| 5 | `internal/data/candles.go:214` | `startMonitoring` | Uses `monitorStopCh` but `monitorStopCh` is never closed from the RuntimeManager's `Stop()` path — `CandleGenerator.Stop()` is registered in code comment as `// stop: nil` |
| 6 | `internal/data/indicators_manager.go:190` | `StartIndicatorCalculations` | `close(im.processedIndicatorCh)` is called in one path but output workers may still be reading — closing a channel being drained is safe, but the 30 output workers have no guaranteed shutdown notification beyond `ctx.Done()` which may not trigger |
| 7 | `internal/api/ticker.go:167` | `go z.Ticker.Serve()` | No context passed to Serve; leaks if OnClose is never called (network partition) |

---

## 7. MEMORY LEAK AUDIT

| # | Location | Type | Worst-Case Growth |
|---|----------|------|------------------|
| 1 | `internal/data/ingest.go: tickSequenceCounters` | `map[uint]map[time.Time]int` | Grows with every unique timestamp × instrument. At 8x sim with 10 instruments and 1 tick per 12.5ms = 80 ticks/sec = 288,000 entries/hour before cleanup. Each `time.Time` key = 24 bytes. ~7MB/hour, potentially 50MB+ in long runs. |
| 2 | `internal/data/indicators_manager.go: candleHistory` | `map[uint32]map[string]*CandleHistory` | Bounded to `maxHistoryPeriods[interval] + 102` per instrument per interval. With 10 instruments × 9 intervals × ~200 candles = 18,000 `indicators.Candle` structs. Each ~200 bytes. ~3.6MB. Bounded but non-trivial. |
| 3 | `internal/data/heatmap.go: VolumeAtPrice` | `map[string]int64` | Bounded to 100 entries per symbol (then reset). Acceptable. |
| 4 | `internal/marketdata/tickbus/inprocess.go: subscribers` | `[]chan marketdata.NormalizedTick` | Never shrinks on subscriber removal (no `Unsubscribe`). If Subscribe is called repeatedly (e.g., on reconnect), old channels accumulate. |
| 5 | WS client channels | `chan []byte` per client | Client channels of 1024 bytes each. On rapid connect/disconnect without proper `UnregisterWebSocketClient` call, channels remain in map. |

---

## 8. REDIS MIGRATION AUDIT

### Migration Status: PARTIAL

The TickBus interface correctly abstracts Redis. However:

| Finding | Status |
|---------|--------|
| `InProcessTickBus` wired by default (`tick_bus: "inprocess"`) | ✓ Correct |
| `RedisTickBus` still exists and compiles | Intentional (future scaling) |
| `DualTickBus` actively publishes to Redis even in inprocess mode | Only if `tick_bus: "dual"` — not default |
| `RedisTickBus.Publish` panics on missing ctx key | P0 BUG — blocks Redis path entirely |
| `cache.RedisClient` is always initialized, even in inprocess mode | Wastes connection; Redis must be running even when not used |
| Old Redis pub/sub path from before TickBus: NONE found | Migration complete structurally |

### Files That Can Be Removed / Scoped

- `internal/marketdata/tickbus/redis.go` — safe to keep for future, but fix the panic
- `internal/marketdata/tickbus/dual.go` — safe to keep
- `internal/cache/redis.go` — KEEP (used by auth JWT caching, hubs)
- Redis is still required as a dependency even in inprocess mode (app.go always initializes it)

**Recommendation:** Gate Redis initialization behind config — only initialize `RedisClient` if `tick_bus != "inprocess"` and Redis-based features are enabled.

---

## 9. DEAD CODE AUDIT — SAFE TO DELETE

| File / Symbol | Reason |
|--------------|--------|
| `internal/strategy/intraday.go` | Empty package declaration only |
| `internal/strategy/scalping.go` | Empty package declaration only |
| `internal/strategy/selector.go` | Empty package declaration only |
| `internal/strategy/swing.go` | Empty package declaration only |
| `internal/execution/order.go` | Comment says "not yet implemented," zero code |
| `internal/backtest/doc.go` | Only a package comment, no implementation |
| `internal/api/handlers/backtest/doc.go` | Only doc comment |
| `internal/api/handlers/broker/doc.go` | Only doc comment |
| `internal/api/handlers/health/doc.go` | Only doc comment |
| `internal/api/handlers/market/doc.go` | Only doc comment |
| `internal/api/handlers/models/doc.go` | Only doc comment |
| `internal/api/handlers/notifications/doc.go` | Only doc comment |
| `internal/api/handlers/orders/doc.go` | Only doc comment |
| `internal/api/handlers/positions/doc.go` | Only doc comment |
| `internal/api/handlers/sentiment/doc.go` | Only doc comment |
| `internal/api/handlers/settings/doc.go` (with no `settings.go`?) | Verify |
| `internal/api/handlers/strategies/doc.go` | Only doc comment |
| `internal/api/handlers/watchlist/doc.go` | Only doc comment |
| `internal/events/doc.go` | Only doc comment, package unused |
| `internal/jobs/doc.go` | Only doc comment, package unused |
| `internal/notifications/doc.go` | Only doc comment, package unused |
| `internal/telemetry/doc.go` | Only doc comment, package unused |
| `internal/validation/doc.go` | Duplicate of `internal/utils/validate.go` |
| `internal/broker/doc.go` | Only doc comment, package unused |
| `internal/services/doc.go` | Only doc comment, package unused |
| `internal/settings/doc.go` | Only doc comment, package unused |
| `internal/model/trainer.go` | `package model` with only placeholder |
| `internal/model/inference.go` | Only struct definitions, no implementation |
| `internal/data/preprocess.go` | 4-line file with only a package comment |
| `nextCandleStart()` in `candle_engine.go` | Defined but never called |
| `cmd/backtest/backtest.go` | 4-line stub with only `package main` |
| `models/sentiment.onnx`, `models/sentiment_optimized.onnx` | 134-byte placeholder files (not real ONNX models) |
| `.access_token` | Live access token committed to repo — DELETE AND ROTATE |

---

## 10. TICKET VERIFICATION AUDIT

### Critical Tickets (as specified)

| Ticket | Description | Status | Finding |
|--------|-------------|--------|---------|
| AUD-009 | Candle bucket alignment | **INCORRECT** | Simulation mode uses `time.Truncate()` relative to epoch, not market-open-relative. Live mode is correct. |
| AUD-010 | Volume correctness | **INCORRECT** | Uses `LastTradedQuantity` accumulation without deduplication. Should use `Volume` delta. |
| AUD-011 | Out-of-order tick handling | **MISSING** | No protection against late ticks. A tick with past `EventTime` triggers finalization of the current candle. |
| AUD-012 | Duplicate tick handling | **PARTIAL** | `tickSequenceCounters` deduplicates for DB writes in `MarketDataIngestor` but `CandleEngine` has no dedup — same tick can be fed twice and doubles the volume. |
| INF-043 | TickBus non-blocking Publish | **INCORRECT** | `InProcessTickBus.Publish` blocks when subscriber channel is full. This is the primary crash cause. |
| INF-044 | TickBus consumer isolation | **PARTIAL** | Consumers have separate channels (isolated). But a slow consumer blocks the publisher, which violates isolation contract. |
| INF-045 | Tick ordering guarantee | **PARTIAL** | Within one subscriber channel, ordering is preserved (FIFO channel). Across subscribers, no ordering guarantee. |
| INF-046 | Graceful TickBus shutdown | **PARTIAL** | `Close()` closes all subscriber channels. But no drain period — consumers may miss in-flight ticks. |
| INF-047 | Context propagation in TickBus | **INCORRECT** | `ticker.go` uses `context.Background()` — no cancellation. `redis.go` panics on missing ctx key. |
| UNI-553 | Candle engine simulation mode | **INCORRECT** | See AUD-009. Simulation bucket alignment is broken. |
| UNI-555 | Market open/close boundary | **PARTIAL** | `isMarketOpen` is exclusive at exact 9:15:00 (first tick dropped). Market close boundary at 15:30 is correct. |

---

## 11. PRODUCTION READINESS REVIEW

### 🔴 CRITICAL BLOCKERS (must fix before ANY live trading)

| # | Issue | Location |
|---|-------|----------|
| C1 | **Live Zerodha API key, secret, and access token committed in `.env`** | `.env:32-33` |
| C2 | **Live Zerodha access token committed in `.access_token` file** | `.access_token` |
| C3 | **Live Redis Cloud URL with password committed in `.env`** | `.env:40` |
| C4 | **Weak `DATA_ENCRYPTION_KEY` in `.env`: `"a_very_strong_32_byte_key_for_encryption"` — this is a placeholder, not a secure key** | `.env:50` |
| C5 | **JWT secret in plaintext in `.env`** | `.env:46` |
| C6 | Order execution module is a stub (`execution/order.go` is empty) — cannot place any orders | `internal/execution/order.go` |
| C7 | Strategy packages are empty stubs — no trading logic exists | `internal/strategy/` |
| C8 | Backtest engine is empty — cannot validate strategies before live trading | `internal/backtest/` |
| C9 | `InProcessTickBus.Publish` blocks under load — causes crash | `tickbus/inprocess.go` |
| C10 | Double `StartCandleDBWriter` launch — data corruption | `app.go:135` + `candles.go:217` |

### 🟠 HIGH-RISK ISSUES (must fix before paper trading)

| # | Issue |
|---|-------|
| H1 | Volume calculation wrong in all 9 candle timeframes |
| H2 | No Global Kill Switch — no way to halt trading programmatically in emergency |
| H3 | No Daily Loss Limit — no circuit breaker on drawdown |
| H4 | No Order Idempotency — duplicate orders possible if retry logic fires |
| H5 | `ctx.Value` nil panic in `RedisTickBus.Publish` |
| H6 | Out-of-order ticks not handled — past-time tick finalizes current candle incorrectly |
| H7 | `isMarketOpen` drops first tick of day (boundary exclusive) |
| H8 | No authentication on WebSocket endpoints — any client can receive market data |
| H9 | `system_monitor` goroutine has no shutdown path |
| H10 | Simulation mode candle bucket alignment is broken |

### 🟡 MEDIUM-RISK ISSUES (fix before multi-user / live)

| # | Issue |
|---|-------|
| M1 | `tickSequenceCounters` unbounded growth under high-speed simulation |
| M2 | Reconnect logic in `ticker.go` goroutine-leaks on repeated disconnect |
| M3 | No rate limiting on TickBus publications — no backpressure signaling |
| M4 | Indicator history not persisted on restart — warm-up required every restart |
| M5 | `heatmap.VolumeAtPrice` silently resets to empty map on overflow |
| M6 | All handlers in `internal/api/handlers/` are stubs with only `doc.go` |
| M7 | GORM error type check `res.Error.Error() == "record not found"` — brittle string match, use `errors.Is(res.Error, gorm.ErrRecordNotFound)` |

---

## PRIORITIZED REMEDIATION PLAN

### Phase 0 — IMMEDIATE (before any commit or test)
| Priority | Action | File |
|----------|--------|------|
| P0-1 | **Rotate all credentials**: Zerodha API key/secret, access token, Redis password, JWT secret, encryption key | `.env`, `.access_token` |
| P0-2 | Add `.env` and `.access_token` to `.gitignore` — they are ALREADY in `.gitignore` but were committed before that rule was added; use `git rm --cached .env .access_token` | `.gitignore` |
| P0-3 | Replace `DATA_ENCRYPTION_KEY` placeholder with a real randomly generated 32-byte AES key | `.env` |

### Phase 1 — Crash Fix (1–2 days)
| Priority | Action | File |
|----------|--------|------|
| P1-1 | Fix `InProcessTickBus.Publish` to be non-blocking (drop with counter) | `tickbus/inprocess.go` |
| P1-2 | Remove duplicate `candle_db_writer` service registration from `app.go` | `app.go:130-137` |
| P1-3 | Fix `RedisTickBus.Publish` `ctx.Value` panic with safe type assertion + fallback | `tickbus/redis.go:31` |
| P1-4 | Move `OnFinalize` callback execution outside of engine lock — queue to channel | `candle_engine.go:finalizeCandleLocked` |
| P1-5 | Pass application context into `SubscribeToTicks` and `SimulateTicks` callbacks | `ticker.go`, `simulated_ticker.go` |

### Phase 2 — Correctness Fix (3–5 days)
| Priority | Action | File |
|----------|--------|------|
| P2-1 | Fix volume calculation: use `tick.Volume` delta, not `tick.LastTradedQuantity` accumulation | `candle_engine.go:153,169` |
| P2-2 | Fix simulation-mode bucket alignment: use market-open-relative bucketing consistent with live mode | `candle_engine.go:206` |
| P2-3 | Add out-of-order tick protection in `ProcessTick` | `candle_engine.go` |
| P2-4 | Fix `isMarketOpen` boundary: `>=` at 9:15:00 | `candle_engine.go:177` |
| P2-5 | Add `tickSequenceCounters` bound by instrument+second key instead of instrument+exact-time | `ingest.go` |
| P2-6 | Fix GORM error check: `errors.Is(res.Error, gorm.ErrRecordNotFound)` | `app.go` |

### Phase 3 — Safety Controls (1 week) — Required for Paper Trading
| Priority | Action |
|----------|--------|
| P3-1 | Implement Global Kill Switch (HTTP endpoint + atomic flag checked before every order) |
| P3-2 | Implement Daily Loss Limit with circuit breaker |
| P3-3 | Implement Order Idempotency key (UUID per order, dedup in DB) |
| P3-4 | Fix goroutine leaks: pass ctx to `StartSystemMonitor`, fix reconnect loop, fix heatmap drain goroutine |

### Phase 4 — Completeness (2–4 weeks) — Required for Live Trading
| Priority | Action |
|----------|--------|
| P4-1 | Implement order execution module (`execution/order.go`) |
| P4-2 | Implement at least one strategy (`strategy/intraday.go`) |
| P4-3 | Implement backtest engine |
| P4-4 | Add WebSocket authentication |
| P4-5 | Clean up all stub packages / dead code |
| P4-6 | Add integration tests for candle correctness (property-based: known tick sequences → expected OHLCV) |

---

*End of Audit — 11 sections, 57 findings, 10 critical bugs identified*
