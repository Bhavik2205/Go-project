# Go-project Repository Audit Master Report

Date: `2026-05-12`  
Audited by: `Codex`  
Audit mode: `incremental, phase-based repository audit`  
Repository: `Go-project`

## Audit Context

This document captures the full repository audit that was performed incrementally across 13 phases to avoid shallow analysis, skipped files, and context loss.

The work was intentionally done in chunks:

1. Repository discovery
2. Build and compilation audit
3. Architecture audit
4. Market data pipeline audit
5. Indicator engine audit
6. WebSocket and realtime audit
7. Database and storage audit
8. Security and authentication audit
9. Performance and concurrency audit
10. API and contract audit
11. Cleanup and technical debt audit
12. Future system design audit
13. Final consolidated report

This audit focused on the real implemented backend runtime, not only the intended roadmap described in `README.md`, `Tasks.md`, and `docs/openapi.yaml`.

## How The Audit Was Performed

- The repository was scanned recursively and mapped package by package.
- Core runtime files were inspected directly, especially:
  - `cmd/server/main.go`
  - `internal/server/*`
  - `internal/data/*`
  - `internal/api/*`
  - `internal/db/*`
  - `internal/indicators/*`
  - `internal/auth/*`
  - `internal/middleware/*`
  - `internal/security/*`
- Migrations, config files, trackers, and documentation were used as supporting references.
- Findings were checkpointed phase by phase instead of merged prematurely.
- The analysis prioritized:
  - correctness
  - determinism
  - recoverability
  - fault tolerance
  - concurrency safety
  - future maintainability

## Executive Summary

`Go-project` is a promising but still prototype-grade trading backend. The strongest part today is that there is a real path for ticks, candles, indicators, auth, and a small REST/WebSocket surface. The weakest part is that the repository structure, README, and OpenAPI spec describe a much more mature multi-user trading platform than the runtime actually implements.

The most important overall conclusion is this:

- Do not rewrite this into microservices.
- Do redesign it into a cleaner modular monolith before adding live trading, strategies, paper trading, backtests, or ML/RL execution.

## Current Project Health Score

- Overall project health: `3.8/10`
- Production readiness: `2.8/10`
- Architecture: `4.5/10`
- Build/package health: `5/10`
- Market data correctness: `3.5/10`
- Candle engine correctness: `3/10`
- Indicator engine readiness: `3.5/10`
- Realtime/WebSocket readiness: `3.5/10`
- Database/storage readiness: `4.5/10`
- Security/auth readiness: `3.5/10`
- Concurrency/runtime safety: `3.5/10`
- API/contract readiness: `4/10`
- Multi-user readiness: `2.5/10`
- Live-trading readiness: `2/10`

## Build Status

Observed state during audit:

- `go version` reported `go1.25.0 windows/amd64`
- many isolated packages built successfully
- `internal/server` and `cmd/server` builds were pathologically slow
- `go build` needed `-buildvcs=false`
- `go test ./...` had environment/cache friction before code-level coverage could be trusted

Primary build problem:

- [internal/server/test_helpers.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/server/test_helpers.go:1) is a non-test `.go` file that imports `testing` and `gorm.io/driver/sqlite`, contaminating the production build graph.

## Critical Risks

### 1. Global Runtime State

Files:
- [internal/server/routes.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/server/routes.go:35)
- [cmd/server/main.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/cmd/server/main.go:31)

Severity: `Critical`

Problem:
- package-level globals are used as dependency injection
- startup orchestration is concentrated in one large `main()`

Why dangerous:
- hidden shared state
- fragile initialization ordering
- poor testability
- no clean lifecycle control

### 2. Non-Deterministic Candle Pipeline

Files:
- [internal/data/candels.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/candels.go:294)
- [internal/data/candels.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/candels.go:361)
- [internal/data/candels.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/candels.go:547)

Severity: `Critical`

Problem:
- candle finalization depends on later tick arrival
- incomplete candles are flushed on shutdown as if final
- no deterministic replay/rebuild path

Why dangerous:
- strategies and indicators can resume from wrong bars
- server restarts can silently corrupt historical truth

### 3. Silent Data Loss Under Load

Files:
- [internal/data/ingest.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/ingest.go:363)
- [internal/data/candels.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/candels.go:445)
- [internal/data/candels.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/candels.go:458)
- [internal/data/indicators_manager.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/indicators_manager.go:437)

Severity: `Critical`

Problem:
- ticks, candles, and indicators can all be dropped on full channels

Why dangerous:
- correctness is sacrificed silently
- replay, audit, and recovery become unreliable

### 4. Global Broker Session

Files:
- [cmd/server/main.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/cmd/server/main.go:134)
- [cmd/get-token/get_token.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/cmd/get-token/get_token.go:20)
- [internal/api/zerodha.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/api/zerodha.go:46)

Severity: `Critical`

Problem:
- broker auth still depends on a single process-wide `.access_token`
- plaintext file storage
- hardcoded request token in source

Why dangerous:
- not multi-user safe
- high credential leakage risk
- blocks investor-grade architecture

### 5. WebSocket Security And Design Gaps

Files:
- [internal/server/routes.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/server/routes.go:110)
- [internal/server/broadcast.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/server/broadcast.go:16)
- [internal/contracts/websocket.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/contracts/websocket.go:22)

Severity: `Critical`

Problem:
- all current websocket endpoints are unauthenticated
- there is no unified hub
- live message shapes do not follow the shared websocket contract

Why dangerous:
- access control risk
- fragile client integration
- poor future scaling

## Architecture Review

Current architecture style:

- single-process monolith
- manual bootstrap in `cmd/server/main.go`
- package-global mutable runtime state in `internal/server`
- Redis pub/sub mixed with in-process channels and direct fanout

Assessment:

- Monolith is still the right deployment model.
- The problem is that this is not yet a clean modular monolith.
- It is a tightly coupled runtime assembled through global setters and side effects.

### Main Architecture Problems

1. `main.go` is acting as bootstrapper, runtime manager, broker bootstrap, simulation bootstrap, and service launcher.
2. `internal/server` mixes routing, dependency registry, websocket handlers, and API behavior.
3. `internal/data` mixes domain processing and transport/persistence side effects.
4. Service boundaries are implied, not explicit.
5. Replay and determinism are not first-class architecture concerns yet.

## Package-By-Package Review

### `internal/server`

Status: `implemented but overloaded`

Issues:
- global setters instead of constructor injection
- REST + WS + runtime state mixed together
- route ownership not split by domain

Recommendation:
- split into `httpapi`, `realtime`, and runtime wiring components

### `internal/data`

Status: `critical runtime package with mixed responsibilities`

Issues:
- ingestion, candle generation, indicator orchestration, persistence, and fanout are too entangled
- correctness and transport policy are mixed

Recommendation:
- split into `ticks`, `candles`, `indicators`, and `publisher/store` boundaries

### `internal/api`

Status: `partly real, partly legacy`

Issues:
- Zerodha/ticker integration is real
- some files are placeholders or minimal
- market handler direction is split with legacy `stockHandler`

Recommendation:
- keep broker adapters and market clients here, but remove legacy/placeholder ambiguity

### `internal/db`

Status: `good starting schema intent, not yet hardened`

Issues:
- destructive migration exists for `market_data`
- time-series and operational concerns need clearer policy
- DB package depends upward on indicator types

Recommendation:
- keep schema direction, remove destructive patterns, move adapter logic out of `db`

### `internal/indicators`

Status: `real calculation library, runtime wrapper needs redesign`

Issues:
- formulas exist
- runtime integration recalculates too much and lacks incremental state

Recommendation:
- keep indicator types/formulas
- redesign `IndicatorManager`

### `internal/auth`, `internal/security`, `internal/middleware`

Status: `good primitives, incomplete production integration`

Issues:
- JWT secret fallback
- refresh revocation gap
- security utilities not fully wired
- no websocket auth

Recommendation:
- keep primitives, harden runtime adoption

### Placeholder / Planned Packages

Packages:
- `internal/backtest`
- `internal/broker`
- `internal/events`
- `internal/jobs`
- `internal/market`
- `internal/notifications`
- `internal/realtime`
- `internal/services`
- `internal/settings`
- `internal/telemetry`

Status: `scaffold-only / orphan-risk`

Recommendation:
- prune, collapse, or implement deliberately

## File-By-File Review

### [cmd/server/main.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/cmd/server/main.go:31)

Severity: `Critical`

Problem:
- too many responsibilities
- no real supervisor model

Impact:
- runtime orchestration gaps
- startup/shutdown complexity

Fix:
- introduce `App` / `RuntimeManager`

### [internal/server/routes.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/server/routes.go:35)

Severity: `Critical`

Problem:
- package-global dependency injection
- unauthenticated websocket routes

Impact:
- hidden shared state
- auth and readiness fragility

Fix:
- constructor-based server dependencies
- authenticated unified WS hub

### [internal/data/ingest.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/ingest.go:31)

Severity: `Critical`

Problem:
- mixed roles: Redis consumer, DB writer, tick sequencer, price cache, WS dispatcher
- silent drop behavior
- channel ownership mistake

Impact:
- correctness loss
- concurrency fragility

Fix:
- separate consumer/store/publisher responsibilities

### [internal/data/candels.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/candels.go:51)

Severity: `Critical`

Problem:
- session alignment model is incomplete
- finalization depends on future tick
- shutdown flush persists incomplete candles

Impact:
- invalid bars and restart inconsistency

Fix:
- deterministic finalized-candle engine with replay

### [internal/data/indicators_manager.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/indicators_manager.go:31)

Severity: `Critical`

Problem:
- hydration exists but depends on flawed candle truth
- full-history recalculation
- goroutine-per-indicator burst pattern
- broadcast even after DB failure

Impact:
- CPU inefficiency
- non-deterministic restarts

Fix:
- incremental stateful engine on finalized candles only

### [internal/auth/jwt.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/auth/jwt.go:29)

Severity: `Critical`

Problem:
- weak default signing secret

Fix:
- fail fast on missing secret

### [internal/api/handlers/auth/refresh.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/api/handlers/auth/refresh.go:20)

Severity: `Critical`

Problem:
- refresh rotation does not check blocklisted refresh tokens

Fix:
- server-side session store with revocation and `jti`

### [internal/db/migrations/000004_create_market_data_table.up.sql](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/db/migrations/000004_create_market_data_table.up.sql:8)

Severity: `Critical`

Problem:
- destructive `DROP TABLE IF EXISTS market_data`

Fix:
- forward-only migrations only

## Duplicate Code Report

### Duplicated Response Helpers

Files:
- [internal/server/api_v1.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/server/api_v1.go:348)
- [internal/api/handlers/auth/helpers.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/api/handlers/auth/helpers.go:10)
- [internal/api/handlers/profile/me.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/api/handlers/profile/me.go:93)

Problem:
- `writeSuccess` and `writeError` duplicated

Recommendation:
- centralize in one shared package

### Duplicated WebSocket Responsibility

Files:
- [internal/data/ingest.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/ingest.go:580)
- [internal/data/candels.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/candels.go:234)
- [internal/data/indicators_manager.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/indicators_manager.go:136)
- [internal/server/broadcast.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/server/broadcast.go:16)

Problem:
- four separate realtime implementations

Recommendation:
- one hub, one envelope, one auth model

## Partial Implementation Report

### Placeholder Commands

- [cmd/backtest/backtest.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/cmd/backtest/backtest.go:1)

### Placeholder Domain Files

- [internal/strategy/intraday.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/strategy/intraday.go:1)
- [internal/strategy/scalping.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/strategy/scalping.go:1)
- [internal/strategy/selector.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/strategy/selector.go:1)
- [internal/strategy/swing.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/strategy/swing.go:1)
- [internal/model/inference.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/model/inference.go:1)
- [internal/model/trainer.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/model/trainer.go:1)
- [internal/execution/order.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/execution/order.go:1)

### Commented-Out Subsystem

- [internal/data/news_pipeline.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/news_pipeline.go:1)

### Planned-Only Handler Namespaces

- `internal/api/handlers/backtest`
- `internal/api/handlers/broker`
- `internal/api/handlers/health`
- `internal/api/handlers/market`
- `internal/api/handlers/models`
- `internal/api/handlers/notifications`
- `internal/api/handlers/orders`
- `internal/api/handlers/positions`
- `internal/api/handlers/runtime`
- `internal/api/handlers/sentiment`
- `internal/api/handlers/settings`
- `internal/api/handlers/strategies`
- `internal/api/handlers/watchlist`

## Concurrency Audit

### Main Findings

1. Shared worker channels are closed by the Redis loop in [internal/data/ingest.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/ingest.go:180).
2. Tick batches are dropped when `dbFlushCh` is full.
3. Finalized candles and indicator results are also dropped on full channels.
4. Indicator computation spawns goroutines per indicator per candle.
5. Shutdown is context-driven but not supervisor-managed.

### Main Runtime Risk

The most dangerous failure mode is not a loud crash. It is silent correctness loss during load or restart.

## WebSocket Audit

### Current State

Routes:
- `/ws`
- `/ws/candles`
- `/ws/indicators`
- `/ws/heatmap`

Problems:
- no websocket auth
- no topic subscription model
- no unified envelope
- fragmented fanout
- per-client heatmap loop duplication

Most likely reason “indicators websocket is not streaming properly”:

- upstream candle or indicator events are being dropped before the socket layer even gets them

## Market Data Pipeline Audit

### Current Flow

```text
Ticker / SimulatedTicker
  -> Redis pub/sub
    -> MarketDataIngestor
    -> CandleGenerator
       -> IndicatorManager
```

### Core Problems

- candles are built from a transient stream
- no authoritative replay model
- incomplete bars are persisted as if final
- late/out-of-order handling is missing
- restart recovery is weak

### Candle Alignment Issue

Important clarification:

If the server starts at `09:18`, the current 5-minute bucket does not become a `09:18` candle. The real failure is that the `09:15-09:20` bucket is incomplete because earlier ticks were missed and not replayed.

That is a replay/recovery problem as much as a bucketing problem.

## Candle Engine Audit

Severity: `Critical`

Main issues:
- wall-clock truncation for sub-hour candles
- finalization on future tick arrival
- incomplete shutdown flush
- no replay-based reconstruction
- no late-data correction model

Recommended redesign:

- event-time bucket assignment
- explicit open vs finalized state
- replay from authoritative tick storage
- correction window for late data

## Indicator Engine Audit

Severity: `Critical`

Main issues:
- warm start depends on flawed candle truth
- full-history recalculation on every candle
- goroutine-per-indicator spikes
- broad historical load query
- silent drop on input/output channels

Recommended redesign:

- finalized candles only
- incremental per-indicator state
- exact bounded hydration queries
- deterministic restart from finalized candles

## Database Audit

### Good Direction

- Postgres + TimescaleDB is appropriate
- schema intent for users, broker accounts, positions, orders, trades, candles, and indicators is strong

### Main Problems

- destructive migration for `market_data`
- raw tick table is too wide for long-term scaling
- no retention/compression policy
- startup/recovery queries are broader than needed
- durable stage boundaries are weak

## Security Audit

### Main Problems

1. weak fallback JWT secret
2. refresh revocation gap
3. global plaintext broker token handling
4. unauthenticated websockets
5. no real multi-user broker isolation
6. audit event table exists, but handlers do not write audit events

### Security Primitives That Are Good

- bcrypt password hashing
- AES-256-GCM encryption utility
- redaction helper
- JWT middleware structure

## Performance Audit

### Main Bottlenecks

1. repeated JSON handling inside internal pipelines
2. full-map websocket fanout per message
3. tick sequencing lock
4. full indicator recalculation
5. too much work per client in heatmap broadcasting

### Main Performance Risk

The current runtime favors liveness with silent drops instead of correctness with explicit backpressure or replay. That is the wrong tradeoff for core trading state.

## Scalability Audit

### What Can Scale For Now

- a small number of internal users
- a small dashboard audience
- a single-node modular-monolith deployment after refactor

### What Will Break First

1. global broker client
2. fragmented websocket implementations
3. non-deterministic candle/indicator path
4. lack of runtime controls
5. lack of strategy/risk/execution boundaries

## Observability Audit

Good:
- Zap logger
- some monitoring goroutines
- request ID middleware

Missing:
- real Prometheus metrics integration in live runtime
- queue depth metrics
- replay/recovery metrics
- lifecycle/readiness metrics
- order/risk/audit traces

## API Audit

### Good Foundation

- shared REST envelope in [internal/contracts/envelope.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/contracts/envelope.go:5)
- shared websocket contract in [internal/contracts/websocket.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/contracts/websocket.go:5)

### Main Problems

- docs/spec are much broader than implemented routes
- validation path does not consistently use the shared error envelope
- websocket runtime ignores the shared websocket contract
- there are effectively two specs:
  - `docs/openapi.yaml`
  - inline `GET /api/v1/openapi.json`

## Cleanup Report

### High-Value Cleanup Work

1. prune or collapse placeholder-only packages
2. centralize duplicated response helpers
3. delete or archive fully commented-out subsystems
4. rename misleading packages/files
5. stop using folder structure as a roadmap placeholder

## Refactoring Plan

### Phase 1

- harden auth and secret management
- remove `.access_token` flow
- fail fast on missing env secrets

### Phase 2

- introduce runtime manager and constructor-based wiring
- remove server globals

### Phase 3

- rebuild tick -> candle -> indicator pipeline around deterministic replay and finalized-state semantics

### Phase 4

- introduce unified authenticated realtime hub and typed websocket events

### Phase 5

- build broker-account abstraction
- add strategy/risk/execution boundaries

### Phase 6

- implement paper trading, PnL service, and backtest reuse

### Phase 7

- add ML feature services and model management only after deterministic market state is trustworthy

## Priority Fix Order

1. JWT secret and refresh-token revocation
2. remove `.access_token` runtime dependency
3. runtime manager + server dependency refactor
4. deterministic candle finalization and replay
5. indicator engine redesign
6. unified websocket hub
7. broker-account abstraction
8. risk and execution boundaries
9. paper trading + PnL
10. backtests

## Short-Term Roadmap

- Secure auth and broker handling
- Fix candle correctness and replay
- Make indicators restart-safe
- Unify websocket contracts
- Reduce structural debt and remove misleading placeholders

## Mid-Term Roadmap

- Runtime control APIs
- Paper trading
- Strategy runtime
- PnL engine
- Backtest jobs
- Audit logging everywhere sensitive

## Long-Term Roadmap

- Multi-user broker-independent trading
- Strong replayable execution and reconciliation
- Model registry and feature serving
- Safe RL experimentation on deterministic replay environments

## Recommended Final Target Architecture

```text
cmd/server
  -> App RuntimeManager
     -> Config / Logger / Metrics
     -> Postgres / Redis
     -> TickSource
     -> TickStore
     -> CandleEngine
     -> CandleStore
     -> IndicatorEngine
     -> IndicatorStore
     -> RealtimeHub
     -> HTTPAPI
     -> BrokerAccountService
     -> StrategyService
     -> RiskService
     -> ExecutionService
     -> PnLService
     -> BacktestService
     -> ModelService
```

## Suggested Package Reorganization

- `internal/app`
- `internal/runtime`
- `internal/httpapi`
- `internal/realtime`
- `internal/marketdata/ticks`
- `internal/marketdata/candles`
- `internal/marketdata/indicators`
- `internal/broker`
- `internal/strategy`
- `internal/execution`
- `internal/risk`
- `internal/pnl`
- `internal/backtest`
- `internal/models`
- `internal/platform/db`
- `internal/platform/cache`

## Suggested Event Schemas

### TickEvent

- `eventId`
- `instrumentToken`
- `symbol`
- `eventTime`
- `ingestedAt`
- `sequenceId`
- `price`
- `volume`
- `source`

### FinalizedCandleEvent

- `eventId`
- `instrumentToken`
- `interval`
- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `tradeCount`
- `isFinal`

### IndicatorEvent

- `eventId`
- `instrumentToken`
- `interval`
- `timestamp`
- `indicatorName`
- `data`

### TradeIntent

- `intentId`
- `userId`
- `brokerAccountId`
- `strategyName`
- `instrumentToken`
- `transactionType`
- `quantity`
- `reason`

## Suggested Database Partitioning Strategy

- `market_data`: hypertable with tuned chunk interval, retention, compression
- `ohlcv_candles`: finalized candles only, hypertable
- indicator tables: convert to hypertables or unify into indicator-value hypertable later
- audit events: append-only operational/security trail

## Suggested Caching Strategy

- Redis for ephemeral cache and optional cross-process fanout
- not authoritative truth
- cache latest prices, derived snapshots, session blocklists

## Suggested TimescaleDB Strategy

- raw tick retention based on replay/audit requirement
- compression for older chunks
- long retention for finalized candles
- optional continuous aggregates for analytics, not live trading truth

## Suggested Redis Strategy

- keep for cache and lightweight pub/sub if needed
- do not rely on Redis pub/sub alone for authoritative replayable event flow

## Suggested Testing Strategy

- unit tests for auth, contracts, validation, security
- deterministic replay tests for ticks -> candles -> indicators
- websocket integration tests
- startup/shutdown/recovery tests
- paper/live parity tests for strategy execution paths

## Suggested CI/CD Flow

- `go mod tidy` validation
- `go test ./...`
- `go vet ./...`
- migration checks
- contract/spec consistency checks
- deterministic replay smoke tests in staging

## Suggested Staging Flow

- simulate mode first
- paper trading next
- live mode only behind explicit runtime guardrails

## Suggested Production Hardening Plan

- strict env validation
- authenticated websockets
- rate limits on auth and sensitive APIs
- audit event logging
- encrypted broker/session secrets
- kill switch
- broker reconciliation worker
- per-user isolation by `user_id`
- deterministic replay path for recovery

## Final Conclusion

This repository is worth continuing, but it is not ready to scale safely by simply adding more features on top of the current shape.

The highest-value move is a focused refactor of:

- runtime orchestration
- market-data determinism
- finalized candle and indicator correctness
- broker ownership
- websocket architecture

If that foundation is corrected first, `Go-project` can realistically grow into a strong multi-user paper/live trading platform. If it is skipped, future strategy, backtest, ML, and live execution work will harden around unstable assumptions and become much more expensive to fix later.
