# Go-project Audit Execution Roadmap

Date: `2026-05-13`  
Source report: [REPOSITORY_AUDIT_MASTER_REPORT_2026-05-12.md](C:/Users/BAPS/Desktop/TradingBot/Go-project/docs/REPOSITORY_AUDIT_MASTER_REPORT_2026-05-12.md:1)

## Purpose

This document converts the repository audit into an implementation roadmap.

It is meant to answer:

- what should be fixed first
- what can wait
- which files are involved
- what order will reduce risk fastest
- how to move from the current prototype-grade runtime to a safer modular monolith

This roadmap assumes:

- single-node deployment first
- paper trading before live trading
- modular monolith, not microservices
- correctness and recoverability before feature expansion

## Planning Principles

Priority order for execution:

1. Security and correctness blockers
2. Deterministic market-data foundation
3. Runtime/service lifecycle control
4. Realtime contract unification
5. Strategy/risk/execution boundaries
6. Paper trading and replay-safe backtesting
7. Advanced ML/RL and broader product expansion

Rules for implementation:

- do not add live trading before deterministic replay exists
- do not add multi-user broker execution before per-user broker ownership exists
- do not add RL/PPO/A3C before the environment is replayable and stateful
- do not add more websocket surfaces before the unified hub exists

## Work Context Summary

The audit already established that the repository has:

- a real tick -> candle -> indicator flow
- a small real REST/auth surface
- partially real websocket streaming
- a useful schema direction for users, broker accounts, strategies, backtests, and audit events

The audit also established that the main blockers are:

- global mutable runtime wiring
- global broker session model
- non-deterministic candle and indicator recovery
- silent data loss under pressure
- fragmented websocket transport
- placeholder-heavy architecture shell

This roadmap is built to fix those in dependency order.

## Delivery Phases

### Phase P0: Immediate Safety And Structural Blockers

Target outcome:

- stop the most dangerous security and correctness issues
- reduce avoidable build/runtime fragility

Estimated effort:

- `4 to 8 working days`

#### P0.1 Fix JWT Secret Handling

Priority: `P0`

Files:

- [internal/auth/jwt.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/auth/jwt.go:1)
- [cmd/server/main.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/cmd/server/main.go:31)

Tasks:

- remove fallback `"change-me-in-production"`
- fail startup if `JWT_SECRET` is missing
- validate minimum secret length/entropy

Why first:

- this is a direct auth bypass risk if env configuration is wrong

#### P0.2 Fix Refresh Revocation

Priority: `P0`

Files:

- [internal/api/handlers/auth/refresh.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/api/handlers/auth/refresh.go:1)
- [internal/api/handlers/auth/logout.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/api/handlers/auth/logout.go:1)
- [internal/middleware/auth.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/middleware/auth.go:1)

Tasks:

- check Redis refresh-token blocklist during refresh
- add token/session identifier support
- prepare for future persisted refresh sessions

Why first:

- current logout is incomplete

#### P0.3 Remove Global Plaintext Broker Token Flow

Priority: `P0`

Files:

- [cmd/server/main.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/cmd/server/main.go:127)
- [cmd/get-token/get_token.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/cmd/get-token/get_token.go:1)
- [internal/api/zerodha.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/api/zerodha.go:1)
- [internal/db/models.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/db/models.go:19)
- [internal/security/encryption.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/security/encryption.go:1)

Tasks:

- remove `.access_token` as live runtime dependency
- remove hardcoded `requestToken`
- design per-user encrypted broker-token storage path
- keep current real-broker flow disabled or clearly development-only until replaced

Why first:

- current flow is unsafe and blocks multi-user design

#### P0.4 Move Test Helpers Out Of Production Build Graph

Priority: `P0`

Files:

- [internal/server/test_helpers.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/server/test_helpers.go:1)

Tasks:

- move file to `_test.go`
- or move shared helpers to `internal/testutil`

Why first:

- improves build clarity and removes SQLite test pollution from `server`

#### P0.5 Remove Destructive Migration Pattern

Priority: `P0`

Files:

- [internal/db/migrations/000004_create_market_data_table.up.sql](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/db/migrations/000004_create_market_data_table.up.sql:1)

Tasks:

- remove `DROP TABLE IF EXISTS market_data`
- replace with production-safe forward migration strategy

Why first:

- destructive migration on core tick storage is unacceptable

## Phase P1: Runtime Refactor Foundation

Target outcome:

- replace global wiring with explicit application structure
- make service lifecycle controllable

Estimated effort:

- `1.5 to 3 weeks`

#### P1.1 Introduce App / RuntimeManager

Priority: `P1`

Files to create:

- `internal/app/app.go`
- `internal/runtime/manager.go`
- `internal/runtime/service.go`

Files to reduce:

- [cmd/server/main.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/cmd/server/main.go:31)

Tasks:

- define `Service` interface
- define startup/shutdown sequencing
- own service registration centrally
- add readiness/status reporting

Suggested interface:

```go
type Service interface {
    Name() string
    Start(ctx context.Context) error
    Stop(ctx context.Context) error
    Status() string
}
```

#### P1.2 Eliminate Package-Global Dependency Injection

Priority: `P1`

Files:

- [internal/server/routes.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/server/routes.go:35)
- [internal/server/runtime_state.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/server/runtime_state.go:1)
- [internal/server/api_v1.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/server/api_v1.go:91)

Tasks:

- replace `SetDBClient`, `SetIngestor`, `SetIndicatorManager`, etc.
- create a real `Server` or `HTTPAPI` dependency struct

Suggested target:

```go
type Dependencies struct {
    DB *db.DBClient
    Redis *cache.RedisClient
    Broker broker.Provider
    Quotes quotes.Service
    Realtime realtime.Hub
}
```

#### P1.3 Split `internal/server`

Priority: `P1`

Suggested target packages:

- `internal/httpapi`
- `internal/realtime`
- `internal/runtime`

Tasks:

- move route registration into transport-focused package
- move websocket handling into realtime package
- leave only composition wiring at the top level

## Phase P2: Deterministic Market Data Foundation

Target outcome:

- raw ticks become authoritative truth
- candles become deterministic and restart-safe
- indicators consume only trustworthy finalized bars

Estimated effort:

- `2 to 4 weeks`

#### P2.1 Define Canonical Tick Event Model

Priority: `P2`

Files:

- [internal/data/ingest.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/ingest.go:1)
- [internal/db/models.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/db/models.go:54)

Tasks:

- define normalized tick event shape
- preserve event-time and ingest-time separately
- decide whether full top-5 depth remains in canonical store

#### P2.2 Fix Channel Ownership And Lossy Persistence

Priority: `P2`

Files:

- [internal/data/ingest.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/ingest.go:180)
- [internal/data/ingest.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/ingest.go:363)

Tasks:

- move channel closure ownership to service owner
- stop dropping authoritative tick batches
- make failure/backpressure explicit

#### P2.3 Rebuild Candle Engine

Priority: `P2`

Files:

- [internal/data/candels.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/candels.go:1)

Tasks:

- replace wall-clock truncation with session-aware bucketing
- separate open vs finalized candle state
- finalize by watermark/clock logic, not only next tick
- stop flushing incomplete candles as final
- add late/out-of-order tick policy

#### P2.4 Add ReplayManager

Priority: `P2`

Files to create:

- `internal/marketdata/replay/replay_manager.go`

Files involved:

- [internal/data/ingest.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/ingest.go:686)
- [internal/data/candels.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/candels.go:547)

Tasks:

- query authoritative ticks after last safe finalized candle
- rebuild open/missing state on startup
- expose replay for recovery and backtesting later

## Phase P3: Indicator Engine Redesign

Target outcome:

- fast restart
- incremental calculations
- deterministic state updates

Estimated effort:

- `1.5 to 3 weeks`

#### P3.1 Replace Full Recalculation With Incremental State

Priority: `P3`

Files:

- [internal/data/indicators_manager.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/indicators_manager.go:393)
- [internal/indicators/indicators.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/indicators/indicators.go:1)

Tasks:

- implement per-indicator state structs
- compute from finalized candle delta only
- remove goroutine-per-indicator pattern

#### P3.2 Optimize Historical Hydration

Priority: `P3`

Files:

- [internal/data/indicators_manager.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/indicators_manager.go:562)

Tasks:

- load only last N finalized candles per token/interval
- validate continuity
- prepare for optional state snapshots later

#### P3.3 Separate Engine / Persistence / Broadcast

Priority: `P3`

Files:

- [internal/data/indicators_manager.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/indicators_manager.go:235)

Tasks:

- isolate compute path from DB writes
- isolate DB writes from websocket fanout
- do not broadcast success if persistence failed unless explicitly marked non-authoritative

## Phase P4: Unified Realtime Layer

Target outcome:

- one authenticated websocket
- consistent event contracts
- scalable topic routing

Estimated effort:

- `1.5 to 3 weeks`

#### P4.1 Build Realtime Hub

Priority: `P4`

Use as basis:

- [internal/contracts/websocket.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/contracts/websocket.go:1)
- [internal/realtime/README.md](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/realtime/README.md:1)

Tasks:

- one websocket endpoint
- authenticated handshake
- topic subscription protocol
- per-client filters
- backpressure classes per topic

#### P4.2 Retire Ad Hoc WS Endpoints

Priority: `P4`

Files to replace:

- [internal/server/routes.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/server/routes.go:110)
- [internal/server/broadcast.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/server/broadcast.go:1)
- [internal/data/ingest.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/ingest.go:580)
- [internal/data/candels.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/candels.go:234)
- [internal/data/indicators_manager.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/data/indicators_manager.go:136)

Tasks:

- move all event delivery through unified hub
- enforce `contracts.WSEvent`

## Phase P5: Broker And Account Model Refactor

Target outcome:

- per-user broker ownership
- safe multi-user path

Estimated effort:

- `1 to 2 weeks`

#### P5.1 Introduce Broker Adapter Interface

Priority: `P5`

Files:

- [internal/broker/doc.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/broker/doc.go:1)
- [internal/api/zerodha.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/api/zerodha.go:1)

Tasks:

- define broker-agnostic interface
- isolate Zerodha-specific session logic behind adapter

#### P5.2 Implement Real Broker Account Flows

Priority: `P5`

Targets from roadmap/spec:

- broker connect
- callback
- disconnect

Tasks:

- store encrypted tokens in `user_broker_accounts`
- bind broker sessions to user identity
- remove global broker state assumptions from API/runtime

## Phase P6: Strategy, Risk, Execution, And PnL Core

Target outcome:

- first real trading-runtime boundary
- paper mode before live mode

Estimated effort:

- `3 to 5 weeks`

#### P6.1 Create Strategy Runtime

Priority: `P6`

Files to create:

- `internal/strategy/registry.go`
- `internal/strategy/runner.go`
- `internal/strategy/types.go`

Tasks:

- define strategy interface
- load user strategy config
- bind strategies to finalized events, not transport endpoints

#### P6.2 Create Risk Service

Priority: `P6`

Tasks:

- max loss checks
- quantity checks
- market-open checks
- kill switch
- live-trading acknowledgement guard

#### P6.3 Create Execution Service

Priority: `P6`

Tasks:

- accept `TradeIntent`
- require risk approval
- enforce idempotency
- persist execution lifecycle and broker responses

#### P6.4 Create PnL Service

Priority: `P6`

Schema already supports:

- [internal/db/models.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/db/models.go:325)

Tasks:

- realized/unrealized PnL
- per-user and per-strategy attribution
- paper and live parity

## Phase P7: Paper Trading Before Live Trading

Target outcome:

- end-to-end safe simulation
- strategy and execution validation without broker risk

Estimated effort:

- `1.5 to 3 weeks`

Tasks:

- implement `PaperBroker`
- simulated fills
- fees and slippage rules
- PnL updates
- realtime order/position feeds through the same hub

Rule:

- do not open live trading before paper trading is stable and observable

## Phase P8: Backtest System

Target outcome:

- same strategy logic reused offline
- replay-safe result generation

Estimated effort:

- `2 to 4 weeks`

Files already relevant:

- [internal/db/extended_models.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/internal/db/extended_models.go:49)
- [cmd/backtest/backtest.go](C:/Users/BAPS/Desktop/TradingBot/Go-project/cmd/backtest/backtest.go:1)

Tasks:

- implement `BacktestService`
- consume historical finalized candles or ticks
- reuse strategy engine and paper execution model
- store jobs/results in `backtest_jobs`

## Phase P9: API And Contract Consolidation

Target outcome:

- API docs match runtime
- one response contract
- one WS contract

Estimated effort:

- `1 to 2 weeks`

Tasks:

- centralize response writers
- update validation to use shared error envelope
- serve one source-of-truth OpenAPI spec
- trim undocumented/planned endpoints from live docs until implemented

## Phase P10: Cleanup And Repository Simplification

Target outcome:

- reduce false architectural surface area
- lower maintenance cost

Estimated effort:

- `3 to 6 days`

Tasks:

- remove or collapse placeholder strategy files
- remove commented-out `news_pipeline.go` if not being restored immediately
- rename `candels.go` to `candles.go`
- rename `monitor` package in `internal/execution`
- prune doc-only packages not ready for implementation

## Phase P11: Production Hardening

Target outcome:

- stable staging and safer production posture

Estimated effort:

- `1 to 2 weeks`

Tasks:

- auth endpoint rate limiting
- websocket auth and origin allowlist
- audit event logging
- env validation at boot
- Redis/Postgres secure connection options
- observability metrics
- migration pipeline validation

## Phase P12: ML Platform And RL Readiness

Target outcome:

- safe ML feature serving
- no direct RL/live coupling until replay is solid

Estimated effort:

- `later phase after core refactor`

Tasks:

- model registry
- inference activation controls
- offline/online feature parity
- replayable environment modeling
- reward/state/action logging

Rule:

- do not attempt PPO/A3C live execution before deterministic replay, strategy runtime, paper broker, and PnL correctness are all stable

## Recommended Sprint Breakdown

### Sprint 1

- JWT secret hardening
- refresh revocation fix
- remove `.access_token` live dependency
- move `test_helpers.go` out of production build
- remove destructive migration pattern

### Sprint 2

- add `App` / `RuntimeManager`
- remove server globals
- split route/server responsibilities

### Sprint 3

- fix ingest channel ownership
- redesign candle finalization
- implement replay manager

### Sprint 4

- incremental indicator engine
- bounded hydration queries
- persistence/broadcast separation

### Sprint 5

- unified authenticated realtime hub
- typed websocket events
- retire ad hoc WS endpoints

### Sprint 6

- broker adapter + encrypted per-user broker sessions
- real broker connect/callback/disconnect flow

### Sprint 7

- strategy runtime
- risk service
- execution service
- kill switch
- idempotency

### Sprint 8

- PnL engine
- paper broker
- paper trading workflows

### Sprint 9

- backtest engine
- API/spec consolidation
- cleanup pass

## Suggested Ownership Areas

If this work is split among future contributors, use ownership by subsystem:

- Runtime and service lifecycle
- Market data and replay
- Indicators
- Realtime hub
- Auth and broker security
- Strategy/risk/execution
- Backtest and analytics
- Contracts/docs/testing

## Definition Of Done For Core Refactor

The refactor should not be considered complete until all of the following are true:

- no global broker session in live runtime
- no package-global dependency injection in `server`
- finalized candles are deterministic and replayable
- indicators rebuild from finalized candles without long warm-up delays
- core market/candle/indicator paths do not silently drop authoritative events
- one authenticated websocket hub exists
- one REST response envelope exists
- runtime services can be started/stopped independently
- paper trading works end to end
- OpenAPI and live routes match

## Final Recommendation

Do not start with “strategy features” next, even though they are the most exciting part of the roadmap.

The best next implementation sequence is:

1. security and broker token cleanup
2. runtime manager
3. deterministic market-data replay/candle fix
4. indicator redesign
5. websocket hub
6. broker abstraction
7. strategy/risk/execution/PnL
8. paper trading
9. backtesting
10. only then broader ML/RL expansion

That order gives you the best chance of turning `Go-project` into a trustworthy trading backend instead of a feature-rich but unstable one.
