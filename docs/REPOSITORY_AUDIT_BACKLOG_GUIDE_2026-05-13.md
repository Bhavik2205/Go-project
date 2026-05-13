# Repository Audit Backlog Guide

Date: `2026-05-13`  
Backlog file: [REPOSITORY_AUDIT_BACKLOG_2026-05-13.csv](C:/Users/BAPS/Desktop/TradingBot/Go-project/docs/REPOSITORY_AUDIT_BACKLOG_2026-05-13.csv:1)

## Purpose

This guide explains how to use the audit backlog generated from the repository review.

The backlog is meant to bridge the gap between:

- the full audit report
- the execution roadmap
- day-to-day engineering work

## Related Documents

- Master audit report:
  [REPOSITORY_AUDIT_MASTER_REPORT_2026-05-12.md](C:/Users/BAPS/Desktop/TradingBot/Go-project/docs/REPOSITORY_AUDIT_MASTER_REPORT_2026-05-12.md:1)
- Execution roadmap:
  [REPOSITORY_AUDIT_EXECUTION_ROADMAP_2026-05-13.md](C:/Users/BAPS/Desktop/TradingBot/Go-project/docs/REPOSITORY_AUDIT_EXECUTION_ROADMAP_2026-05-13.md:1)
- Backlog CSV:
  [REPOSITORY_AUDIT_BACKLOG_2026-05-13.csv](C:/Users/BAPS/Desktop/TradingBot/Go-project/docs/REPOSITORY_AUDIT_BACKLOG_2026-05-13.csv:1)

## Backlog Columns

- `TaskID`
  - stable audit task identifier
- `Priority`
  - rough execution bucket from `P0` to `P12`
- `Severity`
  - architectural or production impact severity
- `Phase`
  - roadmap phase alignment
- `Subsystem`
  - major area of ownership
- `Title`
  - short action name
- `Description`
  - what should be done
- `PrimaryFiles`
  - likely starting files, not an exhaustive file list
- `EstimatedEffort`
  - rough implementation effort
- `DependsOn`
  - tasks that should usually land first
- `Status`
  - current work state

## Recommended Status Values

Use one of these:

- `TODO`
- `READY`
- `IN_PROGRESS`
- `BLOCKED`
- `REVIEW`
- `DONE`
- `DEFERRED`

## How To Execute The Backlog

### Step 1: Start With P0 Only

Do not mix later architectural work into the first pass.

The best first execution batch is:

- `AUD-001`
- `AUD-002`
- `AUD-003`
- `AUD-004`
- `AUD-005`

These reduce immediate security and build risk before deeper runtime refactors begin.

### Step 2: Use Dependencies Strictly

Do not start downstream items too early.

Examples:

- `AUD-024` strategy runtime should not begin before:
  - replay and finalized candle correctness are fixed
  - indicator state is reliable
  - broker abstraction exists
- `AUD-038` RL preparation should not begin before:
  - replay exists
  - paper trading exists
  - backtesting exists

### Step 3: Group By Refactor Stream

A practical way to work through the backlog is by these streams:

1. Security and auth hardening
2. Runtime and lifecycle control
3. Market data determinism
4. Indicator engine redesign
5. Realtime hub
6. Broker abstraction
7. Strategy/risk/execution/PnL
8. Paper trading and backtesting
9. Contract consolidation and cleanup

## Suggested First Three Sprints

### Sprint 1

- `AUD-001` Remove weak JWT secret fallback
- `AUD-002` Fix refresh token revocation gap
- `AUD-003` Remove global plaintext broker token flow
- `AUD-004` Move server test helpers out of production build
- `AUD-005` Remove destructive market_data migration pattern

Expected outcome:

- major security holes closed
- build path cleaner
- broker token handling no longer normalized as plaintext file behavior

### Sprint 2

- `AUD-006` Introduce App and RuntimeManager
- `AUD-007` Eliminate server package globals
- `AUD-008` Split overloaded server package

Expected outcome:

- runtime structure becomes explicit
- future service control becomes possible

### Sprint 3

- `AUD-009` Define canonical normalized tick event model
- `AUD-010` Fix ingest channel ownership and loss semantics
- `AUD-011` Redesign candle bucket alignment and finalization
- begin `AUD-012` if capacity allows

Expected outcome:

- foundation for deterministic replay and reliable downstream state

## What Not To Start Too Early

Avoid these until the foundation work is done:

- live trading
- real strategy deployment
- broker-independent scaling features
- extensive frontend contract expansion
- PPO/A3C or RL execution
- multi-user live broker orchestration

## Recommended Ownership Model

If this work is split across multiple contributors, use ownership by subsystem:

- Security/Auth:
  - `AUD-001`, `AUD-002`, `AUD-023`, `AUD-034`, `AUD-035`
- Runtime/Core:
  - `AUD-006`, `AUD-007`, `AUD-008`
- Market Data:
  - `AUD-009`, `AUD-010`, `AUD-011`, `AUD-012`, `AUD-013`, `AUD-014`
- Indicators:
  - `AUD-015`, `AUD-016`, `AUD-017`
- Realtime:
  - `AUD-018`, `AUD-019`, `AUD-020`
- Broker:
  - `AUD-021`, `AUD-022`
- Strategy/Execution:
  - `AUD-024`, `AUD-025`, `AUD-026`, `AUD-027`, `AUD-028`, `AUD-029`
- API/Docs/Cleanup:
  - `AUD-030`, `AUD-031`, `AUD-032`, `AUD-033`
- Observability/ML:
  - `AUD-036`, `AUD-037`, `AUD-038`

## Definition Of A Safe Milestone

You can consider the backend to have crossed an important safety milestone when all of the following are complete:

- `AUD-001`
- `AUD-002`
- `AUD-003`
- `AUD-006`
- `AUD-007`
- `AUD-010`
- `AUD-011`
- `AUD-012`
- `AUD-013`
- `AUD-015`
- `AUD-018`
- `AUD-021`

At that point, the codebase should be much closer to a real modular monolith with a trustworthy market-data foundation.

## Recommended Next Step

The best immediate next step is to start a `P0` implementation batch and track progress directly in the CSV by updating:

- `Status`
- `EstimatedEffort`
- `DependsOn`

If needed, the backlog can later be expanded with:

- owner
- target sprint
- actual effort
- review notes
- merged PR/commit references
