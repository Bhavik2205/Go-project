# Go-project to signal-execution-desk Integration Audit

Date: 2026-05-06
Last Updated: 2026-05-07

## Integration Status Summary

| Area | Status |
|---|---|
| Auth (signup / login / refresh / logout) | DONE |
| JWT middleware on all protected routes | DONE |
| GET /api/v1/me + PATCH /api/v1/me | DONE |
| GET /api/v1/health | DONE |
| GET /api/v1/brokers/zerodha/status | DONE |
| GET /api/v1/quotes (batch) | DONE |
| GET /api/v1/market/overview | DONE |
| Request ID middleware | DONE |
| Logging middleware | DONE |
| Rate limiting middleware | DONE |
| Input validation layer | DONE |
| JSON response envelope (contracts) | DONE |
| CORS restricted to allowlist | DONE |
| Graceful HTTP server shutdown | DONE |
| WebSocket read/write deadlines + heartbeat | DONE |
| WebSocket ping/pong on all handlers | DONE |
| Heatmap WS context cancellation + backpressure | DONE |
| Redis pubsub reconnect leak fixed | DONE |
| Simulation mode nil-pointer crash fixed | DONE |
| Fatal-in-worker-goroutine replaced with Error+return | DONE |
| DB migrations 000012-000016 | DONE |
| GET /api/v1/settings + PUT /api/v1/settings | TODO |
| Zerodha OAuth connect/callback/disconnect | TODO |
| Watchlist CRUD | TODO |
| Orders + Positions + Trades handlers | TODO |
| Strategy runner + signals | TODO |
| Backtest engine | TODO |
| WebSocket unified hub (/ws/v1/dashboard) | TODO |
| News pipeline + sentiment endpoints | TODO |
| Notifications delivery | TODO |
| Runtime config + metrics endpoints | TODO |
| WS auth handshake | TODO |
| Audit event logging in handlers | TODO |

This document inventories what exists today, what the frontend needs from the Go backend, what is missing, what is duplicated, and what must be fixed before `Go-project` can be cleanly integrated with `signal-execution-desk`.

## Executive Summary

`signal-execution-desk` is a Vite React dashboard. Most dashboard sections are still mock/local UI. The only live backend wiring found is in `DashboardHeader.tsx`, which calls `GET http://localhost:8000/api/instrument?symbol=...` and opens `ws://localhost:8000/ws`.

`Go-project` already has a real HTTP/WebSocket server in `internal/server/routes.go`, live/simulated Zerodha tick ingestion, Redis pub/sub, PostgreSQL/TimescaleDB persistence, candle generation, indicator calculation, and heatmap streaming. It does not yet expose the authenticated user, broker, settings, positions/orders, strategies, model, sentiment, or backtest REST APIs that the frontend screens need.

The settings tab in `signal-execution-desk` has now been fixed to use controlled form state, working tabs/menus/toggles, local persistence, save/reset/test actions, and a backend-ready payload shape. It still needs a backend endpoint before settings can be saved server-side.

## Folders Checked

### `Go-project`

Backend Go service plus Python ML utilities.

Important paths:
- `cmd/main.go`: starts logger, config, DB, Redis, Zerodha, HTTP/WebSocket server, tick ingestion, candle generation, indicator manager, system monitor.
- `cmd/get_token.go`: one-off Zerodha request-token to access-token helper.
- `cmd/heatmap_cli.go`: CLI WebSocket heatmap viewer.
- `internal/server/routes.go`: REST and WebSocket endpoint registration.
- `internal/server/broadcast.go`: heatmap WebSocket handler.
- `internal/api/zerodha.go`: Zerodha client/session helpers.
- `internal/api/ticker.go`: real Kite ticker subscription and Redis publishing.
- `internal/api/simulated_ticker.go`: simulated market feed and Redis publishing.
- `internal/api/instruments.go`: instrument CSV/download/lookup helpers.
- `internal/api/handlers/stockHandler/instrumentData.go`: current quote lookup handler.
- `internal/data/ingest.go`: Redis tick consumption, DB batch writes, `/ws` broadcast payloads.
- `internal/data/candels.go`: candle generation and `/ws/candles` payloads.
- `internal/data/indicators_manager.go`: indicator persistence and `/ws/indicators` payloads.
- `internal/data/heatmap.go`: heatmap snapshot state.
- `internal/db/models.go`: users, broker accounts, instruments, ticks, candles, indicators, orders, trades, positions, strategies, metrics, news.
- `internal/db/migrations`: SQL migrations for the tables above.
- `internal/execution/order.go`: order placement helper.
- `internal/execution/monitor.go`: system monitor helper.
- `internal/model/*`: ONNX sentiment/model helpers.
- `internal/data/news_pipeline.go`: complete-looking news pipeline code, but currently commented out.
- `configs/*.yaml`: app, DB, broker, indicators, model, strategy, docker-compose config.

### `signal-execution-desk`

Frontend React app.

Important paths:
- `src/components/TradingDashboard.tsx`: main section router.
- `src/components/TradingSidebar.tsx`: navigation.
- `src/components/DashboardHeader.tsx`: live quote REST fallback and tick WebSocket connection.
- `src/components/Settings.tsx`: fixed settings tab.
- `src/components/BrokerIntegration.tsx`: mock broker status/config UI.
- `src/components/MarketData.tsx`: mock market overview, option chain, news/calendar.
- `src/components/PositionsPanel.tsx`: mock positions/orders/GTT.
- `src/components/StrategyPanel.tsx`: mock strategy status and system metrics.
- `src/components/MLModelsPanel.tsx`: mock model registry/retraining.
- `src/components/SentimentPanel.tsx`: mock sentiment table/cards.
- `src/components/ui/*`: shadcn/Radix UI primitives.

## Settings Tab Work Completed

File changed:
- `signal-execution-desk/src/components/Settings.tsx`

Fixes made:
- Replaced scattered individual `useState` fields with one typed `SettingsState`.
- Made every tab controlled through Radix `Tabs` `value/onValueChange`.
- Made every select menu controlled and rendered through reusable `SettingsField`.
- Added stable IDs for labels, inputs, switches, and select triggers.
- Added working settings tabs: Zerodha, Notifications, General Trading, Strategy & Models, Data Management, Performance.
- Added missing options useful for backend integration: execution environment, trading mode, candle interval, frontend stream mode, WebSocket buffer size, DB batch size.
- Added Save All action.
- Added Reset action.
- Added Test Tab action.
- Added Zerodha Authenticate action that opens Kite login once an API key is present.
- Added local persistence in `localStorage` under `signal-execution-desk-settings`.
- Added a normalized `buildBackendPayload` function so the future backend payload is clear.
- Removed alert-only save flow.
- Removed unused imports and dead active-tab state.
- Kept secrets local only until a secure backend settings API exists.

Validation:
- `npm run build` passes in `signal-execution-desk`.
- Build warning remains: Vite reports a chunk larger than 500 kB. This is not a settings bug; it can be addressed later with route/component code splitting.

## Current Built Backend APIs

Base URL from config:
- `http://localhost:8000`

### `GET /api/instrument?symbol={SYMBOL}` — Status: `PARTIAL` (legacy, kept for backward compat)

Location:
- `Go-project/internal/server/routes.go`
- `Go-project/internal/api/handlers/stockHandler/instrumentData.go`

Replacement: `GET /api/v1/quotes` is `DONE`. Legacy route kept temporarily.

Remaining issues:
- Hardcoded to NSE.
- No auth.
- No response envelope.
- Must be removed when frontend migrates to `/api/v1/quotes`.

### `GET /api/data/users` — Status: `REMOVED`

Route removed from `registerVersionedRoutes`. Replaced by protected `GET /api/v1/me`.

Remaining: inline lambda in routes.go still present — must be deleted before production.

### `GET /api/cache/test` — Status: `OPEN — must remove before production`

Location:
- `Go-project/internal/server/routes.go`

Debug endpoint still present. Must be removed or protected behind admin auth before production deploy.

## Current Built WebSockets

### `WS /ws` — Status: `PARTIAL` (stability hardened, auth + typed envelope TODO)

Location:
- `Go-project/internal/server/routes.go`
- `Go-project/internal/data/ingest.go`

Stability fixes applied:
- Read deadline (60s) + pong handler + read limit (512KB) added.
- Ping heartbeat (45s) added to writePump.
- Write deadline (10s) added to writePump.

Remaining issues:
- No auth.
- No typed envelope.
- `DashboardHeader.tsx` has hardcoded `ws://localhost:8000/ws`.

### `WS /ws/candles` — Status: `PARTIAL` (stability hardened, auth + typed envelope TODO)

Location:
- `Go-project/internal/server/routes.go`
- `Go-project/internal/data/candels.go`

Stability fixes applied:
- Read deadline + pong handler + read limit added.
- Ping heartbeat + write deadline added to writePump.
- Redis pubsub reconnect leak fixed.
- Fatal on subscribe failure replaced with Error+return.

Remaining issues:
- No frontend consumer.
- No symbol in message, only instrument token.
- No subscribe/filter by symbol or interval.
- No historical candles REST seed endpoint.

### `WS /ws/indicators` — Status: `PARTIAL` (stability hardened, auth + typed envelope TODO)

Location:
- `Go-project/internal/server/routes.go`
- `Go-project/internal/data/indicators_manager.go`

Stability fixes applied:
- Read deadline + pong handler + read limit added.
- Ping heartbeat + write deadline added to writePump.
- Indicator handler cleanup moved to defer (was inside read loop).

Remaining issues:
- No frontend consumer.
- Field casing inconsistent with candles and ticks.
- No subscription/filtering.

### `WS /ws/heatmap` — Status: `PARTIAL` (stability hardened, auth TODO)

Location:
- `Go-project/internal/server/routes.go`
- `Go-project/internal/server/broadcast.go`
- `Go-project/internal/data/heatmap.go`

Stability fixes applied:
- Context cancellation added — handler exits cleanly on server shutdown.
- Write deadline (10s) added to every WriteJSON call.
- Ping heartbeat (45s) added.
- Pong handler + read limit added.
- Background read drain goroutine added so pong handler fires.

Remaining issues:
- No frontend panel consuming it.
- No auth.
- Broadcast frequency (200ms) not client-configurable.

## Backend APIs To Build For Frontend Integration

These endpoints should be built before replacing mock UI sections.

### Health and Runtime

`GET /api/health`
- Returns service status, version, uptime, mode, DB status, Redis status, Zerodha status.
- Needed by: header, strategy/system panel.

`GET /api/runtime/config`
- Returns safe non-secret config: market simulate mode, configured candle intervals, server time, trading session state.
- Needed by: settings, dashboard header.

`GET /api/runtime/metrics`
- Returns DB errors, WebSocket drops, broadcast rate, connected clients, Redis lag, memory/CPU.
- Needed by: StrategyPanel/SystemHealthPanel.

### Authentication and User

`POST /api/auth/signup`
- Body: email, password, userName.
- Creates user.

`POST /api/auth/login`
- Body: email, password.
- Returns JWT/session.

`POST /api/auth/logout`
- Invalidates session/client token.

`GET /api/me`
- Returns current user profile and safe account metadata.

`PATCH /api/me`
- Updates display name/preferences.

Why needed:
- Current backend models include `User`, but no auth routes exist.
- Current frontend hardcodes username as `Demo`.

### Zerodha Broker Integration

`GET /api/brokers/zerodha/login-url`
- Returns Kite login URL for the current user's API key.
- Avoid frontend hardcoding external auth URL.

`GET /api/brokers/zerodha/callback?request_token=...&status=...`
- Exchanges request token for access token.
- Stores encrypted token in `user_broker_accounts`.
- Current README mentions this flow, but the actual route is not implemented.

`GET /api/brokers/zerodha/status`
- Returns connected/disconnected, broker user ID, session expiry, last sync, trading enabled.
- Needed by: BrokerIntegrationPanel, header broker badge, settings.

`POST /api/brokers/zerodha/connect`
- Stores API key/secret safely for current user or starts OAuth flow.

`POST /api/brokers/zerodha/disconnect`
- Deactivates broker account and stops ticker/order operations for that user.

`POST /api/brokers/zerodha/refresh`
- Refresh/revalidate session.

Security requirement:
- Never store API secret/access token in frontend localStorage.
- Add encryption helpers; README references `internal/utils/encryption.go`, but that file does not exist today.

### Settings

`GET /api/settings`
- Returns user-level settings with secrets redacted.

`PUT /api/settings`
- Saves all settings from `Settings.tsx` backend payload.

`PATCH /api/settings/{section}`
- Saves one section: `zerodha`, `notifications`, `general`, `strategy`, `data`, `performance`.

`POST /api/settings/test`
- Body: `{section:"notifications"}` or `{channel:"telegram"}`.
- Sends test alert or validates config.

Needed because:
- Fixed settings tab currently saves locally only.
- Backend currently has YAML configs for service-level values, but no per-user settings API.

Recommended DB tables:
- `user_settings` with `user_id`, `section`, `settings_json`, timestamps.
- `notification_channels` for Telegram/WhatsApp secrets if these are not kept in a vault.
- Add encryption for secret fields.

### Instruments and Market Data

`GET /api/instruments/search?q=RELIANCE&exchange=NSE`
- Uses instrument table/cache, not live quote call.

`GET /api/instruments/{token}`
- Returns instrument metadata.

`POST /api/watchlist`
- Adds symbols/tokens to current user's watchlist.

`GET /api/watchlist`
- Returns user's symbols and latest quote snapshot.

`DELETE /api/watchlist/{token}`
- Removes symbol.

`GET /api/quotes?symbols=NSE:RELIANCE,NSE:TCS`
- Batch quote endpoint. Better than one request per symbol.

`GET /api/market/overview`
- Indices, top gainers/losers, breadth, active volume.
- Needed by: MarketDataPage, MarketOverview.

`GET /api/market/heatmap`
- Current heatmap snapshot for initial render before WebSocket updates.

`GET /api/candles?instrumentToken=...&interval=1m&from=...&to=...`
- Seeds chart with historical candles before `/ws/candles`.

`GET /api/indicators?instrumentToken=...&interval=1m&names=RSI,MACD`
- Seeds indicator panels before `/ws/indicators`.

### Orders, Trades, and Positions

`GET /api/positions`
- Returns current user positions from DB/broker reconciliation.
- Needed by: PositionsPanel.

`GET /api/orders?status=open|executed|rejected|all`
- Returns current user's orders.

`POST /api/orders`
- Places order. Body should map to `internal/execution.OrderRequest`.

`DELETE /api/orders/{brokerOrderId}`
- Cancels an open order.

`PUT /api/orders/{brokerOrderId}`
- Modifies order.

`GET /api/trades`
- Returns executed trades.

`POST /api/gtt`
- Creates GTT order.

`GET /api/gtt`
- Lists GTT orders.

Why needed:
- DB models exist for orders/trades/positions.
- `internal/execution/order.go` exists, but no HTTP handlers expose it.
- Frontend positions/orders panel is fully mock.

### Strategies and Signals

`GET /api/strategies`
- Lists available strategies: scalping, intraday, swing, custom.

`GET /api/user/strategies`
- Lists current user's enabled strategies and params.

`PUT /api/user/strategies/{strategyName}`
- Enables/disables and saves params.

`POST /api/strategies/{strategyName}/start`
- Starts strategy runner.

`POST /api/strategies/{strategyName}/stop`
- Stops strategy runner.

`GET /api/signals`
- Returns latest generated signals.

`WS /ws/signals`
- Streams signal creation/update/expiry.

Critical issue:
- `internal/strategy/intraday.go`, `scalping.go`, `swing.go`, and `selector.go` are empty files. This currently breaks Go package compilation. Strategy implementation has not started in code.

### ML Models

`GET /api/models`
- Lists loaded model files, versions, metrics, active status.

`POST /api/models/{modelId}/activate`
- Makes a model active.

`POST /api/models/{modelId}/retrain`
- Starts retraining job.

`GET /api/models/jobs`
- Lists training/inference jobs.

`WS /ws/models`
- Streams job progress/status.

Current state:
- ONNX files exist in `models/`.
- `internal/model/*` exists.
- Frontend MLModelsPanel is mock and alert-based.

### News and Sentiment

`GET /api/news?symbol=RELIANCE&from=...&to=...`
- Returns news articles and sentiment.

`GET /api/sentiment/summary`
- Overall market/company sentiment summary.

`POST /api/sentiment/analyze`
- Optional on-demand sentiment for text/article.

`WS /ws/sentiment`
- Streams new articles/sentiment updates.

Current state:
- DB `NewsArticle` model and migration exist.
- `internal/api/newsapi.go`, `marketwatch.go`, and model code exist.
- `internal/data/news_pipeline.go` is commented out, so no active pipeline is compiled.
- Frontend SentimentPanel is mock.

### Backtesting

`POST /api/backtests`
- Starts a backtest with strategy, instruments, timeframe, capital, fees.

`GET /api/backtests`
- Lists jobs.

`GET /api/backtests/{id}`
- Returns metrics, trades, equity curve.

`WS /ws/backtests/{id}`
- Streams progress.

Critical issue:
- `cmd/backtest.go` is empty and currently breaks `go test ./...`.
- Frontend backtest section only says "Backtesting Module Coming Soon".

### Notifications

`POST /api/notifications/test`
- Sends test Telegram/WhatsApp notification.

`GET /api/notifications/history`
- Lists delivered/failed alerts.

`WS /ws/alerts`
- Streams risk/order/error alerts.

Current state:
- No notification backend found.
- Settings UI now captures Telegram/WhatsApp fields locally.

## Frontend Mock Areas To Replace

### `DashboardHeader.tsx`

Already connected:
- `GET /api/instrument`
- `WS /ws`

Needs fixes:
- Move `http://localhost:8000` and `ws://localhost:8000` into env vars such as `VITE_API_BASE_URL` and `VITE_WS_BASE_URL`.
- Remove unused `axios` import.
- Use backend health/broker status instead of local `connectionStatus`.
- Fix Stop Bot reconnect behavior.
- Avoid hardcoded symbols; load from watchlist or settings.
- Normalize tick parsing for real and simulated payloads.

### `BrokerIntegration.tsx`

Current state:
- Hardcoded `isConnected = true`.
- Hardcoded last synced date.
- Alert-based connect/disconnect/save.

Needs APIs:
- `GET /api/brokers/zerodha/status`
- `GET /api/brokers/zerodha/login-url`
- `POST /api/brokers/zerodha/disconnect`
- `PUT /api/settings/zerodha`

### `MarketData.tsx`

Current state:
- Entire page uses mock arrays for indices, gainers, losers, option chain, news, calendar, watchlist.

Needs APIs/WebSockets:
- `GET /api/market/overview`
- `GET /api/watchlist`
- `POST /api/watchlist`
- `GET /api/quotes`
- `GET /api/market/heatmap`
- `WS /ws/heatmap`
- `WS /ws/candles`

### `PositionsPanel.tsx`

Current state:
- All equity/F&O positions, open orders, executed orders, rejected orders, GTT orders are mock.

Needs APIs/WebSockets:
- `GET /api/positions`
- `GET /api/orders`
- `POST /api/orders`
- `DELETE /api/orders/{id}`
- `GET /api/trades`
- `GET /api/gtt`
- `WS /ws/orders`
- `WS /ws/positions`

### `StrategyPanel.tsx`

Current state:
- Strategies and system metrics are mock.

Needs APIs/WebSockets:
- `GET /api/strategies`
- `GET /api/user/strategies`
- `PUT /api/user/strategies/{strategyName}`
- `POST /api/strategies/{strategyName}/start`
- `POST /api/strategies/{strategyName}/stop`
- `GET /api/runtime/metrics`
- `WS /ws/signals`

### `MLModelsPanel.tsx`

Current state:
- Mock model registry and alert-based retrain actions.

Needs APIs/WebSockets:
- `GET /api/models`
- `POST /api/models/{id}/activate`
- `POST /api/models/{id}/retrain`
- `GET /api/models/jobs`
- `WS /ws/models`

### `SentimentPanel.tsx`

Current state:
- Mock sentiment rows.

Needs APIs/WebSockets:
- `GET /api/news`
- `GET /api/sentiment/summary`
- `POST /api/sentiment/analyze`
- `WS /ws/sentiment`

### `Settings.tsx`

Current state after fix:
- Working locally.
- Saves to `localStorage`.
- Logs backend-ready payload.

Needs APIs:
- `GET /api/settings`
- `PUT /api/settings`
- `POST /api/settings/test`
- broker-specific secret handling.

## Backend Fixes Required

### Compile blockers — Status: `DONE`

All previously empty files now have valid package declarations and compile cleanly:
- `cmd/backtest/backtest.go` — `DONE` (empty main placeholder, compiles)
- `internal/strategy/intraday.go` — `DONE` (package strategy)
- `internal/strategy/scalping.go` — `DONE` (package strategy)
- `internal/strategy/swing.go` — `DONE` (package strategy)
- `internal/strategy/selector.go` — `DONE` (package strategy)

`go build ./...` ✅ `go vet ./...` ✅ `go test ./...` ✅

### Security gaps

| Gap | Status |
|---|---|
| No auth middleware on HTTP routes | DONE — Authenticate middleware + protected subrouter |
| CORS allows all origins | DONE — corsAllowedOrigins() reads ALLOWED_ORIGINS env var |
| /api/data/users exposes all users publicly | DONE — route removed, replaced by /api/v1/me |
| /api/cache/test is publicly writable to Redis | OPEN — must remove before production |
| Access token loaded from .access_token file globally | OPEN — must be per-user encrypted in DB |
| No encryption helper | DONE — AES-GCM in internal/security/encryption.go |
| .access_token and .env in working tree | OPEN — review .gitignore before commit |

### API design gaps

| Gap | Status |
|---|---|
| No consistent JSON envelope | DONE — contracts package + writeSuccess/writeError helpers |
| No request validation layer | DONE — internal/validation/validate.go |
| No versioning /api/v1/ | DONE — registerVersionedRoutes in api_v1.go |
| No OpenAPI/Swagger contract | PARTIAL — inline spec stub in handleV1OpenAPISpec |
| No typed DTO package | OPEN |
| No pagination for lists | OPEN |
| No WebSocket subscription protocol | OPEN — planned in hub refactor |

### Runtime gaps

| Gap | Status |
|---|---|
| Zerodha session validated in simulate mode | DONE — guarded behind !simulate |
| Instrument list hardcoded in main.go | OPEN — move to DB watchlist |
| No graceful HTTP server shutdown | DONE — srv.Shutdown with 15s context |
| Fatal calls in worker goroutines | DONE — all replaced with Error+return |
| WS handlers no read deadlines or max message size | DONE — SetReadLimit + SetReadDeadline added |
| WS handlers no ping/pong | DONE — ping ticker + pong handler added to all handlers |
| /ws/heatmap no backpressure or context cancel | DONE — write deadlines + context cancel + ping added |

### Data and schema gaps

| Gap | Status |
|---|---|
| No settings table | DONE — migration 000012 |
| No watchlist table | DONE — migration 000013 |
| No notification channel/history table | DONE — migration 000015 |
| No backtest job/result schema | DONE — migration 000014 |
| No audit log table | DONE — migration 000016 |
| InstrumentToken type mismatch uint vs uint32 | OPEN — normalize before production |
| No broker reconciliation worker | OPEN |

### Frontend integration gaps

| Gap | Status |
|---|---|
| Mixed WS message casing | OPEN — normalize when hub is built |
| Symbol format inconsistency | PARTIAL — normalizeToAPISymbol helper in api_v1.go |
| No common apiClient or wsClient in frontend | OPEN |
| No React Query hooks | OPEN |
| No VITE_API_BASE_URL env var | OPEN |
| No frontend type definitions for backend payloads | OPEN |

## Duplicated or Risky Code/Concepts

### Duplicated symbols and instruments

- `cmd/main.go` contains a long real symbol list.
- `cmd/main.go` contains a separate simulated instrument list.
- `DashboardHeader.tsx` contains its own hardcoded symbol maps.
- `MarketData.tsx` contains separate mock market data symbols.
- `PositionsPanel.tsx` contains separate mock symbols/orders.

Recommendation:
- Store instruments/watchlists in DB.
- Expose `/api/watchlist` and `/api/instruments/search`.
- Let frontend render from backend data only.

### Duplicate configuration sources

- Backend uses YAML configs and environment variables.
- Frontend settings now captures similar values locally.
- BrokerIntegration also has API key/secret inputs.

Recommendation:
- Settings tab should be the single frontend surface.
- Backend should expose user settings and redact secrets.
- BrokerIntegration should display connection state and link to Settings for credential updates.

### Duplicate WebSocket client logic potential

- Current frontend opens raw WebSocket in `DashboardHeader.tsx`.
- Future panels will need more sockets.

Recommendation:
- Create `src/lib/api.ts` for REST.
- Create `src/lib/ws.ts` with reconnect, heartbeat, subscription, and typed event dispatch.
- Prefer one multiplexed `/ws/dashboard` or a small number of sockets with message types.

### Dead or misleading documentation

- README references routes/files that do not exist, such as `internal/server/http.go`, `internal/server/websocket.go`, `internal/server/router.go`, `internal/auth/handlers.go`, `internal/utils/encryption.go`, `/signup`, `/login`, `/metrics`.
- Actual routing is in `internal/server/routes.go`.

Recommendation:
- Update README after APIs are implemented.
- Keep this audit as the implementation checklist until then.

## Recommended Target Contract

Use these shared conventions:

REST success:

```json
{
  "data": {},
  "meta": {
    "requestId": "uuid",
    "serverTime": "2026-05-06T12:00:00+05:30"
  }
}
```

REST error:

```json
{
  "error": {
    "code": "ZERODHA_SESSION_EXPIRED",
    "message": "Zerodha session expired",
    "details": {}
  }
}
```

WebSocket envelope:

```json
{
  "type": "TICK",
  "topic": "market.ticks",
  "ts": "2026-05-06T12:00:00+05:30",
  "data": {}
}
```

Frontend env vars:

```text
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
```

## Suggested Build Order

1. Fix Go compile blockers by adding packages or implementation to empty Go files.
2. Add `/api/health`, `/api/runtime/config`, and safe `/api/brokers/zerodha/status`.
3. Add frontend `apiClient`, `wsClient`, and env var base URLs.
4. Add settings backend table and `GET/PUT /api/settings`; wire fixed Settings tab to it.
5. Add auth/session middleware before exposing user/broker/order data.
6. Replace BrokerIntegration mocks.
7. Add watchlist, batch quotes, candles history, and typed market WebSocket consumption.
8. Replace PositionsPanel with real orders/positions APIs.
9. Implement strategy package skeletons and strategy APIs.
10. Implement backtest job API and UI section.
11. Wire ML model registry/retraining APIs.
12. Un-comment or rebuild news pipeline and wire SentimentPanel.
13. Add OpenAPI docs and contract tests for each endpoint.

## Verification Performed

Frontend:
- Command: `npm run build`
- Result: passed.
- Warning: chunk larger than 500 kB after minification.

Backend:
- Command: `go test ./...`
- First run hit sandbox cache permission for Go build cache.
- Re-run with permission reached actual compile blockers.
- Result: failed because empty Go files have no package declaration.

## Immediate Pending Checklist

### DONE
- [x] Fix empty Go files (package declarations added to all strategy + backtest files)
- [x] Add auth middleware and protect all /api/v1/ routes
- [x] Add JWT signup / login / refresh / logout
- [x] Add GET /api/v1/me and PATCH /api/v1/me
- [x] Add GET /api/v1/health with DB + Redis + Zerodha dependency checks
- [x] Add GET /api/v1/brokers/zerodha/status
- [x] Add GET /api/v1/quotes (batch quote endpoint)
- [x] Add GET /api/v1/market/overview
- [x] Add request ID middleware with atomic counter
- [x] Add logging middleware
- [x] Add rate limiting middleware
- [x] Add input validation layer
- [x] Add JSON response envelope (contracts package)
- [x] Restrict CORS to allowlist from ALLOWED_ORIGINS env var
- [x] Add graceful HTTP server shutdown
- [x] Add WebSocket read deadlines + pong handlers + read limits to all four WS handlers
- [x] Add ping heartbeat + write deadlines to all three writePump implementations
- [x] Fix heatmap WS context cancellation + write deadlines + ping
- [x] Fix Redis pubsub reconnect connection leak
- [x] Fix simulation mode nil-pointer crash
- [x] Replace Fatal calls in worker goroutines with Error+return
- [x] Fix double-close panic on monitorStopCh
- [x] Fix flaky TestRequestID_UniquePerRequest test
- [x] Run DB migrations 000012-000016
- [x] Add AES-GCM encryption utilities in internal/security

### TODO — Next Priority
- [ ] Remove /api/cache/test debug route
- [ ] Add backend settings API (GET + PUT /api/v1/settings)
- [ ] Add Zerodha OAuth connect/callback/disconnect
- [ ] Add watchlist CRUD
- [ ] Add orders + positions + trades handlers
- [ ] Add env-based frontend API/WS base URLs (VITE_API_BASE_URL)
- [ ] Add WS auth handshake
- [ ] Add audit event logging in handlers
- [ ] Normalize WS message casing and symbol format
- [ ] Build candles + indicators REST seed endpoints
- [ ] Build unified WebSocket hub (/ws/v1/dashboard)
- [ ] Replace mock frontend panels one by one
- [ ] Update README to match actual code structure
