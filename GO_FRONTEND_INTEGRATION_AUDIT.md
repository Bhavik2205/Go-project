# Go-project to signal-execution-desk Integration Audit

Date: 2026-05-06

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

### `GET /api/instrument?symbol={SYMBOL}`

Location:
- `Go-project/internal/server/routes.go`
- `Go-project/internal/api/handlers/stockHandler/instrumentData.go`

Purpose:
- Fetches a live quote from Zerodha for `NSE:{SYMBOL}` using `z.Kite.GetQuote`.

Frontend usage today:
- `signal-execution-desk/src/components/DashboardHeader.tsx`
- Used as fallback quote fetch for `NIFTY 50`, `NIFTY BANK`, `RELIANCE`, `TCS`.

Current response shape:
- Zerodha quote map keyed by values such as `NSE:RELIANCE`.
- Frontend expects fields like `last_price` and `net_change`.

Issues:
- Hardcoded to NSE.
- No auth.
- No backend-side symbol validation beyond missing query param.
- Handler type-casts `zerodhaClient.(*api.ZerodhaClient)` in routes, which can panic if a different `ZerodhaAPI` implementation is injected.
- No local/simulated fallback for this endpoint when market simulation is enabled.
- No response wrapper or consistent error envelope.

### `GET /api/data/users`

Location:
- `Go-project/internal/server/routes.go`

Purpose:
- Dumps all users from DB.

Frontend usage today:
- None.

Issues:
- No auth.
- Unsafe for production because it exposes user records.
- Should be removed, protected, or replaced by `GET /api/me`.

### `GET /api/cache/test`

Location:
- `Go-project/internal/server/routes.go`

Purpose:
- Sets and reads a Redis test key.

Frontend usage today:
- None.

Issues:
- Debug endpoint; should not remain public in production.
- Should become `GET /api/health/cache` or be hidden behind admin auth.

## Current Built WebSockets

### `WS /ws`

Location:
- `Go-project/internal/server/routes.go`
- `Go-project/internal/data/ingest.go`

Purpose:
- Streams tick data consumed from Redis channel `market_data_ticks`.

Frontend usage today:
- `signal-execution-desk/src/components/DashboardHeader.tsx`

Current message shape:

```json
{
  "symbol": "RELIANCE (NSE)",
  "tick": {
    "InstrumentToken": 738561,
    "LastPrice": 3000.5,
    "NetChange": 15.25,
    "OHLC": {},
    "Depth": {},
    "VolumeTraded": 123456
  }
}
```

Notes:
- Real Zerodha ticks use `kitemodels.Tick`.
- Simulated ticks use a custom `SimTick`, but the frontend currently reads the same key names it needs: `LastPrice`, `NetChange`.

Issues:
- No auth.
- No subscription message from client; every connected frontend receives the full broadcast.
- No heartbeat/ping protocol.
- No typed envelope such as `{type:"TICK", data:{...}}`.
- `DashboardHeader.tsx` has hardcoded `ws://localhost:8000/ws`.
- Reconnect loop is always enabled even when the user presses Stop Bot; closing the socket can schedule reconnect.

### `WS /ws/candles`

Location:
- `Go-project/internal/server/routes.go`
- `Go-project/internal/data/candels.go`

Purpose:
- Streams generated OHLCV candles.

Frontend usage today:
- None.

Current message shape:

```json
{
  "instrument_token": 738561,
  "interval": "1m",
  "timestamp": "2026-05-06T09:15:00Z",
  "open": 3000,
  "high": 3005,
  "low": 2998,
  "close": 3002,
  "volume": 10000,
  "trade_count": 50
}
```

Issues:
- No frontend consumer.
- No symbol included, only instrument token.
- No subscribe/filter by symbol or interval.
- No initial historical candles endpoint to seed charts.

### `WS /ws/indicators`

Location:
- `Go-project/internal/server/routes.go`
- `Go-project/internal/data/indicators_manager.go`

Purpose:
- Streams calculated indicator updates.

Frontend usage today:
- None.

Current message shape:

```json
{
  "type": "INDICATOR_UPDATE",
  "instrumentToken": 738561,
  "interval": "1m",
  "timestamp": "2026-05-06T09:15:00Z",
  "indicator": {
    "indicator_name": "RSI"
  }
}
```

Issues:
- No frontend consumer.
- Field casing differs from candles and ticks (`instrumentToken` vs `instrument_token` vs `InstrumentToken`).
- No subscription/filtering.

### `WS /ws/heatmap`

Location:
- `Go-project/internal/server/routes.go`
- `Go-project/internal/server/broadcast.go`
- `Go-project/internal/data/heatmap.go`

Purpose:
- Streams heatmap snapshots every 200 ms.

Frontend usage today:
- None.

Known client:
- `cmd/heatmap_cli.go` connects to `ws://localhost:8000/ws/heatmap`.

Issues:
- No frontend panel consuming it.
- Broadcast frequency may be heavy for browser rendering; should be throttled or client-configurable.
- No auth or subscription/filtering.

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

### Compile blockers

These files are empty and make `go test ./...` fail:
- `cmd/backtest.go`
- `internal/strategy/intraday.go`
- `internal/strategy/scalping.go`
- `internal/strategy/swing.go`
- `internal/strategy/selector.go`

Observed test result:

```text
cmd/backtest.go:1:1: expected 'package', found 'EOF'
internal/strategy/intraday.go:1:1: expected 'package', found 'EOF'
```

Fix:
- Either remove empty files, add valid package declarations, or implement minimal skeletons.
- Example: `package strategy` for strategy files and `package main` for `cmd/backtest.go`.

### Security gaps

- No auth middleware on any HTTP or WebSocket route.
- CORS allows all origins.
- `/api/data/users` exposes all users publicly.
- `/api/cache/test` is publicly writable to Redis.
- Access token is loaded from `.access_token` file globally, not per user.
- README mentions encrypted token storage and `internal/utils/encryption.go`, but no encryption helper file exists.
- `.access_token`, `.env`, logs, and `instruments.csv` are modified in the working tree and should be reviewed for accidental secret/data commits.

### API design gaps

- No consistent JSON envelope for success/errors.
- No request validation layer.
- No versioning (`/api/v1/...`).
- No OpenAPI/Swagger contract.
- No typed DTO package shared by handlers.
- No pagination for lists.
- No filtering/sorting standards.
- No websocket subscription protocol.

### Runtime gaps

- `cmd/main.go` validates Zerodha session even when `market.simulate: true`; simulation should not require live Zerodha credentials/session.
- Instrument subscription list is hardcoded in `cmd/main.go`.
- Simulated instrument list duplicates much of the real hardcoded list.
- There is no graceful HTTP server shutdown; `ListenAndServe` is started but not shut down through `http.Server.Shutdown`.
- Several goroutines call `zap.L().Fatal`, which exits the process from worker paths instead of surfacing recoverable errors.
- WebSocket handlers do not set read deadlines, pong handlers, write deadlines, or max message sizes.
- `/ws/heatmap` publishes every 200 ms to every client with no backpressure strategy beyond write failure.

### Data and schema gaps

- `Instrument.InstrumentToken` is `uint` in model while tick/candle/indicator flows use `uint32`; migrations use BIGINT for instruments and INTEGER for market data/indicators. This should be normalized.
- `positions`, `orders`, and `trades` models exist but there is no broker reconciliation API/worker exposed to frontend.
- No settings table exists.
- No watchlist table exists.
- No notification channel/history table exists.
- No backtest job/result schema exists.
- No WebSocket connection/session table or audit log exists.

### Frontend integration gaps

- Backend emits mixed casing:
  - Tick: `InstrumentToken`, `LastPrice`, `NetChange`.
  - Candle: `instrument_token`, `trade_count`.
  - Indicator: `instrumentToken`.
- Backend tick symbol is `"RELIANCE (NSE)"`, while REST quote keys are `"NSE:RELIANCE"` and DB instruments use `Tradingsymbol` plus `Exchange`.
- No common `apiClient` or `wsClient` exists in frontend.
- No React Query hooks are defined for backend data.
- No environment variable abstraction exists.
- No frontend type definitions match backend payloads.

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

- Fix empty Go files.
- Add env-based frontend API/WS base URLs.
- Add backend settings API and secure secret storage.
- Replace public debug endpoints.
- Add auth and protect WebSockets.
- Normalize message casing and symbols.
- Build initial data endpoints for candles/indicators/heatmap.
- Add order/position/trade handlers.
- Replace mock frontend panels one by one.
- Update README so it matches actual code.
