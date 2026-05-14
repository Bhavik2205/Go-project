# API & WebSocket Status Report

**Project:** TradingBot (Go-project + signal-execution-desk)  
**Generated:** 2025-06-04  
**Scope:** Every REST endpoint, every WebSocket stream, and the frontend auth gap — what is done, what is pending, and what each one feeds.

---

## Table of Contents

1. [REST APIs — Done](#1-rest-apis--done)
2. [REST APIs — Pending](#2-rest-apis--pending)
3. [WebSocket Endpoints — Done](#3-websocket-endpoints--done)
4. [WebSocket Topics — Pending](#4-websocket-topics--pending)
5. [Frontend Auth Gap](#5-frontend-auth-gap)
6. [Summary Counts](#6-summary-counts)

---

## 1. REST APIs — Done

These endpoints have full handler implementations registered in `internal/server/api_v1.go` and are live when the server starts.

| # | Method | Endpoint | Handler File | Auth Required | What It Does |
|---|--------|----------|--------------|:---:|---|
| 1 | `POST` | `/api/v1/auth/signup` | `handlers/auth/signup.go` | No | Creates user with bcrypt-hashed password, returns JWT access + refresh token pair |
| 2 | `POST` | `/api/v1/auth/login` | `handlers/auth/login.go` | No | Verifies bcrypt password, returns JWT access + refresh token pair |
| 3 | `POST` | `/api/v1/auth/refresh` | `handlers/auth/refresh.go` | No | Validates refresh token, checks Redis blocklist, rotates and returns new token pair |
| 4 | `POST` | `/api/v1/auth/logout` | `handlers/auth/logout.go` | No | Blocklists both the access token (from `Authorization` header) and refresh token in Redis |
| 5 | `GET` | `/api/v1/health` | `server/api_v1.go` | No | Returns service status, version, uptime seconds, trading mode (live/simulation), and postgres/redis/zerodha dependency health |
| 6 | `GET` | `/api/v1/brokers/zerodha/status` | `server/api_v1.go` | **Yes** | Fetches live Zerodha user profile via KiteConnect, returns connected status, broker user ID, account name, trading enabled flag |
| 7 | `GET` | `/api/v1/quotes` | `server/api_v1.go` | **Yes** | Batch quote lookup by comma-separated symbols (e.g. `NSE:RELIANCE,NSE:TCS`). Tries Zerodha first, falls back to heatmap snapshot |
| 8 | `GET` | `/api/v1/market/overview` | `server/api_v1.go` | **Yes** | Returns top gainers, top losers, most active by volume, market indices, and market breadth (advancers/decliners/unchanged) from live heatmap |
| 9 | `GET` | `/api/v1/me` | `handlers/profile/me.go` | **Yes** | Returns authenticated user's id, email, userName, isActive, createdAt |
| 10 | `PATCH` | `/api/v1/me` | `handlers/profile/me.go` | **Yes** | Updates authenticated user's `userName` field |
| 11 | `GET` | `/api/v1/openapi.json` | `server/api_v1.go` | No | Returns inline OpenAPI 3.0 spec (only documents the 5 implemented routes — drifted from reality) |
| 12 | `GET` | `/api/instrument` | `handlers/stockHandler/instrumentData.go` | No | Legacy Zerodha instrument token lookup by symbol query param |

**Frontend components currently consuming these:**

- `MarketOverview.tsx` → `/api/v1/health` + `/api/v1/market/overview`
- `BrokerIntegration.tsx` → `/api/v1/brokers/zerodha/status`
- `DashboardHeader.tsx` → `/api/v1/health` + `/api/v1/brokers/zerodha/status` + `/api/v1/quotes`
- `ApiDocsPage.tsx` → `/api/v1/openapi.json`

---

## 2. REST APIs — Pending

These endpoints have handler package folders under `internal/api/handlers/` but contain only a `doc.go` placeholder — zero implementation. DB models and migrations for all of them already exist.

### 2.1 Auth & Session (UI missing, backend done)

| # | Method | Endpoint | Handler Package | Feeds | Notes |
|---|--------|----------|-----------------|-------|-------|
| 1 | — | — | — | `LoginPage.tsx` (missing) | Backend login is done. Frontend has no login page or token storage |
| 2 | — | — | — | `SignupPage.tsx` (missing) | Backend signup is done. Frontend has no signup page |

### 2.2 Broker OAuth Flow

| # | Method | Endpoint | Handler Package | Feeds | Notes |
|---|--------|----------|-----------------|-------|-------|
| 3 | `GET` | `/api/v1/brokers/zerodha/connect` | `handlers/broker/` | `Settings.tsx` → Zerodha tab | Initiates OAuth redirect to Zerodha login. DB model `UserBrokerAccount` exists. Encryption utility exists in `internal/security/encryption.go` |
| 4 | `GET` | `/api/v1/brokers/zerodha/callback` | `handlers/broker/` | Zerodha OAuth redirect | Exchanges `request_token` for access token, encrypts and stores in `user_broker_accounts` |
| 5 | `POST` | `/api/v1/brokers/zerodha/disconnect` | `handlers/broker/` | `Settings.tsx` → Zerodha tab | Deactivates broker session for the authenticated user |

### 2.3 Positions & Orders

| # | Method | Endpoint | Handler Package | Feeds | Notes |
|---|--------|----------|-----------------|-------|-------|
| 6 | `GET` | `/api/v1/positions` | `handlers/positions/` | `PositionsPanel.tsx` → Equity + F&O tabs | DB model `Position` exists (migration 000007). Should return `symbol, qty, buyAvg, ltp, currentValue, dayPnl, totalPnl, product` |
| 7 | `GET` | `/api/v1/orders` | `handlers/orders/` | `PositionsPanel.tsx` → Order History tabs | DB model `Order` exists (migration 000005). Filter by `status` query param: `open`, `executed`, `rejected`, `gtt` |

### 2.4 Strategies

| # | Method | Endpoint | Handler Package | Feeds | Notes |
|---|--------|----------|-----------------|-------|-------|
| 8 | `GET` | `/api/v1/strategies` | `handlers/strategies/` | `StrategyPanel.tsx` → strategy cards | DB model `UserStrategy` exists (migration 000009). Strategy logic files exist in `internal/strategy/` (intraday, scalping, swing) |
| 9 | `PATCH` | `/api/v1/strategies/:id/status` | `handlers/strategies/` | `StrategyPanel.tsx` → toggle switch | Updates `is_enabled` on `user_strategies` table |
| 10 | `GET` | `/api/v1/strategies/:id/metrics` | `handlers/strategies/` | `StrategyPanel.tsx` → performance metrics | Returns trades, win rate, daily P&L, capital allocated per strategy |

### 2.5 ML Models

| # | Method | Endpoint | Handler Package | Feeds | Notes |
|---|--------|----------|-----------------|-------|-------|
| 11 | `GET` | `/api/v1/models` | `handlers/models/` | `MLModelsPanel.tsx` → model cards | Should return model name, type, accuracy, F1 score, status, last trained, next retrain, prediction, confidence. ONNX models exist in `models/` directory |
| 12 | `POST` | `/api/v1/models/:id/retrain` | `handlers/models/` | `MLModelsPanel.tsx` → Retrain button | Triggers retraining job. `internal/model/trainer.go` is a placeholder |
| 13 | `GET` | `/api/v1/models/:id/status` | `handlers/models/` | `MLModelsPanel.tsx` → training progress bar | Returns current training progress percentage and status |

### 2.6 Sentiment

| # | Method | Endpoint | Handler Package | Feeds | Notes |
|---|--------|----------|-----------------|-------|-------|
| 14 | `GET` | `/api/v1/sentiment` | `handlers/sentiment/` | `SentimentPanel.tsx` → news cards | DB model `NewsArticle` exists (migration 000008) with `sentiment_score` and `sentiment_label`. `internal/data/news_pipeline.go` is fully commented out |
| 15 | `GET` | `/api/v1/sentiment/summary` | `handlers/sentiment/` | `SentimentPanel.tsx` → aggregated badge | Returns overall sentiment score, positive/negative/neutral counts |

### 2.7 Market Data Extensions

| # | Method | Endpoint | Handler Package | Feeds | Notes |
|---|--------|----------|-----------------|-------|-------|
| 16 | `GET` | `/api/v1/market/sectors` | `handlers/market/` | `MarketData.tsx` → Sector Performance panel | No DB model yet. Can be derived from heatmap snapshot grouped by sector |
| 17 | `GET` | `/api/v1/market/forex` | `handlers/market/` | `MarketData.tsx` → Forex & Commodities panel | No DB model yet. Requires external data source (MCX/forex feed) |
| 18 | `GET` | `/api/v1/market/futures` | `handlers/market/` | `MarketData.tsx` → F&O Insights → Nifty Futures tab | Requires Zerodha F&O instrument data via KiteConnect |
| 19 | `GET` | `/api/v1/market/options` | `handlers/market/` | `MarketData.tsx` → F&O Insights → Option Chain tab | Requires Zerodha option chain data via KiteConnect |

### 2.8 Watchlist

| # | Method | Endpoint | Handler Package | Feeds | Notes |
|---|--------|----------|-----------------|-------|-------|
| 20 | `GET` | `/api/v1/watchlist` | `handlers/watchlist/` | `MarketData.tsx` → My Watchlist panel | DB models `Watchlist` + `WatchlistItem` exist (migration 000013) |
| 21 | `POST` | `/api/v1/watchlist/items` | `handlers/watchlist/` | `MarketData.tsx` → add symbol | Adds instrument to user's default watchlist |
| 22 | `DELETE` | `/api/v1/watchlist/items/:id` | `handlers/watchlist/` | `MarketData.tsx` → remove symbol | Removes item from watchlist |

### 2.9 Settings

| # | Method | Endpoint | Handler Package | Feeds | Notes |
|---|--------|----------|-----------------|-------|-------|
| 23 | `GET` | `/api/v1/settings` | `handlers/settings/` | `Settings.tsx` → load saved settings on mount | DB model `UserSetting` exists (migration 000012) with per-section JSONB storage |
| 24 | `PATCH` | `/api/v1/settings/:section` | `handlers/settings/` | `Settings.tsx` → Save All Settings button | Upserts settings JSON for a section (zerodha, notifications, general, strategy, data, performance) |

### 2.10 Notifications

| # | Method | Endpoint | Handler Package | Feeds | Notes |
|---|--------|----------|-----------------|-------|-------|
| 25 | `GET` | `/api/v1/notifications/channels` | `handlers/notifications/` | `Settings.tsx` → Notifications tab | DB model `NotificationChannel` exists (migration 000015) |
| 26 | `PATCH` | `/api/v1/notifications/channels/:type` | `handlers/notifications/` | `Settings.tsx` → toggle Telegram/WhatsApp | Updates `is_enabled` and `config` JSONB for a channel type |

### 2.11 System Metrics

| # | Method | Endpoint | Handler Package | Feeds | Notes |
|---|--------|----------|-----------------|-------|-------|
| 27 | `GET` | `/api/v1/system/metrics` | `handlers/runtime/` | `SystemHealthPanel.tsx` → all 21 metric cards | `handlers/runtime/doc.go` exists. Should expose: CPU load, memory usage, goroutine count, GC pause ms, uptime, API latency p50/p95, orders/sec, fill rate, slippage, disk I/O, network stats. Currently all 21 metrics are `Math.random()` on the frontend |

### 2.12 Backtest

| # | Method | Endpoint | Handler Package | Feeds | Notes |
|---|--------|----------|-----------------|-------|-------|
| 28 | `POST` | `/api/v1/backtest/jobs` | `handlers/backtest/` | `TradingDashboard.tsx` → Backtest section (currently "Coming Soon") | DB model `BacktestJob` exists (migration 000014). `cmd/backtest/backtest.go` is a placeholder |
| 29 | `GET` | `/api/v1/backtest/jobs` | `handlers/backtest/` | Backtest section → job history list | Returns list of past backtest jobs for the user |
| 30 | `GET` | `/api/v1/backtest/jobs/:id` | `handlers/backtest/` | Backtest section → job result view | Returns job status, progress, result JSON, equity curve |

---

## 3. WebSocket Endpoints — Done

All four endpoints are registered in `internal/server/routes.go` and actively stream data. None require authentication (open to any client).

### `/ws` — Raw Market Ticks

- **URL:** `ws://host/ws`
- **Handler:** `handleConnections` in `routes.go` → `MarketDataIngestor`
- **Payload shape:**
  ```json
  {
    "symbol": "NSE:RELIANCE",
    "tick": {
      "LastPrice": 3052.5,
      "NetChange": 12.3,
      "VolumeTraded": 521000,
      "OHLC": { "Open": 3040, "High": 3060, "Low": 3035, "Close": 3040 },
      "Depth": { "Buy": [...], "Sell": [...] }
    }
  }
  ```
- **Frequency:** Every tick from Zerodha KiteTicker (real-time, ~100ms in live mode)
- **Used by:** `DashboardHeader.tsx` — live price ticker bar for SBIN, HDFCBANK, RELIANCE, TCS
- **Issues:** No auth, no topic filtering, broadcasts all symbols to all clients

---

### `/ws/candles` — OHLCV Candle Updates

- **URL:** `ws://host/ws/candles`
- **Handler:** `handleCandleConnections` in `routes.go` → `CandleGenerator`
- **Payload shape:**
  ```json
  {
    "instrument_token": 738561,
    "interval": "1m",
    "timestamp": "2025-06-04T09:16:00Z",
    "open": 3040.0,
    "high": 3055.0,
    "low": 3038.0,
    "close": 3052.5,
    "volume": 12400.0,
    "trade_count": 87
  }
  ```
- **Frequency:** On every candle finalization (when a new candle bucket starts). Intervals: `1m`, `5m`, `15m`, `1h`, `1d` (configured in `configs/app.yaml`)
- **Used by:** Not yet consumed by any frontend component
- **Issues:** No auth, no symbol filtering, sends all instruments to all clients

---

### `/ws/indicators` — Technical Indicator Updates

- **URL:** `ws://host/ws/indicators`
- **Handler:** `handleIndicatorConnections` in `routes.go` → `IndicatorManager`
- **Payload shape:**
  ```json
  {
    "type": "INDICATOR_UPDATE",
    "instrumentToken": 738561,
    "interval": "1m",
    "timestamp": "2025-06-04T09:16:00Z",
    "indicator": {
      "indicator_name": "RSI",
      "Value": 62.4
    }
  }
  ```
- **Indicators streamed:** SMA, EMA, MACD, ATR, RSI, Stochastic, Bollinger Bands, OBV, VWAP, ADX (all configurable via `configs/indicators.yaml`)
- **Frequency:** On every candle close, one message per indicator per instrument per interval
- **Used by:** Not yet consumed by any frontend component
- **Issues:** No auth, no topic filtering, no symbol filtering

---

### `/ws/heatmap` — Market Heatmap Snapshot

- **URL:** `ws://host/ws/heatmap`
- **Handler:** `HeatmapWebSocketHandler` in `broadcast.go`
- **Payload shape:**
  ```json
  [
    {
      "Symbol": "RELIANCE (NSE)",
      "LastPrice": 3052.5,
      "PriceChangePct": 0.41,
      "Volume": 521000,
      "BidPrice": 3052.0,
      "AskPrice": 3053.0,
      "LastUpdated": "2025-06-04T09:16:00Z"
    }
  ]
  ```
- **Frequency:** Every 200ms (fixed ticker in `broadcast.go`)
- **Used by:** Not yet consumed by any frontend component (heatmap section in `MarketData.tsx` is a placeholder card)
- **Issues:** No auth, pushes full snapshot of all instruments every 200ms regardless of client interest

---

## 4. WebSocket Topics — Pending

These topic constants are defined in `internal/contracts/websocket.go` but the unified WebSocket hub (`internal/realtime/`) is empty — only a `doc.go` exists. None of these streams exist yet.

| # | Topic Constant | Topic String | Endpoint Needed | What It Should Stream | Needed By |
|---|---------------|--------------|-----------------|----------------------|-----------|
| 1 | `WSTopicOrders` | `"orders"` | `/ws` unified hub | Real-time order status transitions: `PENDING → OPEN → FILLED / REJECTED / CANCELLED` with orderId, symbol, qty, price | `PositionsPanel.tsx` — live order status without polling |
| 2 | `WSTopicPositions` | `"positions"` | `/ws` unified hub | Real-time position P&L updates as LTP changes: unrealizedPnl, lastPrice per position | `PositionsPanel.tsx` — live unrealized P&L |
| 3 | `WSTopicAlerts` | `"alerts"` | `/ws` unified hub | Risk alerts (daily loss limit hit, kill switch triggered), system error alerts | `Settings.tsx` — critical error alert notifications |
| 4 | `WSTopicStrategies` | `"strategies"` | `/ws` unified hub | Strategy status changes (active/inactive), signal events (BUY/SELL signal generated), strategy P&L updates | `StrategyPanel.tsx` — live strategy state without polling |
| 5 | `WSTopicModels` | `"models"` | `/ws` unified hub | ML model training progress (0–100%), training completion events, new prediction outputs | `MLModelsPanel.tsx` — training progress bar |
| 6 | `WSTopicBacktests` | `"backtests"` | `/ws` unified hub | Backtest job progress (% complete), job completion with result summary | Backtest section — live job progress |
| 7 | *(not yet defined)* | `"system.metrics"` | `/ws` unified hub | Live system metrics push every 5s: CPU, memory, goroutine count, GC pause, API latency, orders/sec, fill rate | `SystemHealthPanel.tsx` — replace all 21 `Math.random()` values |

**Note:** The `WSSubscribeRequest` contract is already defined in `contracts/websocket.go` with `type`, `topics[]`, and `filters` fields. The hub just needs to be built in `internal/realtime/`.

---

## 5. Frontend Auth Gap

The backend has complete auth (signup, login, logout, refresh with Redis blocklist). The frontend has **zero auth integration** — no login page, no token storage, no protected route guards, and no `Authorization` header on any API call.

### What is missing on the frontend

| # | What | File to Create/Modify | Details |
|---|------|-----------------------|---------|
| 1 | Login page | `src/pages/LoginPage.tsx` (new) | Form with email + password. Calls `POST /api/v1/auth/login`. On success stores `accessToken` and `refreshToken` |
| 2 | Signup page | `src/pages/SignupPage.tsx` (new) | Form with email, password, userName. Calls `POST /api/v1/auth/signup`. Auto-logs in after |
| 3 | Auth context / token store | `src/lib/auth.ts` (new) | Stores tokens in `localStorage` or memory. Exposes `getAccessToken()`, `setTokens()`, `clearTokens()` |
| 4 | Auth routes in App.tsx | `src/App.tsx` (modify) | Add `/login` and `/signup` routes. Wrap `/` in a route guard that redirects to `/login` if no token |
| 5 | Bearer token on all API calls | `src/lib/api.ts` (modify) | Add `Authorization: Bearer <token>` header to `apiGet` and any future `apiPost`/`apiPatch` helpers |
| 6 | Token refresh interceptor | `src/lib/api.ts` (modify) | On 401 response, call `POST /api/v1/auth/refresh` with stored refresh token, update stored tokens, retry original request |
| 7 | Logout button | `src/components/DashboardHeader.tsx` (modify) | Button calls `POST /api/v1/auth/logout` with refresh token, then clears stored tokens and redirects to `/login` |
| 8 | WebSocket auth | `src/components/DashboardHeader.tsx` (modify) | Pass JWT as query param or first message after connect: `ws://host/ws?token=<accessToken>` |

### Why this matters right now

Every protected endpoint (`/api/v1/brokers/zerodha/status`, `/api/v1/market/overview`, `/api/v1/quotes`, `/api/v1/me`) returns **401 Unauthorized** for any unauthenticated request. The frontend currently calls all of these without a token, so they silently fail and show loading/error states. The app appears to work only because the heatmap and health endpoints are public.

---

## 6. Summary Counts

| Category | Count |
|---|---|
| REST endpoints — Done (backend + registered) | **12** |
| REST endpoints — Pending (handler is `doc.go` only) | **30** |
| WebSocket endpoints — Done (streaming) | **4** |
| WebSocket topics — Pending (contracts defined, hub missing) | **7** |
| Frontend auth items missing | **8** |
| **Total gaps** | **45** |

### Priority order for development

1. **Frontend auth** — Login/Signup pages + token storage + Bearer header. Nothing protected works without this.
2. **`GET /api/v1/system/metrics`** — Replaces all 21 random mock values in `SystemHealthPanel.tsx`.
3. **Broker OAuth** — Connect/callback/disconnect so `Settings.tsx` Zerodha tab actually works.
4. **Positions + Orders** — `PositionsPanel.tsx` is the most-used trading view, currently 100% mock.
5. **Strategies** — `StrategyPanel.tsx` toggle switches need real backend state.
6. **Unified WebSocket hub** — `orders`, `positions`, `strategies` topics for real-time updates without polling.
7. **ML Models + Sentiment** — `MLModelsPanel.tsx` and `SentimentPanel.tsx` real data.
8. **Settings persistence** — `GET/PATCH /api/v1/settings` so saved config survives page reload.
9. **Watchlist CRUD** — `MarketData.tsx` watchlist persistence.
10. **Backtest section** — Replace "Coming Soon" with real job submission and results.
