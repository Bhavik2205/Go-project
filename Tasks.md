# TradingBot Tasks

Last updated: 2026-05-06

This file is the living backend-first execution tracker for turning `Go-project` into a secure, production-grade trading platform that can power `signal-execution-desk` for personal use first and later support paying multi-user customers.

Status legend:
- `DONE`: implemented and verified.
- `SKELETON`: folders/files exist, implementation pending.
- `PARTIAL`: some implementation exists, integration incomplete.
- `TODO`: not implemented.
- `BLOCKED`: cannot proceed until listed blocker is fixed.

## Current Scope

Backend first:
- Finish clean package structure.
- Define production API/WebSocket contracts.
- Add security, encryption, auth, audit, observability, and deployment hardening tasks.
- Only after backend contracts are stable, wire frontend screens one by one.

## Structural Work Completed

Status: `DONE`

Created backend package structure in `Go-project`:
- `cmd/server`
- `cmd/get-token`
- `cmd/heatmap-cli`
- `cmd/backtest`
- `internal/api/dto`
- `internal/api/handlers/auth`
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
- `internal/auth`
- `internal/backtest`
- `internal/broker`
- `internal/contracts`
- `internal/events`
- `internal/jobs`
- `internal/market`
- `internal/middleware`
- `internal/notifications`
- `internal/realtime`
- `internal/security`
- `internal/services`
- `internal/settings`
- `internal/telemetry`
- `internal/validation`

Created placeholder migration files:
- `000012_create_user_settings_table`
- `000013_create_watchlists_table`
- `000014_create_backtest_jobs_table`
- `000015_create_notification_tables`
- `000016_create_audit_events_table`

Fixed structural compile blockers by adding package declarations:
- `Go-project/cmd/backtest.go`
- `Go-project/internal/strategy/intraday.go`
- `Go-project/internal/strategy/scalping.go`
- `Go-project/internal/strategy/selector.go`
- `Go-project/internal/strategy/swing.go`

New compile blocker discovered after the empty files were fixed:
- `cmd/main.go`, `cmd/get_token.go`, and `cmd/heatmap_cli.go` all declare `package main` with a `main()` function in the same `cmd` package.
- Target structure folders now exist: `cmd/server`, `cmd/get-token`, `cmd/heatmap-cli`, `cmd/backtest`.
- Next implementation pass should move each command into its target folder so `go test ./...` can compile the command packages independently.

## Global API Standards

Status: `TODO`

Base path:
- Production: `/api/v1`
- Local temporary compatibility: existing `/api/*` can redirect or be kept during migration.

Required request headers:

```http
Authorization: Bearer <jwt>
Content-Type: application/json
Accept: application/json
X-Request-ID: <uuid optional>
X-Client-Version: signal-execution-desk/<version optional>
```

Public endpoints that do not require `Authorization`:
- `GET /api/v1/health`
- `POST /api/v1/auth/signup`
- `POST /api/v1/auth/login`
- Zerodha OAuth callback, with signed `state`.

Standard success response:

```json
{
  "data": {},
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

Standard paginated response:

```json
{
  "data": [],
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1",
    "pagination": {
      "limit": 50,
      "cursor": "next_cursor",
      "nextCursor": "next_cursor",
      "hasMore": true
    }
  }
}
```

Standard error response:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "One or more fields are invalid.",
    "details": {
      "field": "symbol",
      "reason": "symbol is required"
    }
  },
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

Common HTTP codes:
- `200`: success.
- `201`: created.
- `202`: async job accepted.
- `204`: success with no body.
- `400`: validation error.
- `401`: missing/invalid auth.
- `403`: forbidden.
- `404`: not found.
- `409`: conflict/idempotency conflict.
- `422`: semantically invalid order/config.
- `429`: rate limited.
- `500`: internal error.
- `503`: dependency unavailable.

## Global WebSocket Standards

Status: `TODO`

Required handshake:

```http
GET /ws/v1/dashboard
Authorization: Bearer <jwt>
X-Request-ID: <uuid optional>
```

Fallback auth for browsers if custom headers are not practical:

```text
ws://localhost:8000/ws/v1/dashboard?token=<short_lived_ws_token>
```

Client subscribe message:

```json
{
  "type": "SUBSCRIBE",
  "requestId": "req_01HX...",
  "topics": [
    "market.ticks",
    "market.candles",
    "orders",
    "positions",
    "alerts"
  ],
  "filters": {
    "symbols": ["NSE:RELIANCE", "NSE:TCS"],
    "instrumentTokens": [738561],
    "intervals": ["1m", "5m"]
  }
}
```

Server event envelope:

```json
{
  "type": "MARKET_TICK",
  "topic": "market.ticks",
  "eventId": "evt_01HX...",
  "serverTime": "2026-05-06T23:10:00+05:30",
  "data": {}
}
```

Server error event:

```json
{
  "type": "ERROR",
  "topic": "system",
  "eventId": "evt_01HX...",
  "error": {
    "code": "SUBSCRIPTION_DENIED",
    "message": "You are not allowed to subscribe to this topic."
  }
}
```

Realtime security requirements:
- Authenticated handshake.
- Short-lived WebSocket token support.
- Topic-level authorization.
- Ping/pong heartbeat.
- Read/write deadlines.
- Max message size.
- Per-client send queue.
- Backpressure policy.
- Connection audit logs.

## APIs Already Built

### `GET /api/instrument?symbol={SYMBOL}`

Status: `PARTIAL`

Current location:
- `Go-project/internal/api/handlers/stockHandler/instrumentData.go`

Current behavior:
- Fetches Zerodha quote for `NSE:{SYMBOL}`.

Current headers:

```http
Accept: application/json
```

Current response:
- Raw Zerodha quote map.

Target replacement:
- `GET /api/v1/quotes?symbols=NSE:RELIANCE,NSE:TCS`

Issues:
- No auth.
- Hardcoded `NSE`.
- Raw third-party response shape leaks into frontend.
- No simulated quote fallback.

### `GET /api/data/users`

Status: `PARTIAL`

Current behavior:
- Returns all users.

Action:
- Remove or replace with protected `GET /api/v1/me`.

Security issue:
- Must not be public.

### `GET /api/cache/test`

Status: `PARTIAL`

Current behavior:
- Writes and reads Redis test key.

Action:
- Remove or replace with protected/admin `GET /api/v1/health/dependencies`.

## WebSockets Already Built

### `WS /ws`

Status: `PARTIAL`

Streams:
- Tick data.

Target replacement:
- `WS /ws/v1/dashboard` topic `market.ticks`.

Target event:

```json
{
  "type": "MARKET_TICK",
  "topic": "market.ticks",
  "eventId": "evt_01HX...",
  "serverTime": "2026-05-06T23:10:00+05:30",
  "data": {
    "symbol": "NSE:RELIANCE",
    "instrumentToken": 738561,
    "lastPrice": 3000.5,
    "netChange": 15.25,
    "percentChange": 0.51,
    "volumeTraded": 123456,
    "ohlc": {
      "open": 2990,
      "high": 3010,
      "low": 2980,
      "close": 2985
    },
    "depth": {
      "buy": [],
      "sell": []
    },
    "exchangeTimestamp": "2026-05-06T09:15:01+05:30"
  }
}
```

### `WS /ws/candles`

Status: `PARTIAL`

Target event:

```json
{
  "type": "MARKET_CANDLE",
  "topic": "market.candles",
  "eventId": "evt_01HX...",
  "serverTime": "2026-05-06T23:10:00+05:30",
  "data": {
    "symbol": "NSE:RELIANCE",
    "instrumentToken": 738561,
    "interval": "1m",
    "timestamp": "2026-05-06T09:15:00+05:30",
    "open": 3000,
    "high": 3005,
    "low": 2998,
    "close": 3002,
    "volume": 10000,
    "tradeCount": 50
  }
}
```

### `WS /ws/indicators`

Status: `PARTIAL`

Target event:

```json
{
  "type": "INDICATOR_UPDATE",
  "topic": "market.indicators",
  "eventId": "evt_01HX...",
  "serverTime": "2026-05-06T23:10:00+05:30",
  "data": {
    "symbol": "NSE:RELIANCE",
    "instrumentToken": 738561,
    "interval": "1m",
    "timestamp": "2026-05-06T09:15:00+05:30",
    "name": "RSI",
    "values": {
      "period": 14,
      "value": 62.5
    }
  }
}
```

### `WS /ws/heatmap`

Status: `PARTIAL`

Target event:

```json
{
  "type": "HEATMAP_SNAPSHOT",
  "topic": "market.heatmap",
  "eventId": "evt_01HX...",
  "serverTime": "2026-05-06T23:10:00+05:30",
  "data": {
    "items": [
      {
        "symbol": "NSE:RELIANCE",
        "lastPrice": 3000.5,
        "percentChange": 0.51,
        "bid": 3000,
        "ask": 3001,
        "volume": 123456
      }
    ]
  }
}
```

## Backend API Implementation Plan

### Health

#### `GET /api/v1/health`

Status: `TODO`

Auth:
- Public.

Headers:

```http
Accept: application/json
```

Response:

```json
{
  "data": {
    "status": "ok",
    "service": "go-project",
    "version": "0.1.0",
    "uptimeSeconds": 3600,
    "mode": "simulation",
    "dependencies": {
      "postgres": "ok",
      "redis": "ok",
      "zerodha": "not_configured"
    }
  },
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

Tasks:
- Create handler.
- Add dependency checks.
- Add readiness variant if needed.

### Auth

#### `POST /api/v1/auth/signup`

Status: `TODO`

Headers:

```http
Content-Type: application/json
Accept: application/json
```

Payload:

```json
{
  "email": "user@example.com",
  "password": "StrongPassword123!",
  "userName": "Bhavik",
  "country": "IN",
  "timezone": "Asia/Kolkata"
}
```

Response:

```json
{
  "data": {
    "user": {
      "id": "usr_01HX...",
      "email": "user@example.com",
      "userName": "Bhavik",
      "isActive": true
    },
    "accessToken": "<jwt>",
    "refreshToken": "<refresh_token>",
    "expiresIn": 900
  },
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

Security tasks:
- Hash password with Argon2id or bcrypt.
- Validate password strength.
- Enforce unique email.
- Add email verification later.
- Rate limit signup.

#### `POST /api/v1/auth/login`

Status: `TODO`

Payload:

```json
{
  "email": "user@example.com",
  "password": "StrongPassword123!",
  "mfaCode": "123456"
}
```

Response:

```json
{
  "data": {
    "user": {
      "id": "usr_01HX...",
      "email": "user@example.com",
      "userName": "Bhavik"
    },
    "accessToken": "<jwt>",
    "refreshToken": "<refresh_token>",
    "expiresIn": 900
  },
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

Security tasks:
- Add login throttling.
- Add failed login audit events.
- Add MFA later for public launch.

#### `POST /api/v1/auth/refresh`

Status: `TODO`

Payload:

```json
{
  "refreshToken": "<refresh_token>"
}
```

Response:

```json
{
  "data": {
    "accessToken": "<jwt>",
    "refreshToken": "<rotated_refresh_token>",
    "expiresIn": 900
  },
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

#### `POST /api/v1/auth/logout`

Status: `TODO`

Headers:

```http
Authorization: Bearer <jwt>
Content-Type: application/json
```

Payload:

```json
{
  "refreshToken": "<refresh_token>",
  "allDevices": false
}
```

Response:
- `204 No Content`

### Profile

#### `GET /api/v1/me`

Status: `TODO`

Headers:

```http
Authorization: Bearer <jwt>
Accept: application/json
```

Response:

```json
{
  "data": {
    "id": "usr_01HX...",
    "email": "user@example.com",
    "userName": "Bhavik",
    "country": "IN",
    "timezone": "Asia/Kolkata",
    "createdAt": "2026-05-06T23:10:00+05:30"
  },
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

### Broker: Zerodha

#### `GET /api/v1/brokers/zerodha/status`

Status: `TODO`

Headers:

```http
Authorization: Bearer <jwt>
Accept: application/json
```

Response:

```json
{
  "data": {
    "broker": "ZERODHA",
    "connected": true,
    "brokerUserId": "AB1234",
    "accountName": "Zerodha Personal",
    "sessionExpiry": "2026-05-07T06:00:00+05:30",
    "lastSyncedAt": "2026-05-06T23:10:00+05:30",
    "tradingEnabled": false
  },
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

#### `POST /api/v1/brokers/zerodha/connect`

Status: `TODO`

Payload:

```json
{
  "apiKey": "kite_api_key",
  "apiSecret": "kite_api_secret",
  "redirectUrl": "http://localhost:8000/api/v1/brokers/zerodha/callback",
  "accountName": "Zerodha Personal"
}
```

Response:

```json
{
  "data": {
    "loginUrl": "https://kite.trade/connect/login?api_key=...",
    "state": "oauth_state_token",
    "expiresAt": "2026-05-06T23:20:00+05:30"
  },
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

Security tasks:
- Encrypt API secret before persistence.
- Store OAuth state server-side with expiry.
- Never return API secret.

#### `GET /api/v1/brokers/zerodha/callback?request_token=...&status=success&state=...`

Status: `TODO`

Auth:
- Valid OAuth `state`; may not include JWT because broker redirects browser.

Response:

```json
{
  "data": {
    "connected": true,
    "broker": "ZERODHA",
    "brokerUserId": "AB1234",
    "sessionExpiry": "2026-05-07T06:00:00+05:30"
  },
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

#### `POST /api/v1/brokers/zerodha/disconnect`

Status: `TODO`

Payload:

```json
{
  "revokeLocalSession": true
}
```

Response:

```json
{
  "data": {
    "connected": false
  },
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

### Settings

#### `GET /api/v1/settings`

Status: `TODO`

Response:

```json
{
  "data": {
    "zerodha": {
      "apiKey": "kite_api_key",
      "apiSecretSet": true,
      "accessTokenSet": true,
      "redirectUrl": "http://localhost:8000/api/v1/brokers/zerodha/callback",
      "brokerageType": "Flat",
      "environment": "Paper"
    },
    "notifications": {
      "frequency": "Instant",
      "telegramEnabled": true,
      "telegramBotTokenSet": true,
      "telegramChatId": "-1234567890",
      "whatsappEnabled": false,
      "whatsappApiUrlSet": false,
      "whatsappPhoneNumber": "",
      "notifyTradeExecution": true,
      "notifyPnlThreshold": false,
      "pnlThresholdValue": 1000,
      "notifyErrorAlerts": true
    },
    "general": {
      "riskPerTrade": 1,
      "maxDailyLoss": 5000,
      "maxOpenPositions": 10,
      "slippageTolerance": 0.1,
      "orderTypePreference": "Market",
      "defaultExchange": "NSE",
      "tradingHoursStart": "09:15",
      "tradingHoursEnd": "15:30",
      "logLevel": "INFO",
      "tradingMode": "Paper"
    },
    "strategy": {
      "enableAutoRetrain": true,
      "retrainFrequency": "Daily",
      "backtestingFrequency": "Weekly",
      "deploymentMethod": "Paper Trading",
      "dataRetentionDays": 30,
      "modelVersionControlEnabled": true,
      "modelVersionControlSystem": "Git"
    },
    "data": {
      "dataSource": "KiteConnect",
      "dataRefreshRate": 5,
      "dataStorageLocation": "/data/historical/",
      "candleInterval": "1m",
      "websocketMode": "Ticks + Candles + Indicators"
    },
    "performance": {
      "enableGpuAcceleration": false,
      "parallelProcessingCores": 4,
      "memoryLimitGB": 8,
      "websocketBufferSize": 20000,
      "dbBatchSize": 1000
    }
  },
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

#### `PUT /api/v1/settings`

Status: `TODO`

Payload:

```json
{
  "zerodha": {
    "apiKey": "kite_api_key",
    "apiSecret": "kite_api_secret",
    "accessToken": "optional_manual_token",
    "redirectUrl": "http://localhost:8000/api/v1/brokers/zerodha/callback",
    "brokerageType": "Flat",
    "environment": "Paper"
  },
  "notifications": {
    "telegram": {
      "enabled": true,
      "botToken": "telegram_bot_token",
      "chatId": "-1234567890"
    },
    "whatsapp": {
      "enabled": false,
      "apiUrl": "",
      "phoneNumber": ""
    },
    "frequency": "Instant",
    "notifyTradeExecution": true,
    "notifyPnlThreshold": false,
    "pnlThresholdValue": 1000,
    "notifyErrorAlerts": true
  },
  "trading": {
    "riskPerTrade": 1,
    "maxDailyLoss": 5000,
    "maxOpenPositions": 10,
    "slippageTolerance": 0.1,
    "orderTypePreference": "Market",
    "defaultExchange": "NSE",
    "tradingHours": {
      "start": "09:15",
      "end": "15:30"
    },
    "logLevel": "INFO",
    "tradingMode": "Paper"
  },
  "strategy": {
    "enableAutoRetrain": true,
    "retrainFrequency": "Daily",
    "backtestingFrequency": "Weekly",
    "deploymentMethod": "Paper Trading",
    "dataRetentionDays": 30,
    "modelVersionControlEnabled": true,
    "modelVersionControlSystem": "Git"
  },
  "data": {
    "dataSource": "KiteConnect",
    "dataRefreshRate": 5,
    "dataStorageLocation": "/data/historical/",
    "candleInterval": "1m",
    "websocketMode": "Ticks + Candles + Indicators"
  },
  "performance": {
    "enableGpuAcceleration": false,
    "parallelProcessingCores": 4,
    "memoryLimitGB": 8,
    "websocketBufferSize": 20000,
    "dbBatchSize": 1000
  }
}
```

Response:

```json
{
  "data": {
    "saved": true,
    "redactedSecrets": [
      "zerodha.apiSecret",
      "zerodha.accessToken",
      "notifications.telegram.botToken",
      "notifications.whatsapp.apiUrl"
    ],
    "updatedAt": "2026-05-06T23:10:00+05:30"
  },
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

Security tasks:
- Encrypt secret fields.
- Redact secret fields from reads and logs.
- Audit settings changes.
- Validate live trading mode requires explicit acknowledgement.

### Market Data

#### `GET /api/v1/instruments/search?q=RELIANCE&exchange=NSE`

Status: `TODO`

Response:

```json
{
  "data": [
    {
      "instrumentToken": 738561,
      "symbol": "NSE:RELIANCE",
      "tradingSymbol": "RELIANCE",
      "exchange": "NSE",
      "name": "Reliance Industries",
      "instrumentType": "EQ",
      "lotSize": 1,
      "tickSize": 0.05
    }
  ],
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

#### `GET /api/v1/quotes?symbols=NSE:RELIANCE,NSE:TCS`

Status: `TODO`

Response:

```json
{
  "data": [
    {
      "symbol": "NSE:RELIANCE",
      "instrumentToken": 738561,
      "lastPrice": 3000.5,
      "netChange": 15.25,
      "percentChange": 0.51,
      "volumeTraded": 123456,
      "ohlc": {
        "open": 2990,
        "high": 3010,
        "low": 2980,
        "close": 2985
      },
      "updatedAt": "2026-05-06T09:15:01+05:30"
    }
  ],
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

#### `GET /api/v1/candles?instrumentToken=738561&interval=1m&from=2026-05-06T09:15:00+05:30&to=2026-05-06T15:30:00+05:30`

Status: `TODO`

Response:

```json
{
  "data": [
    {
      "instrumentToken": 738561,
      "symbol": "NSE:RELIANCE",
      "interval": "1m",
      "timestamp": "2026-05-06T09:15:00+05:30",
      "open": 3000,
      "high": 3005,
      "low": 2998,
      "close": 3002,
      "volume": 10000,
      "tradeCount": 50
    }
  ],
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

#### `GET /api/v1/market/overview`

Status: `TODO`

Response:

```json
{
  "data": {
    "indices": [],
    "topGainers": [],
    "topLosers": [],
    "mostActiveByVolume": [],
    "marketBreadth": {
      "advancers": 0,
      "decliners": 0,
      "unchanged": 0
    },
    "updatedAt": "2026-05-06T23:10:00+05:30"
  },
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

### Watchlist

#### `GET /api/v1/watchlists/default`

Status: `TODO`

Response:

```json
{
  "data": {
    "id": "wl_01HX...",
    "name": "Default",
    "items": [
      {
        "symbol": "NSE:RELIANCE",
        "instrumentToken": 738561,
        "addedAt": "2026-05-06T23:10:00+05:30"
      }
    ]
  },
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

#### `POST /api/v1/watchlists/default/items`

Status: `TODO`

Payload:

```json
{
  "symbol": "NSE:RELIANCE",
  "instrumentToken": 738561
}
```

Response:

```json
{
  "data": {
    "added": true
  },
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

### Orders and Positions

#### `GET /api/v1/positions`

Status: `TODO`

Response:

```json
{
  "data": [
    {
      "symbol": "NSE:RELIANCE",
      "instrumentToken": 738561,
      "product": "MIS",
      "quantity": 10,
      "averagePrice": 2980,
      "lastPrice": 3000,
      "realizedPnl": 0,
      "unrealizedPnl": 200,
      "updatedAt": "2026-05-06T23:10:00+05:30"
    }
  ],
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

#### `GET /api/v1/orders?status=open`

Status: `TODO`

Response:

```json
{
  "data": [
    {
      "id": "ord_01HX...",
      "brokerOrderId": "240506000001",
      "symbol": "NSE:RELIANCE",
      "instrumentToken": 738561,
      "transactionType": "BUY",
      "orderType": "MARKET",
      "product": "MIS",
      "quantity": 10,
      "price": 0,
      "triggerPrice": 0,
      "status": "OPEN",
      "placedAt": "2026-05-06T09:20:00+05:30"
    }
  ],
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

#### `POST /api/v1/orders`

Status: `TODO`

Required headers:

```http
Authorization: Bearer <jwt>
Content-Type: application/json
Idempotency-Key: ordreq_01HX...
```

Payload:

```json
{
  "symbol": "NSE:RELIANCE",
  "instrumentToken": 738561,
  "transactionType": "BUY",
  "orderType": "MARKET",
  "product": "MIS",
  "quantity": 10,
  "price": 0,
  "triggerPrice": 0,
  "validity": "DAY",
  "tag": "manual"
}
```

Response:

```json
{
  "data": {
    "id": "ord_01HX...",
    "brokerOrderId": "240506000001",
    "status": "PLACED",
    "placedAt": "2026-05-06T09:20:00+05:30"
  },
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

Trading safety tasks:
- Paper trading by default.
- Live trading requires broker connected, market open, risk checks, max loss checks, quantity checks, idempotency key, and audit event.
- Add kill switch.
- Add daily loss circuit breaker.

### Strategies

#### `GET /api/v1/strategies`

Status: `TODO`

Response:

```json
{
  "data": [
    {
      "name": "IntradayMomentum",
      "displayName": "Intraday Momentum",
      "status": "available",
      "supportedModes": ["Paper", "Live"],
      "parametersSchema": {}
    }
  ],
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

#### `PUT /api/v1/user/strategies/{strategyName}`

Status: `TODO`

Payload:

```json
{
  "enabled": true,
  "parameters": {
    "riskPerTrade": 1,
    "symbols": ["NSE:RELIANCE"],
    "timeframe": "1m"
  }
}
```

Response:

```json
{
  "data": {
    "saved": true,
    "strategyName": "IntradayMomentum",
    "enabled": true
  },
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

### Backtests

#### `POST /api/v1/backtests`

Status: `TODO`

Payload:

```json
{
  "strategyName": "IntradayMomentum",
  "symbols": ["NSE:RELIANCE"],
  "from": "2026-01-01T09:15:00+05:30",
  "to": "2026-05-01T15:30:00+05:30",
  "initialCapital": 100000,
  "fees": {
    "brokerageType": "Flat",
    "brokerageValue": 20,
    "slippagePercent": 0.1
  },
  "parameters": {}
}
```

Response:

```json
{
  "data": {
    "jobId": "bt_01HX...",
    "status": "queued"
  },
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

### ML Models

#### `GET /api/v1/models`

Status: `TODO`

Response:

```json
{
  "data": [
    {
      "id": "model_sentiment_optimized",
      "name": "Sentiment Optimized",
      "type": "sentiment",
      "path": "models/sentiment_optimized.onnx",
      "active": true,
      "version": "0.1.0",
      "metrics": {
        "accuracy": null,
        "latencyMs": null
      }
    }
  ],
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

### Sentiment and News

#### `GET /api/v1/news?symbol=NSE:RELIANCE&limit=50`

Status: `TODO`

Response:

```json
{
  "data": [
    {
      "id": "news_01HX...",
      "symbol": "NSE:RELIANCE",
      "source": "NewsAPI",
      "title": "Article title",
      "url": "https://example.com/article",
      "publishedAt": "2026-05-06T10:00:00+05:30",
      "sentimentScore": 0.42,
      "sentimentLabel": "Positive"
    }
  ],
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1",
    "pagination": {
      "limit": 50,
      "cursor": "",
      "nextCursor": "",
      "hasMore": false
    }
  }
}
```

### Notifications

#### `POST /api/v1/notifications/test`

Status: `TODO`

Payload:

```json
{
  "channel": "telegram",
  "message": "Test alert from TradingBot"
}
```

Response:

```json
{
  "data": {
    "sent": true,
    "channel": "telegram",
    "providerMessageId": "msg_01HX..."
  },
  "meta": {
    "requestId": "req_01HX...",
    "serverTime": "2026-05-06T23:10:00+05:30",
    "version": "v1"
  }
}
```

## Production Security Checklist

Status: `TODO`

Authentication and authorization:
- JWT access tokens with short expiry.
- Refresh token rotation.
- Session invalidation.
- Role-based access for admin endpoints.
- Optional MFA before public launch.
- Device/session list.

Password security:
- Argon2id or bcrypt.
- Per-password salt.
- Password strength validation.
- Rate-limited login.

Broker secret security:
- Encrypt API secret, access token, public token, Telegram bot token, WhatsApp credentials.
- Use envelope encryption.
- Master key from environment or managed KMS.
- Never log secrets.
- Never return secrets in API responses.
- Secret redaction middleware/logger.

Trading safety:
- Paper trading default.
- Live trading requires explicit user acknowledgement.
- Daily loss limit.
- Max order value.
- Max open positions.
- Kill switch.
- Idempotency key for order placement.
- Audit log every order request and broker response.
- Broker reconciliation worker.

Web/API security:
- Strict CORS allowlist.
- HTTPS only in production.
- Secure cookies if cookie sessions are used.
- Rate limits per IP and per user.
- Request size limits.
- JSON validation.
- Panic recovery that hides internals from clients.
- Security headers.
- CSRF protection for browser cookie flows.

Data and privacy:
- User data isolation by `user_id`.
- No public `/api/data/users`.
- Backups encrypted.
- PII minimization.
- Data deletion/export plan.
- Audit trails for account, settings, broker, order, and admin actions.

Deployment:
- Docker image without secrets baked in.
- Separate dev/staging/prod configs.
- Health/readiness endpoints.
- Structured logs.
- Metrics and alerts.
- DB migrations in deployment pipeline.
- TLS termination.
- Environment variable validation at boot.

International product readiness:
- Multi-currency design.
- Country/timezone stored per user.
- Broker adapter interface, not Zerodha-only business logic.
- Locale-aware date/time display.
- Compliance disclaimers.
- Terms of service and risk disclosures.
- Per-market trading calendar support.
- Feature flags for country/broker rollout.

Business/profit readiness:
- Subscription/billing module later.
- Usage limits by plan.
- Broker-independent strategy engine.
- Paper-trading onboarding.
- Auditability for user trust.
- Robust monitoring before any paid launch.

## Backend Issues

Status: `OPEN`

- Existing routes are not versioned.
- `cmd` currently has multiple `main()` functions in one package; move command files into `cmd/server`, `cmd/get-token`, and `cmd/heatmap-cli`.
- Existing routes are not authenticated.
- CORS currently allows all origins.
- `.env`, `.access_token`, and logs are modified in `Go-project`; review before commit.
- README mentions auth/encryption/routes that are not implemented yet.
- `cmd/main.go` requires Zerodha credentials even when market simulation is enabled.
- Symbol lists are hardcoded and duplicated.
- WebSocket events have inconsistent field casing.
- Frontend currently hardcodes `localhost:8000`.
- No settings persistence API.
- No watchlist API.
- No backtest implementation.
- No notification implementation.
- No billing/user plan design yet.

## Next Backend Tasks

1. Move command entrypoints into `cmd/server`, `cmd/get-token`, and `cmd/heatmap-cli`.
2. Run `go test ./...` and fix any compile issues after command structure changes.
3. Create `internal/contracts` response envelope structs. `DONE`
4. Create `internal/security` encryption and redaction utilities. `DONE` AES-GCM encryption and redaction utilities are in place; KMS integration remains a production enhancement.
5. Create `internal/middleware` request ID, recovery, logging, CORS, and auth placeholders.
6. Refactor `internal/server/routes.go` into versioned route registration.
7. Implement `GET /api/v1/health`.
8. Implement auth models/session tables if needed.
9. Implement settings migration and API.
10. Implement broker status/connect/callback/disconnect.
11. Implement watchlist and batch quotes.
12. Implement WebSocket hub and topic subscriptions.
13. Replace old `/ws`, `/ws/candles`, `/ws/indicators`, `/ws/heatmap` with compatible typed events.
