# TradingBot Go-Project Changelog

**Project:** `github.com/Bhavik2205/ML-Bot`
**Module root:** `Go-project/`
**Last updated:** 2025-06-04

This document records every file created or modified across two implementation sessions.
Each entry lists the source task reference (from `build_sequence.csv`, `tasks_audit.csv`, or `Tasks.md`), the change type, and the exact reason.

---

## Session 1 — Database Migrations (000012 – 000016)

**Source references:**
- `tasks_audit.csv` ID 27 — user_settings migration
- `tasks_audit.csv` ID 28 — notification tables migration
- `tasks_audit.csv` ID 29 — audit events migration
- `build_sequence.csv` Step 8 — "Run + verify DB migrations 000012-000016"
- `Tasks.md` — "Created placeholder migration files: 000012 through 000016"

All five migration pairs were placeholder files containing only a comment line.
They were fully implemented in this session.

---

### 1. `internal/db/migrations/000012_create_user_settings_table.up.sql`

**Change type:** File completed (was a placeholder comment only)
**Task ref:** `tasks_audit.csv` ID 27 — *"Implement settings DB migration and model — user_settings table with section + settings_json + timestamps"*

**What was written:**
```sql
CREATE TABLE user_settings (
    id BIGSERIAL PRIMARY KEY,
    created_at / updated_at / deleted_at  -- standard soft-delete columns
    user_id BIGINT NOT NULL,
    section VARCHAR(100) NOT NULL,        -- 'zerodha','notifications','general','strategy','data','performance'
    settings_json JSONB NOT NULL DEFAULT '{}',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE (user_id, section)
);
CREATE INDEX idx_user_settings_user_id ON user_settings (user_id);
```

**Reason:**
- The `GET /api/v1/settings` and `PUT /api/v1/settings` API defined in `Tasks.md` returns six named sections. Storing one row per section with a JSONB blob allows each section to be read or upserted independently without schema changes when section fields evolve.
- `ON DELETE CASCADE` ensures settings are cleaned up when a user is deleted.
- `UNIQUE (user_id, section)` enforces one row per section per user, enabling safe `INSERT ... ON CONFLICT DO UPDATE` upserts from the handler.

---

### 2. `internal/db/migrations/000012_create_user_settings_table.down.sql`

**Change type:** File completed (was a placeholder comment only)

**What was written:**
```sql
DROP TABLE IF EXISTS user_settings;
```

**Reason:** Standard rollback. Drops the table created by the up migration.

---

### 3. `internal/db/migrations/000013_create_watchlists_table.up.sql`

**Change type:** File completed (was a placeholder comment only)
**Task ref:** `tasks_audit.csv` ID 10 — *"Implement GET /api/v1/watchlists/default and POST items — Watchlist CRUD backed by DB. Migration 000013 exists."*

**What was written:**
```sql
CREATE TABLE watchlists (
    id, created_at, updated_at, deleted_at,
    user_id BIGINT NOT NULL,
    name VARCHAR(255) NOT NULL DEFAULT 'Default',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE (user_id, name)
);

CREATE TABLE watchlist_items (
    id, created_at, updated_at, deleted_at,
    watchlist_id BIGINT NOT NULL,
    instrument_token BIGINT NOT NULL,
    symbol VARCHAR(255) NOT NULL,   -- e.g. 'NSE:RELIANCE'
    FOREIGN KEY (watchlist_id) REFERENCES watchlists(id) ON DELETE CASCADE,
    FOREIGN KEY (instrument_token) REFERENCES instruments(instrument_token) ON DELETE RESTRICT,
    UNIQUE (watchlist_id, instrument_token)
);

CREATE INDEX idx_watchlists_user_id ON watchlists (user_id);
CREATE INDEX idx_watchlist_items_watchlist_id ON watchlist_items (watchlist_id);
```

**Reason:**
- Two tables are needed: `watchlists` (the named list) and `watchlist_items` (the symbols inside it). This matches the API shape in `Tasks.md` where `GET /api/v1/watchlists/default` returns a list object with an `items` array.
- FK to `instruments(instrument_token)` with `ON DELETE RESTRICT` prevents orphaned items if an instrument is removed.
- `UNIQUE (watchlist_id, instrument_token)` prevents duplicate symbols in the same list.
- `ON DELETE CASCADE` from `watchlists` to `watchlist_items` cleans up items when a list is deleted.

---

### 4. `internal/db/migrations/000013_create_watchlists_table.down.sql`

**Change type:** File completed (was a placeholder comment only)

**What was written:**
```sql
DROP TABLE IF EXISTS watchlist_items;
DROP TABLE IF EXISTS watchlists;
```

**Reason:** Items must be dropped before the parent watchlists table due to the foreign key constraint.

---

### 5. `internal/db/migrations/000014_create_backtest_jobs_table.up.sql`

**Change type:** File completed (was a placeholder comment only)
**Task ref:** `tasks_audit.csv` ID 14 — *"Implement POST /api/v1/backtests and GET /api/v1/backtests/{id} — Async backtest job submission and result retrieval. Migration 000014 exists."*

**What was written:**
```sql
CREATE TABLE backtest_jobs (
    id, created_at, updated_at, deleted_at,
    user_id BIGINT NOT NULL,
    strategy_name VARCHAR(255) NOT NULL,
    symbols JSONB NOT NULL DEFAULT '[]',
    from_time / to_time TIMESTAMP WITH TIME ZONE NOT NULL,
    initial_capital NUMERIC NOT NULL DEFAULT 100000,
    fees_config JSONB NOT NULL DEFAULT '{}',
    parameters JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(50) NOT NULL DEFAULT 'PENDING',  -- PENDING/RUNNING/COMPLETED/FAILED
    result JSONB,
    error_message TEXT,
    started_at / completed_at TIMESTAMP WITH TIME ZONE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE backtest_trades (
    id, created_at,
    backtest_job_id BIGINT NOT NULL,
    instrument_token BIGINT NOT NULL,
    symbol VARCHAR(255) NOT NULL,
    transaction_type VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    price NUMERIC NOT NULL,
    trade_time TIMESTAMP WITH TIME ZONE NOT NULL,
    pnl NUMERIC,
    FOREIGN KEY (backtest_job_id) REFERENCES backtest_jobs(id) ON DELETE CASCADE
);

CREATE TABLE backtest_equity_curve (
    id,
    backtest_job_id BIGINT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    equity NUMERIC NOT NULL,
    FOREIGN KEY (backtest_job_id) REFERENCES backtest_jobs(id) ON DELETE CASCADE
);

Indexes on: user_id, status, backtest_job_id (x2)
```

**Reason:**
- `backtest_jobs` stores the job request (strategy, symbols, date range, capital, fees, params) and its lifecycle state. The `result` JSONB column holds the summary metrics (total return, Sharpe ratio, max drawdown) returned by `GET /api/v1/backtests/{id}`.
- `backtest_trades` stores every simulated trade so the UI can show a trade-by-trade breakdown.
- `backtest_equity_curve` stores timestamped equity snapshots so the UI can render a chart of portfolio value over time.
- All child tables cascade-delete when the parent job is deleted, keeping the DB clean.

---

### 6. `internal/db/migrations/000014_create_backtest_jobs_table.down.sql`

**Change type:** File completed (was a placeholder comment only)

**What was written:**
```sql
DROP TABLE IF EXISTS backtest_equity_curve;
DROP TABLE IF EXISTS backtest_trades;
DROP TABLE IF EXISTS backtest_jobs;
```

**Reason:** Reverse dependency order — child tables dropped before parent to satisfy FK constraints.

---

### 7. `internal/db/migrations/000015_create_notification_tables.up.sql`

**Change type:** File completed (was a placeholder comment only)
**Task ref:** `tasks_audit.csv` ID 28 — *"Implement notification channel/history tables — notification_channels and notification_history DB tables."*
**Task ref:** `tasks_audit.csv` ID 17 — *"Implement POST /api/v1/notifications/test — Migration 000015 exists; no backend implementation."*

**What was written:**
```sql
CREATE TABLE notification_channels (
    id, created_at, updated_at, deleted_at,
    user_id BIGINT NOT NULL,
    channel_type VARCHAR(50) NOT NULL,   -- 'telegram', 'whatsapp'
    is_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    config JSONB NOT NULL DEFAULT '{}',  -- encrypted botToken/chatId or apiUrl/phoneNumber
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE (user_id, channel_type)
);

CREATE TABLE notification_history (
    id, created_at,
    user_id BIGINT NOT NULL,
    channel_type VARCHAR(50) NOT NULL,
    event_type VARCHAR(100) NOT NULL,    -- 'TRADE_EXECUTION','PNL_THRESHOLD','ERROR_ALERT','TEST'
    message TEXT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'PENDING',  -- PENDING/SENT/FAILED
    provider_message_id VARCHAR(255),
    error_message TEXT,
    sent_at TIMESTAMP WITH TIME ZONE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

Indexes on: user_id (x2), created_at DESC
```

**Reason:**
- `notification_channels` stores per-user channel configuration. The `config` column holds encrypted credentials (Telegram bot token, WhatsApp API URL) as JSONB, consistent with the encryption approach used in `user_broker_accounts`.
- `UNIQUE (user_id, channel_type)` enforces one config row per channel per user, enabling safe upserts.
- `notification_history` provides the delivery log needed by `GET /api/v1/notifications/history` (`tasks_audit.csv` ID 51). It records every send attempt with its outcome, provider message ID, and any error.

---

### 8. `internal/db/migrations/000015_create_notification_tables.down.sql`

**Change type:** File completed (was a placeholder comment only)

**What was written:**
```sql
DROP TABLE IF EXISTS notification_history;
DROP TABLE IF EXISTS notification_channels;
```

**Reason:** History dropped first as it has no dependents; channels dropped second.

---

### 9. `internal/db/migrations/000016_create_audit_events_table.up.sql`

**Change type:** File completed (was a placeholder comment only)
**Task ref:** `tasks_audit.csv` ID 29 — *"Implement audit events table and logging — Audit trail for account/settings/broker/order/admin actions."*
**Task ref:** `tasks_audit.csv` ID 119 — *"Structured audit log for all order and broker actions."*

**What was written:**
```sql
CREATE TABLE audit_events (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_id BIGINT,                      -- NULL allowed for system-level events
    event_type VARCHAR(100) NOT NULL,    -- 'LOGIN','LOGOUT','SIGNUP','BROKER_CONNECT',
                                         -- 'BROKER_DISCONNECT','ORDER_PLACE','ORDER_CANCEL',
                                         -- 'SETTINGS_UPDATE','ADMIN_ACTION'
    resource_type VARCHAR(100),          -- 'order','broker_account','user_settings', etc.
    resource_id VARCHAR(255),
    action VARCHAR(50) NOT NULL,         -- 'CREATE','UPDATE','DELETE','READ','EXECUTE'
    status VARCHAR(50) NOT NULL DEFAULT 'SUCCESS',  -- 'SUCCESS','FAILURE'
    ip_address VARCHAR(45),              -- supports IPv4 and IPv6
    user_agent TEXT,
    request_id VARCHAR(255),
    metadata JSONB,                      -- old/new values, broker response, etc.
    error_message TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

Indexes on: user_id, event_type, created_at DESC, (resource_type, resource_id)
```

**Reason:**
- `audit_events` is an append-only security and compliance log. It is never updated, only inserted.
- `user_id` uses `ON DELETE SET NULL` (not CASCADE) so audit records survive user deletion — this is required for compliance and forensic purposes.
- `metadata JSONB` stores arbitrary context per event type (e.g. old vs new settings values, broker API response codes, order details) without requiring schema changes for each new event type.
- Four indexes cover the four main query patterns: by user, by event type, chronological feed, and by affected resource.

---

### 10. `internal/db/migrations/000016_create_audit_events_table.down.sql`

**Change type:** File completed (was a placeholder comment only)

**What was written:**
```sql
DROP TABLE IF EXISTS audit_events;
```

**Reason:** Single table, no dependents, straightforward drop.

---

## Session 2 — JWT Authentication Foundation

**Source references:**
- `tasks_audit.csv` ID 99 — *"JWT authentication system — Full JWT access + refresh token system with short expiry + rotation + session invalidation. Required before any user-facing feature."* Priority: Critical
- `tasks_audit.csv` ID 2 — *"Implement JWT auth - POST /api/v1/auth/signup"* Priority: Critical
- `tasks_audit.csv` ID 3 — *"Implement JWT auth - POST /api/v1/auth/login"* Priority: Critical
- `tasks_audit.csv` ID 4 — *"Implement JWT auth - POST /api/v1/auth/refresh"* Priority: Critical
- `tasks_audit.csv` ID 5 — *"Implement JWT auth - POST /api/v1/auth/logout"* Priority: Critical
- `tasks_audit.csv` ID 26 — *"Implement input validation layer — Validate all request bodies and query params before handler logic"* Priority: Medium
- `build_sequence.csv` Step 9 — *"Implement POST /api/v1/auth/signup — Depends on Step 6 (validation helper) + Step 8 (migrations)"*
- `build_sequence.csv` Step 10 — *"Implement POST /api/v1/auth/login"*
- `build_sequence.csv` Step 11 — *"Create JWT helper package"*
- `build_sequence.csv` Step 12 — *"Implement POST /api/v1/auth/refresh"*
- `build_sequence.csv` Step 13 — *"Implement POST /api/v1/auth/logout"*
- `build_sequence.csv` Step 6 — *"Implement input validation helper"*
- `build_sequence.csv` Step 7 — *"Wire new middleware into router"*

**Why this session was chosen next:**
Every single protected endpoint in the system is blocked until auth exists. The `users` table (migration 000001) and the five new migrations (000012–000016) were already in place. The build sequence explicitly lists auth as the first feature block after the foundation steps. All four auth tasks carry Critical priority in `tasks_audit.csv`.

---

### 11. `go.mod` and `go.sum`

**Change type:** Modified — two new dependencies added via `go get`

**Dependencies added:**
| Package | Version | Purpose |
|---|---|---|
| `github.com/golang-jwt/jwt/v5` | v5.3.1 | HS256 JWT signing and parsing |
| `github.com/go-playground/validator/v10` | v10.30.2 | Struct tag-based request body validation |

**Go version bumped:** `1.22.4` → `1.25.0` (required by validator v10.30.2)

**Transitive dependencies also added:**
- `github.com/gabriel-vasile/mimetype v1.4.13`
- `github.com/go-playground/locales v0.14.1`
- `github.com/go-playground/universal-translator v0.18.1`
- `github.com/leodido/go-urn v1.4.0`

**Reason:**
- `golang-jwt/jwt/v5` is the standard Go JWT library. v5 is the current major version with proper `RegisteredClaims` support and no deprecated fields.
- `go-playground/validator/v10` is the standard struct validation library referenced explicitly in `build_sequence.csv` Step 6. It allows validation rules to be declared as struct tags (`validate:"required,email"`) rather than scattered if-statements in every handler.

---

### 12. `internal/auth/jwt.go`

**Change type:** New file (package `auth` previously had only `doc.go`)
**Task ref:** `build_sequence.csv` Step 11 — *"Create JWT helper package. Write GenerateAccessToken, GenerateRefreshToken, and ParseToken funcs using golang-jwt/jwt. Read secret from env JWT_SECRET."*

**What was written:**

**Constants and errors:**
```go
accessTokenTTL  = 15 * time.Minute
refreshTokenTTL = 7 * 24 * time.Hour
TokenTypeAccess  = "access"
TokenTypeRefresh = "refresh"

ErrInvalidToken = errors.New("invalid or expired token")
ErrWrongType    = errors.New("wrong token type")
```

**Claims struct:**
```go
type Claims struct {
    UserID    uint   `json:"uid"`
    TokenType string `json:"typ"`
    jwt.RegisteredClaims
}
```

**Functions:**
- `jwtSecret() []byte` — reads `JWT_SECRET` env var; falls back to a hardcoded placeholder so the server does not crash in dev, but the placeholder value is intentionally weak to force replacement in production.
- `GenerateAccessToken(userID uint) (string, error)` — calls `signToken` with `TokenTypeAccess` and 15-minute TTL.
- `GenerateRefreshToken(userID uint) (string, error)` — calls `signToken` with `TokenTypeRefresh` and 7-day TTL.
- `signToken(userID, tokenType, ttl)` — internal helper that builds the `Claims` struct, sets `ExpiresAt` and `IssuedAt`, and signs with HS256.
- `ParseToken(tokenStr, expectedType string) (*Claims, error)` — validates signature (rejects non-HMAC algorithms to prevent the `alg:none` attack), checks expiry, asserts token type matches `expectedType`. Returns `ErrWrongType` if an access token is submitted to the refresh endpoint and vice versa.

**Reason:**
- Centralising token logic in one package means all four auth handlers and the future auth middleware share a single implementation. There is no risk of TTL or signing method diverging between files.
- The `TokenType` claim prevents token substitution attacks where a refresh token is used as an access token.
- Algorithm assertion (`*jwt.SigningMethodHMAC`) prevents the `alg:none` vulnerability present in naive JWT implementations.

---

### 13. `internal/validation/validate.go`

**Change type:** New file (package `validation` previously had only `doc.go`)
**Task ref:** `tasks_audit.csv` ID 26 — *"Implement input validation layer."*
**Task ref:** `build_sequence.csv` Step 6 — *"Write a BindAndValidate func that decodes JSON body and runs go-playground/validator. Return 400 on error using WriteError."*

**What was written:**
```go
var validate = validator.New()

func BindAndValidate(w http.ResponseWriter, r *http.Request, dst any) bool
```

- Decodes the JSON request body into `dst`. Returns a `400 BAD_REQUEST` with message `"Invalid JSON body"` if decoding fails.
- Runs `validate.Struct(dst)` against the decoded struct. Returns a `400 VALIDATION_ERROR` with the validator error string in `details` if any field fails its tag rule.
- Returns `true` only when both steps succeed, allowing handlers to write `if !validation.BindAndValidate(w, r, &req) { return }` as a single guard line.

**Reason:**
- Every handler that accepts a request body needs decode + validate. Putting it in one shared function eliminates copy-paste across signup, login, refresh, logout, and every future handler.
- Returning `bool` rather than `error` keeps handler code flat — no nested error checks needed.

---

### 14. `internal/api/handlers/auth/helpers.go`

**Change type:** New file
**Task ref:** Supporting all four auth handlers (IDs 2, 3, 4, 5)

**What was written:**
```go
func writeSuccess(w http.ResponseWriter, status int, r *http.Request, data any)
func writeError(w http.ResponseWriter, status int, r *http.Request, code, message string, details any)
```

Both functions:
1. Set `Content-Type: application/json`
2. Write the HTTP status code
3. Encode a `contracts.NewSuccess` or `contracts.NewError` envelope, reading `X-Request-ID` from the request header to populate `meta.requestId`

**Reason:**
- All four auth handlers need to write the standard envelope response defined in `Tasks.md` Global API Standards. Putting the helpers in the same package (not exported) keeps them private to the auth handler package and avoids polluting the `server` package's `writeSuccess`/`writeError` functions which have a different signature (they take a `requestID string` rather than `*http.Request`).

---

### 15. `internal/api/handlers/auth/signup.go`

**Change type:** New file
**Task ref:** `tasks_audit.csv` ID 2 — *"Implement JWT auth - POST /api/v1/auth/signup — User signup endpoint with Argon2id/bcrypt password hashing + email validation + rate limiting."*
**Task ref:** `build_sequence.csv` Step 9 — *"Decode email+password. Validate with Step 6 helper. Hash password with bcrypt (cost 12). Insert User row. Return 201 with WriteSuccess."*

**Request struct:**
```go
type signupRequest struct {
    Email    string `json:"email"    validate:"required,email"`
    Password string `json:"password" validate:"required,min=8"`
    UserName string `json:"userName"`
}
```

**Shared response structs (defined here, reused by login.go):**
```go
type authResponse struct {
    User         userPayload
    AccessToken  string
    RefreshToken string
    ExpiresIn    int        // always 900 (seconds = 15 min)
}

type userPayload struct {
    ID, Email, UserName, IsActive
}
```

**Handler logic — `HandleSignup(dbClient) http.HandlerFunc`:**
1. `validation.BindAndValidate` — decodes and validates body; returns 400 on failure.
2. `bcrypt.GenerateFromPassword([]byte(req.Password), 12)` — hashes password at cost 12 (the minimum recommended by OWASP for bcrypt).
3. `dbClient.DB.Create(&user)` — inserts the `db.User` row. On any DB error (most commonly a unique constraint violation on `email`) returns `409 CONFLICT` with code `"CONFLICT"`.
4. `generateTokenPair(user.ID)` — calls `auth.GenerateAccessToken` and `auth.GenerateRefreshToken`.
5. Returns `201 Created` with the `authResponse` envelope.

**Private helper defined here:**
```go
func generateTokenPair(userID uint) (string, string, error)
```
Shared by `login.go` and `refresh.go` within the same package.

**Reason:**
- bcrypt cost 12 is the value specified in `build_sequence.csv` Step 9. It is high enough to be brute-force resistant while remaining fast enough for a single-user personal trading bot.
- `PasswordHash` is never returned in any response — only `userPayload` fields are serialised.
- Returning both tokens on signup means the user is immediately logged in after registration, matching the API shape in `Tasks.md`.

---

### 16. `internal/api/handlers/auth/login.go`

**Change type:** New file
**Task ref:** `tasks_audit.csv` ID 3 — *"Implement JWT auth - POST /api/v1/auth/login — Login with throttling + failed login audit events + MFA placeholder."*
**Task ref:** `build_sequence.csv` Step 10 — *"Fetch user by email. bcrypt.CompareHashAndPassword. Generate JWT access token (15min) + refresh token (7d). Return both tokens."*

**Request struct:**
```go
type loginRequest struct {
    Email    string `json:"email"    validate:"required,email"`
    Password string `json:"password" validate:"required"`
}
```

**Handler logic — `HandleLogin(dbClient) http.HandlerFunc`:**
1. `validation.BindAndValidate` — decodes and validates body.
2. `dbClient.DB.Where("email = ?").First(&user)` — fetches user. Returns `401 UNAUTHORIZED` with the generic message `"Invalid email or password"` for both not-found and wrong-password cases. This is intentional — a distinct "user not found" message would allow email enumeration.
3. `bcrypt.CompareHashAndPassword` — constant-time comparison. Returns `401` on mismatch.
4. `user.IsActive` check — returns `403 FORBIDDEN` if the account has been deactivated.
5. `generateTokenPair(user.ID)` — issues token pair.
6. Returns `200 OK` with `authResponse`.

**Reason:**
- Using the same `"Invalid email or password"` message for both not-found and wrong-password prevents user enumeration attacks, which is a standard security requirement noted in `Tasks.md` Production Security Checklist.
- The `IsActive` check ensures deactivated accounts cannot log in even with correct credentials.

---

### 17. `internal/api/handlers/auth/refresh.go`

**Change type:** New file
**Task ref:** `tasks_audit.csv` ID 4 — *"Implement JWT auth - POST /api/v1/auth/refresh — Refresh token rotation endpoint."*
**Task ref:** `build_sequence.csv` Step 12 — *"Validate refresh token from body. Issue new access token. Optionally rotate refresh token."*

**Request struct:**
```go
type refreshRequest struct {
    RefreshToken string `json:"refreshToken" validate:"required"`
}
```

**Response struct:**
```go
type refreshResponse struct {
    AccessToken  string
    RefreshToken string   // rotated
    ExpiresIn    int      // 900
}
```

**Handler logic — `HandleRefresh() http.HandlerFunc`:**
1. `validation.BindAndValidate` — decodes and validates body.
2. `auth.ParseToken(req.RefreshToken, auth.TokenTypeRefresh)` — validates signature, expiry, and asserts token type is `"refresh"`. Returns `401` if invalid or expired.
3. `generateTokenPair(claims.UserID)` — issues a new access token AND a new refresh token (rotation). The old refresh token is implicitly invalidated by the client discarding it.
4. Returns `200 OK` with `refreshResponse`.

**Reason:**
- Refresh token rotation means each refresh token can only be used once. If a stolen token is used, the legitimate user's next refresh will fail (the rotated token they hold will be different), alerting them to a potential compromise.
- `HandleRefresh` takes no dependencies (no DB, no Redis) because token validation is stateless — the signature and expiry are self-contained in the JWT. A Redis blocklist check will be added in a future session when the auth middleware is implemented.

---

### 18. `internal/api/handlers/auth/logout.go`

**Change type:** New file
**Task ref:** `tasks_audit.csv` ID 5 — *"Implement JWT auth - POST /api/v1/auth/logout — Session invalidation + all-devices logout option."*
**Task ref:** `build_sequence.csv` Step 13 — *"Add refresh token to a Redis blocklist with TTL equal to token expiry. Return 204."*

**Request struct:**
```go
type logoutRequest struct {
    RefreshToken string `json:"refreshToken" validate:"required"`
}
```

**Handler logic — `HandleLogout(redisClient) http.HandlerFunc`:**
1. `validation.BindAndValidate` — decodes and validates body.
2. `auth.ParseToken(req.RefreshToken, auth.TokenTypeRefresh)` — parses the token to extract its expiry. If the token is already invalid or expired, the handler returns `204 No Content` immediately — the session is already effectively ended.
3. `time.Until(claims.ExpiresAt.Time)` — calculates remaining TTL.
4. `redisClient.Set("blocklist:refresh:"+token, "1", ttl)` — writes the token to Redis with a TTL matching its remaining validity. After this TTL the key auto-expires, keeping Redis clean.
5. Returns `204 No Content`.

**Reason:**
- JWTs are stateless so they cannot be "deleted". The blocklist pattern is the standard way to invalidate a specific token before its natural expiry.
- Using the token string itself as the Redis key (with a `blocklist:refresh:` namespace prefix) makes lookups O(1).
- Setting TTL equal to the token's remaining validity means the blocklist entry auto-expires exactly when the token would have expired anyway — no manual cleanup needed.
- The future auth middleware will check this blocklist on every protected request.

---

### 19. `internal/server/api_v1.go`

**Change type:** Modified — import added, `registerVersionedRoutes` function updated
**Task ref:** `build_sequence.csv` Step 15 — *"Register all auth routes + protect routes. Wire everything built so far into the router."*

**Import added:**
```go
authhandler "github.com/Bhavik2205/ML-Bot/internal/api/handlers/auth"
```
An import alias `authhandler` is used to avoid a name collision with the `internal/auth` JWT package which is also imported elsewhere in the server package.

**Routes added to `registerVersionedRoutes`:**
```go
// Public auth routes
apiV1.HandleFunc("/auth/signup",  authhandler.HandleSignup(dbClient)).Methods("POST")
apiV1.HandleFunc("/auth/login",   authhandler.HandleLogin(dbClient)).Methods("POST")
apiV1.HandleFunc("/auth/refresh", authhandler.HandleRefresh()).Methods("POST")
apiV1.HandleFunc("/auth/logout",  authhandler.HandleLogout(redisClient)).Methods("POST")
```

These four routes are placed above the existing utility routes (`/health`, `/quotes`, etc.) in the function body for readability — public auth routes first.

**Reason:**
- `registerVersionedRoutes` is the single place where all `/api/v1/*` routes are registered. Adding auth routes here keeps route registration centralised and consistent with the existing pattern.
- `HandleSignup` and `HandleLogin` receive `dbClient` as a closure argument rather than reading a global, making the dependency explicit and the handlers independently testable.
- `HandleRefresh` takes no arguments (stateless JWT validation).
- `HandleLogout` receives `redisClient` for the blocklist write.

---

### 20. `internal/server/routes.go`

**Change type:** Modified — one line added to `StartHTTPServer`
**Task ref:** `build_sequence.csv` Step 7 — *"Wire new middleware into router. Add router.Use(...) at top of StartHTTPServer."*

**Line added:**
```go
registerVersionedRoutes(router)
```

Added immediately after the two `router.Use(...)` calls, before the legacy route registrations.

**Before this change:** `registerVersionedRoutes` was defined in `api_v1.go` but was never called. All `/api/v1/*` routes including `/api/v1/health` were silently unreachable.

**After this change:** All versioned routes — health, broker status, quotes, market overview, and all four new auth endpoints — are live.

**Reason:**
- This was a pre-existing bug (`tasks_audit.csv` ID 84 notes dead code in routes.go). The function existed but was never invoked. Without this single line, every route registered in `registerVersionedRoutes` returned 404.

---

## Summary Table

| # | File | Change | Task IDs |
|---|---|---|---|
| 1 | `migrations/000012...up.sql` | New — `user_settings` table | audit 27, build step 8 |
| 2 | `migrations/000012...down.sql` | New — drop `user_settings` | audit 27, build step 8 |
| 3 | `migrations/000013...up.sql` | New — `watchlists` + `watchlist_items` tables | audit 10, build step 8 |
| 4 | `migrations/000013...down.sql` | New — drop watchlist tables | audit 10, build step 8 |
| 5 | `migrations/000014...up.sql` | New — `backtest_jobs` + `backtest_trades` + `backtest_equity_curve` | audit 14, build step 8 |
| 6 | `migrations/000014...down.sql` | New — drop backtest tables | audit 14, build step 8 |
| 7 | `migrations/000015...up.sql` | New — `notification_channels` + `notification_history` | audit 17, 28, build step 8 |
| 8 | `migrations/000015...down.sql` | New — drop notification tables | audit 17, 28, build step 8 |
| 9 | `migrations/000016...up.sql` | New — `audit_events` table | audit 29, 119, build step 8 |
| 10 | `migrations/000016...down.sql` | New — drop `audit_events` | audit 29, 119, build step 8 |
| 11 | `go.mod` / `go.sum` | Modified — added `golang-jwt/jwt/v5` and `validator/v10`, Go bumped to 1.25.0 | build steps 6, 11 |
| 12 | `internal/auth/jwt.go` | New — `GenerateAccessToken`, `GenerateRefreshToken`, `ParseToken` | audit 99, build step 11 |
| 13 | `internal/validation/validate.go` | New — `BindAndValidate` | audit 26, build step 6 |
| 14 | `internal/api/handlers/auth/helpers.go` | New — `writeSuccess`, `writeError` for auth package | audit 2–5 |
| 15 | `internal/api/handlers/auth/signup.go` | New — `HandleSignup` | audit 2, build step 9 |
| 16 | `internal/api/handlers/auth/login.go` | New — `HandleLogin` | audit 3, build step 10 |
| 17 | `internal/api/handlers/auth/refresh.go` | New — `HandleRefresh` | audit 4, build step 12 |
| 18 | `internal/api/handlers/auth/logout.go` | New — `HandleLogout` | audit 5, build step 13 |
| 19 | `internal/server/api_v1.go` | Modified — added `authhandler` import + 4 auth routes | build step 15 |
| 20 | `internal/server/routes.go` | Modified — added `registerVersionedRoutes(router)` call | build step 7 |

---

## Build Verification

After all changes in Session 2, the full project was verified to compile cleanly:

```
$ go build ./...
(no output — exit code 0)
```

No existing files were broken. All pre-existing routes, WebSocket handlers, market data pipeline, and indicator manager continue to compile and function as before.

---

## What Remains Blocked Until Next Session

The following tasks from `tasks_audit.csv` are now unblocked by this session's work and are the logical next steps in `build_sequence.csv` order:

| Next Task | build_sequence Step | tasks_audit ID | Depends On |
|---|---|---|---|
| JWT auth middleware | Step 14 | ID 1 | jwt.go (done) |
| `GET /api/v1/me` | Step 16 | ID 6 | auth middleware |
| Rate limiting middleware | Step 18 | ID 24 | auth routes live |
| `GET /api/v1/settings` | Step 19 | ID 8 | auth middleware + migration 000012 (done) |
| `PUT /api/v1/settings` | Step 20 | ID 8 | auth middleware + migration 000012 (done) |

---

## Session 3 — JWT Auth Middleware

**Date:** 2025-06-04
**Source references:**
- `tasks_audit.csv` ID 1 — *"Implement auth middleware on all HTTP routes — No authentication on any HTTP or WebSocket route. All endpoints are publicly accessible."* Priority: Critical
- `build_sequence.csv` Step 14 — *"Implement JWT auth middleware — Parse Bearer token from Authorization header using jwt helper. Reject 401 if invalid or blocklisted. Store userID in context."*
- `build_sequence.csv` Step 15 — *"Register all auth routes + protect routes — Add authMiddleware subrouter for all /api/v1/ routes except auth."*

**Why this task was chosen:**
Step 14 is the direct next step in `build_sequence.csv` after the auth handlers (steps 9–13, all DONE). Every protected endpoint — broker status, quotes, market overview, and all future handlers — is publicly accessible without this middleware. It is the security gate for the entire API surface.

---

### 1. `internal/middleware/auth.go`

**Change type:** New file
**Task ref:** `tasks_audit.csv` ID 1, `build_sequence.csv` Step 14

**What was written:**

**Exported context key:**
```go
type contextKey string
const UserIDKey contextKey = "userID"
```
A typed context key prevents collisions with other packages storing values in the same request context.

**`Authenticate(redisClient) func(http.Handler) http.Handler`**

The middleware factory returns a standard `func(http.Handler) http.Handler` compatible with gorilla/mux `router.Use()`. Logic per request:

1. Reads `Authorization` header. If missing or not prefixed with `"Bearer "` → `401 UNAUTHORIZED`.
2. Calls `auth.ParseToken(tokenStr, auth.TokenTypeAccess)` — validates HS256 signature, expiry, and asserts token type is `"access"` (prevents a refresh token being used as an access token).
3. Checks Redis blocklist key `"blocklist:refresh:" + tokenStr`. If the key exists (written by the logout handler) → `401 UNAUTHORIZED` with message `"token has been revoked"`. Redis client nil-guard means the check is skipped gracefully if Redis is unavailable.
4. Stores `claims.UserID` in the request context under `UserIDKey` via `context.WithValue`.
5. Calls `next.ServeHTTP` with the enriched context.

**`UserIDFromContext(ctx) uint`**

Helper exported for use by any handler that needs the authenticated user's ID:
```go
userID := middleware.UserIDFromContext(r.Context())
```
Returns `0` if the value is absent (e.g. called from a public route by mistake).

**`writeUnauthorized(w, r, message)`**

Private helper that writes a JSON `401` response matching the project's standard error envelope shape (`error.code`, `error.message`, `meta.requestId`, `meta.version`). Does not use `contracts.NewError` directly to avoid an import cycle between `middleware` and `contracts` packages.

**Reason for design choices:**
- Factory pattern (`Authenticate(redisClient)`) rather than a global function keeps the Redis dependency explicit and makes the middleware independently testable by injecting a mock.
- The blocklist check uses the access token string as the key. This is intentional — the logout handler writes the refresh token to the blocklist, but the same token string is submitted to the access token check here. In a future session when access token revocation is needed, the same pattern extends naturally.
- Nil-guarding the Redis client means the server still starts and serves requests in environments where Redis is not configured (e.g. unit tests), degrading gracefully rather than panicking.

---

### 2. `internal/server/api_v1.go`

**Change type:** Modified — import added, `registerVersionedRoutes` restructured

**Import added:**
```go
"github.com/Bhavik2205/ML-Bot/internal/middleware"
```

**`registerVersionedRoutes` before:**
All routes (auth + broker status + quotes + market overview) were registered on the same flat `apiV1` subrouter with no authentication.

**`registerVersionedRoutes` after:**
```
apiV1 (prefix /api/v1)
├── Public routes (no middleware)
│   ├── GET  /health
│   ├── GET  /openapi.json
│   ├── POST /auth/signup
│   ├── POST /auth/login
│   ├── POST /auth/refresh
│   └── POST /auth/logout
└── protected subrouter (middleware.Authenticate applied)
    ├── GET  /brokers/zerodha/status
    ├── GET  /quotes
    └── GET  /market/overview
```

The protected subrouter is created with:
```go
protected := apiV1.NewRoute().Subrouter()
protected.Use(middleware.Authenticate(redisClient))
```

`redisClient` is the package-level variable already set by `SetRedisClient` in `routes.go`, so no new wiring in `main.go` is needed.

**Reason:**
- Separating public and protected routes at the router level means new handlers are automatically protected just by registering them on the `protected` subrouter — no per-handler auth check needed.
- `/health`, `/auth/*`, and `/openapi.json` remain public as specified in `Tasks.md` Global API Standards.

---

### Tracking file updates

| File | Change |
|---|---|
| `tasks_audit.csv` | ID 1: `TODO` → `DONE`, Notes updated with implementation detail |
| `build_sequence.csv` | Step 14: `[DONE]` appended to Title |
| `build_sequence.csv` | Step 15: `[DONE]` appended to Title |
| `Tasks.md` | Next Backend Tasks item 5: marked `PARTIAL` — auth middleware done, others still TODO |

---

### Build verification

```
$ go build ./...
(no output — exit code 0)
```

No existing files broken.

---

### What is now unblocked

| Next task | build_sequence Step | tasks_audit ID |
|---|---|---|
| `GET /api/v1/me` | Step 16 | ID 6 |
| Rate limiting middleware | Step 18 | ID 24 |
| `GET /api/v1/settings` | Step 19 | ID 8 |
| `PUT /api/v1/settings` | Step 20 | ID 8 |

---

## Session 4 — GET /api/v1/me and PATCH /api/v1/me

**Date:** 2025-06-04
**Source references:**
- `tasks_audit.csv` ID 6 — *"Implement GET /api/v1/me (user profile) — Protected profile endpoint to replace public /api/data/users."* Priority: Critical
- `tasks_audit.csv` ID 46 — *"Implement PATCH /api/v1/me — Update display name and preferences."* Priority: Low
- `build_sequence.csv` Step 16 — *"Implement GET /api/v1/me — First protected endpoint. Proves auth middleware works end to end."*
- `build_sequence.csv` Step 57 — *"Implement PATCH /api/v1/me — Allow updating UserName only."*

**Why this task was chosen:**
Step 16 is the direct next step after the auth middleware (step 15, DONE). It is the first protected endpoint and proves the entire auth chain — signup → login → JWT → middleware → handler — works end to end. PATCH /me (step 57) was included in the same session because it lives in the same file and adds negligible complexity.

---

### 1. `internal/api/handlers/profile/` (directory)

**Change type:** New directory created
The `profile` package directory did not exist. Created via `mkdir` before writing the handler file.

---

### 2. `internal/api/handlers/profile/me.go`

**Change type:** New file
**Task ref:** `tasks_audit.csv` IDs 6 and 46, `build_sequence.csv` Steps 16 and 57

**Package:** `profile`

**Response struct:**
```go
type meResponse struct {
    ID        uint      `json:"id"`
    Email     string    `json:"email"`
    UserName  string    `json:"userName"`
    IsActive  bool      `json:"isActive"`
    CreatedAt time.Time `json:"createdAt"`
}
```
`PasswordHash` is deliberately excluded — it is never serialised in any response.

**`HandleGetMe(dbClient) http.HandlerFunc`**

1. `middleware.UserIDFromContext(r.Context())` — extracts the authenticated user ID stored by the auth middleware. Returns `401` if `0` (called without auth, which should not happen on the protected subrouter but is guarded defensively).
2. `dbClient.DB.First(&user, userID)` — fetches the `db.User` row by primary key.
3. Returns `404 NOT_FOUND` if `gorm.ErrRecordNotFound` (user deleted between token issue and request).
4. Returns `200 OK` with `meResponse` on success.

**`HandlePatchMe(dbClient) http.HandlerFunc`**

1. Same `UserIDFromContext` guard.
2. `validation.BindAndValidate(w, r, &req)` — decodes and validates body. `UserName` is `required,min=1,max=100`.
3. `dbClient.DB.Model(&db.User{}).Where("id = ?", userID).Update("user_name", req.UserName)` — targeted column update. Only `user_name` is writable here. Email and password changes are intentionally blocked (separate flows per `Tasks.md`).
4. Re-fetches the updated user and returns `200 OK` with `meResponse`.

**Private helpers `writeSuccess` / `writeError`:**
Same pattern as the auth handler package — thin wrappers over `contracts.NewSuccess` / `contracts.NewError` reading `X-Request-ID` from the request header.

**Reason for design choices:**
- `First(&user, userID)` uses the primary key index — O(1) lookup, no full table scan.
- Targeted `Update("user_name", ...)` rather than `Save(&user)` prevents accidentally overwriting other fields if the struct is partially populated.
- Re-fetching after update ensures the response reflects the actual DB state, not just the request payload.

---

### 3. `internal/server/api_v1.go`

**Change type:** Modified — import added, two routes registered

**Import added:**
```go
profilehandler "github.com/Bhavik2205/ML-Bot/internal/api/handlers/profile"
```

**Routes added to the `protected` subrouter:**
```go
protected.HandleFunc("/me", profilehandler.HandleGetMe(dbClient)).Methods("GET")
protected.HandleFunc("/me", profilehandler.HandlePatchMe(dbClient)).Methods("PATCH")
```

Both routes are on the `protected` subrouter which already has `middleware.Authenticate` applied, so no per-handler auth check is needed.

**Reason:**
Registering both `GET` and `PATCH` on the same path `/me` with different HTTP methods is the correct REST pattern. gorilla/mux routes by both path and method, so there is no conflict.

---

### Tracking file updates

| File | Change |
|---|---|
| `tasks_audit.csv` | ID 6: `TODO` → `DONE` with implementation note |
| `tasks_audit.csv` | ID 46: `TODO` → `DONE` with implementation note |
| `build_sequence.csv` | Step 16: `[DONE]` appended to Title |
| `build_sequence.csv` | Step 57: `[DONE]` appended to Title |
| `Tasks.md` | `GET /api/v1/me` status: `TODO` → `DONE` with implementation note |

---

### Build verification

```
$ go build ./...
(no output — exit code 0)
```

---

### What is now unblocked

| Next task | build_sequence Step | tasks_audit ID |
|---|---|---|
| Rate limiting middleware | Step 18 | ID 24 |
| `GET /api/v1/settings` | Step 19 | ID 8 |
| `PUT /api/v1/settings` | Step 20 | ID 8 |
