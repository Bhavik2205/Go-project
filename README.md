# Equity Trading Bot

A high-performance, fault-tolerant, multi-user intraday trading bot built with Go. Integrates with financial market APIs (Zerodha Kite Connect), leverages machine learning for sentiment analysis, and provides real-time insights via WebSockets.

---

## Table of Contents

- [Project Structure](#project-structure)
- [System Flow](#system-flow)
- [Tech Stack](#tech-stack)
- [Development Plan](#development-plan)
- [Getting Started](#getting-started)
  - [Docker Setup](#docker-setup)
  - [Database Setup](#database-setup)
  - [Running the Bot](#running-the-bot)
- [Database Migrations](#database-migrations)
- [Testing WebSockets](#testing-websockets)

---

## Project Structure

```
equity-trading-bot/
│
├── cmd/                          # Entry points for different application modes
│   ├── main.go                   # Main bot execution (live trading)
│   └── backtest.go               # Historical data testing / backtesting utility
│
├── internal/                     # Core internal application logic
│   ├── api/                      # Integrations with external APIs (broker, news)
│   │   ├── handlers/             # HTTP handlers for external API callbacks (e.g., Zerodha OAuth)
│   │   ├── zerodha.go            # Zerodha KiteConnect client implementation
│   │   ├── newsapi.go            # NewsAPI client
│   │   ├── marketwatch.go        # MarketWatch API client
│   │   ├── instruments.go        # Logic for managing tradable instruments (fetch, store)
│   │   └── ticker.go             # Real-time market data (WebSocket) handling
│   │
│   ├── auth/                     # User authentication and authorization module
│   │   ├── auth.go               # JWT generation, validation, user session management
│   │   └── handlers.go           # HTTP handlers for user login, signup, and broker OAuth
│   │
│   ├── db/                       # Database interactions layer
│   │   ├── postgres.go           # PostgreSQL client and ORM/SQL logic
│   │   ├── models.go             # Go structs representing database schema models
│   │   └── migrations/           # Database migration scripts (SQL files)
│   │
│   ├── data/                     # Data loading, scraping, and preprocessing
│   │   ├── ingest.go             # Logic for ingesting OHLCV + indicators into DB
│   │   └── preprocess.go         # Data cleaning and feature engineering
│   │
│   ├── model/                    # Machine Learning & Deep Learning model logic
│   │   ├── inference.go          # Logic for running model predictions (ONNX runtime)
│   │   ├── trainer.go            # Model training pipeline (Go orchestration, Python execution)
│   │   └── sentiment.go          # FinBERT / LLM sentiment model integration
│   │
│   ├── strategy/                 # Trading strategy definitions
│   │   ├── intraday.go           # Intraday trading strategy logic
│   │   ├── swing.go              # Swing trading strategy logic
│   │   ├── scalping.go           # Scalping strategy logic
│   │   └── selector.go           # Strategy selection based on market conditions
│   │
│   ├── execution/                # Order execution and monitoring logic
│   │   ├── order.go              # Logic for placing, modifying, and canceling orders
│   │   └── monitor.go            # Position/order monitoring, trailing stop, exit logic
│   │
│   ├── metrics/                  # Prometheus metrics definitions and collection
│   │   └── metrics.go            # Custom metrics for bot performance and health
│   │
│   ├── utils/                    # Common helper utilities
│   │   ├── config.go             # Configuration loading and management
│   │   ├── logger.go             # Centralized logging utility
│   │   └── errors.go             # Custom error types and handling
│   │
│   └── server/                   # HTTP/WebSocket server
│       ├── broadcast.go          # WebSocket server for real-time data push
│       └── routes.go             # Centralized routing for all HTTP/WebSocket handlers
│
├── configs/                      # YAML configuration files
│   ├── app.yaml                  # General application settings (ports, service names)
│   ├── zerodha.yaml              # Zerodha API configuration (non-sensitive: API key only)
│   ├── database.yaml             # Database connection details
│   ├── model.yaml                # Model-specific parameters
│   └── strategy.yaml             # Strategy parameters
│
├── models/                       # Saved ML models and ONNX files
│   └── sentiment.onnx            # Pre-trained sentiment analysis model
│
├── data/                         # Cached or downloaded datasets
│   ├── nse/                      # Historical NSE market data
│   └── news/                     # Cached news articles
│
├── scripts/                      # Utility scripts (Python, Shell)
│   ├── download_data.py          # Bulk historical data download
│   ├── retrain_sentiment.py      # Sentiment model retraining
│   ├── performance_report.py     # Backtest/live performance report generation
│   └── db_migrate.sh             # Database migration runner
│
├── notebooks/                    # Jupyter notebooks for EDA and experiments
│   └── EDA_intraday.ipynb        # Intraday data analysis example
│
├── .env                          # Environment variables (API keys, secrets — DO NOT COMMIT)
├── go.mod                        # Go module dependency file
├── requirements.txt              # Python dependency file
├── Dockerfile                    # Docker build instructions
├── docker-compose.yaml           # Docker Compose setup (bot, DB, Prometheus, Grafana)
└── README.md                     # This file
```

---

## System Flow

### 1. System Initialization & Configuration

At startup, `cmd/main.go` orchestrates the environment and essential services:

- Loads application settings from YAML config files via `internal/utils/config.go`
- Securely loads secrets (API keys, DB passwords, JWT signing keys) from environment variables — never hardcoded
- Initializes the logging system (`internal/utils/logger.go`) with configurable levels and output destinations
- Sets up Prometheus metrics (`internal/metrics/metrics.go`) for API latencies, trade executions, and WebSocket connections
- Establishes a PostgreSQL connection pool (`internal/db/postgres.go`)
- Optionally initializes a Redis client pool for high-speed caching and Pub/Sub messaging
- Launches the HTTP server and WebSocket server for REST APIs and real-time frontend updates
- Starts dedicated goroutines for long-running processes: market data ingestion, strategy execution loops, order monitoring, and news fetching

### 2. User Management & Secure API Key Handling

**Database Schema** (`internal/db/models.go`):

| Table | Purpose |
|---|---|
| `users` | Core user info (ID, hashed password, email) |
| `user_broker_accounts` | Per-user broker credentials; `access_token` is encrypted at rest |
| `user_strategies` | Which strategies each user has enabled, with config parameters |

**Authentication** (`internal/auth/`): Handles registration, secure password hashing, JWT generation/validation, and HTTP route middleware.

**Key Endpoints**:

| Endpoint | Method | Description |
|---|---|---|
| `/signup` | POST | New user registration |
| `/login` | POST | Authenticate user, issue JWT |
| `/user/broker/zerodha/connect` | GET | Initiate Zerodha OAuth flow |
| `/api/zerodha/callback` | GET | Receive OAuth token, encrypt and store |
| `/user/strategies` | GET/POST | Manage user strategies and parameters |
| `/user/dashboard` | WS | Real-time account updates |

### 3. Data Ingestion Pipeline

**Instrument Management** (`internal/api/instruments.go`): Periodically refreshes the tradable instrument list from Zerodha into the `instruments` table.

**Real-time Market Data** (`internal/api/ticker.go` + `internal/data/ingest.go`):

1. A goroutine pool manages WebSocket connections to Zerodha Kite Ticker per active user
2. Raw tick data is piped through a Go channel to `ingest.go`
3. Ticks are aggregated into OHLCV bars (e.g., 1-minute)
4. Technical indicators (SMA, RSI, MACD) are calculated
5. Processed data is persisted to the `market_data` TimescaleDB hypertable
6. Data is published to Redis Pub/Sub channels (`market_data:<instrument_id>`) for strategy consumption and frontend broadcasting
7. Robust error handling with retries and automatic WebSocket reconnection

**Historical Data**: `scripts/download_data.py` handles bulk downloads; `ingest.go` supports backfilling from Zerodha's historical API.

**News & Sentiment**: Dedicated goroutines fetch articles from NewsAPI and MarketWatch. `internal/model/sentiment.go` performs real-time FinBERT sentiment analysis. Results are stored in the `news_sentiment` table and published to Redis.

### 4. Strategy & Execution Engine

**Strategy Selection** (`internal/strategy/selector.go`): Dynamically selects the most appropriate strategy (intraday, swing, scalping) per user based on market conditions, time of day, and user settings.

**Strategy Execution** (`internal/strategy/*.go`):

1. Each active user's strategy runs in a dedicated goroutine
2. Goroutines subscribe to Redis Pub/Sub for real-time data with minimal latency
3. Entry/exit conditions are evaluated against latest data and user-defined parameters
4. Trade signals are passed to `internal/execution/order.go`

**Order Management** (`internal/execution/order.go`):

1. Receives trade signals (BUY/SELL, instrument, quantity, target, stop-loss)
2. Performs risk management checks: available capital, daily loss limits, position sizing, exposure limits
3. Constructs the order payload and sends it to Zerodha via `internal/api/zerodha.go`
4. Logs all order details to the `orders` table

**Position Monitoring** (`internal/execution/monitor.go`): A per-user goroutine monitors open orders and positions, manages trailing stops, handles end-of-day square-offs, and triggers closing orders when exit conditions are met.

### 5. Monitoring & Observability

| Component | Role |
|---|---|
| `internal/metrics/metrics.go` | Defines Prometheus metrics (goroutines, DB queries, API calls, order fills, P&L) |
| `GET /metrics` | Endpoint scraped by Prometheus |
| Prometheus (Docker) | Scrapes, stores, and aggregates time-series metrics |
| Grafana (Docker) | Dashboards for bot health, trade activity, P&L, and alerting |
| `internal/utils/logger.go` | Structured JSON logging for all critical events and errors |

### 6. Real-time Data Broadcasting

- The WebSocket server (`internal/server/broadcast.go`) authenticates clients via JWT and associates connections with `user_id`
- It subscribes to Redis Pub/Sub channels: `market_data:<instrument_id>`, `user_pnl:<user_id>`, `order_updates:<user_id>`
- Incoming Redis messages are filtered by `user_id` and pushed to the appropriate frontend client
- This decoupled architecture keeps real-time broadcasting independent of core bot logic

### 7. Fault Tolerance & Performance

- **Concurrency**: Go goroutines and channels for all independent tasks (per-user WebSocket, strategy loops, API polling)
- **Graceful Shutdown**: Signal handling (`SIGINT`, `SIGTERM`) flushes in-flight data and closes connections cleanly
- **Error Handling & Retries**: Exponential backoff for transient errors; circuit breakers for external services
- **Database Resilience**: Connection pooling, proper indexing on `market_data`, `orders`, `trades`, `positions`, and regular backups
- **Rate Limiting**: Client-side rate limiting on all Zerodha API calls
- **Idempotency**: Order placement designed to prevent duplicates on retry
- **Stateless Services**: State pushed to PostgreSQL/Redis for scalability and easier recovery
- **Containerization**: Docker + Docker Compose for consistent environments across dev and production

---

## Tech Stack

### Core
| Layer | Technology |
|---|---|
| Language | Go (Golang) |
| Web Framework | `net/http` stdlib (or Gin / Echo) |
| Concurrency | Goroutines & Channels |

### Data Storage
| Layer | Technology |
|---|---|
| Primary Database | PostgreSQL |
| Time-Series Extension | TimescaleDB |
| Cache / Message Broker | Redis |

### Machine Learning
| Layer | Technology |
|---|---|
| Training Framework | PyTorch / TensorFlow |
| Inference Format | ONNX (via `microsoft/onnxruntime v1.21.0`) |
| NLP Model | FinBERT (via Hugging Face Transformers) |
| Data Libraries | Pandas, NumPy, Scikit-learn, Matplotlib |
| EDA | Jupyter Notebooks |

### Integrations
| Integration | Technology |
|---|---|
| Broker API | Zerodha Kite Connect |
| News APIs | NewsAPI, MarketWatch |
| WebSocket Client | `gorilla/websocket` |

### Observability & Deployment
| Layer | Technology |
|---|---|
| Metrics | Prometheus |
| Dashboards & Alerts | Grafana |
| Logging | `zap` or `logrus` |
| Containerization | Docker + Docker Compose |
| Authentication | JWT |
| Encryption | Go `crypto` packages |

---

## Development Plan

| Phase | Description | Status | Start Date |
|---|---|---|---|
| 1 | Zerodha Integration | 🚧 In Progress | May 27, 2025 |
| 2 | Data Ingestion & Feature Engineering | ⏳ Planned | TBD |
| 3 | Strategy Framework | ⏳ Planned | TBD |
| 4 | ML/DL Price Forecasting | ⏳ Planned | TBD |
| 5 | Sentiment Analysis (FinBERT / NewsAPI) | 💤 Last Phase | TBD |
| 6 | Execution Engine | ⏳ Planned | TBD |
| 7 | Backtesting & Logging | ⏳ Planned | TBD |
| 8 | Deployment & Automation | 🧾 Final Polish | TBD |

---

## Getting Started

### Docker Setup

**1. Configure your `.env` file** in the project root:

```env
# PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_USER=admin
DB_PASSWORD=secret
DB_NAME=trading_bot_db

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=redispass
```

> **Note:** Never commit `.env` to version control. Values must use `KEY=value` format — no `export`, no quotes.

**2. Start all services:**

```bash
docker compose --env-file .env -f configs/docker-compose.yaml up -d
```

**3. Stop and remove containers + volumes:**

```bash
docker compose --env-file .env -f configs/docker-compose.yaml down -v
```

**4. Validate config (verify `.env` values are loading correctly):**

```bash
docker compose --env-file .env -f configs/docker-compose.yaml config
```

You should see actual values populated for `POSTGRES_DB`, `REDIS_PASSWORD`, etc.

---

### Database Setup

The `market_data` table uses the TimescaleDB extension. This must be enabled **before** running migrations.

#### Step 1: Enable TimescaleDB

```bash
psql -h localhost -p 5432 -U postgres -d trading_bot_db \
  -f internal/db/migrations/pre_migration_enable_timescaledb.sql
```

You will be prompted for your database password.

#### Step 2: Install the migration tool

```bash
go install -tags 'postgres' github.com/golang-migrate/migrate/v4/cmd/migrate@latest
```

#### Step 3: Export your database URL

```bash
export DATABASE_URL="postgres://<user>:<password>@<host>:<port>/<dbname>?sslmode=disable"

# Example:
export DATABASE_URL="postgres://postgres:admin@localhost:5432/trading_bot_db?sslmode=disable"
```

#### Step 4: Apply migrations

```bash
migrate -path internal/db/migrations -database "$DATABASE_URL" up
```

---

### Running the Bot

```bash
go run cmd/main.go 2>&1 | tee startup.log
```

---

## Database Migrations

The `scripts/db_migrate.sh` script wraps the `golang-migrate` CLI.

**Prerequisites:**
- `migrate` CLI installed (see above)
- PostgreSQL running and the target database created
- TimescaleDB extension enabled
- `.env` correctly configured with `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`
- Migration files present in `internal/db/migrations/`
- Script is executable: `chmod +x scripts/db_migrate.sh`

**Commands:**

```bash
# Apply all pending migrations
./scripts/db_migrate.sh up

# Apply next N migrations
./scripts/db_migrate.sh up 1

# Revert all migrations (WARNING: potential data loss)
./scripts/db_migrate.sh down

# Revert last N migrations
./scripts/db_migrate.sh down 1

# Create new migration files
./scripts/db_migrate.sh create <migration_name>

# Show current migration version
./scripts/db_migrate.sh version

# Force set version (emergency recovery only)
./scripts/db_migrate.sh force <N>
```

> **Important:** Once using this script, remove all `db.AutoMigrate()` calls from the Go application. Always back up your database before running migrations in production.

---

## Testing WebSockets

Test the WebSocket server locally using `wscat`:

```bash
# Install wscat
npm install -g wscat

# Connect to the WebSocket server
wscat -c ws://localhost:8000/ws
```