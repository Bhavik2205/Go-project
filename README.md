
# Equity Trading Bot

This repository contains the backend for a high-performance, fault-tolerant, and multi-user intraday trading bot built with Go. It integrates with financial market APIs (like Zerodha Kite Connect), leverages machine learning for sentiment analysis, and provides real-time insights via WebSockets.

---

## Project Structure

Our project follows a clear and modular structure to maintain scalability and readability.

```
equity-trading-bot/
│
├── cmd/                          # Entry points for different application modes
│   ├── main.go                   # Main bot execution (live trading)
│   └── backtest.go               # Historical data testing / backtesting utility
│
├── internal/                     # Core internal application logic
│   ├── api/                      # Integrations with external APIs (broker, news)
|   |   |── handlers/             # HTTP handlers for external API callbacks (e.g., Zerodha OAuth)
│   │   ├── zerodha.go            # Zerodha KiteConnect client implementation
│   │   ├── newsapi.go            # NewsAPI client
│   │   ├── marketwatch.go        # MarketWatch API client
|   |   ├── instruments.go        # Logic for managing tradable instruments (fetch, store)
|   |   ├── ticker.go             # Real-time market data (WebSocket) handling
│   │
│   ├── auth/                     # User authentication and authorization module
│   │   ├── auth.go               # JWT generation, validation, user session management
│   │   └── handlers.go           # HTTP handlers for user login, signup, and broker OAuth
│   │
│   ├── db/                       # Database interactions layer
│   │   ├── postgres.go           # PostgreSQL client and ORM/SQL logic
│   │   ├── models.go             # Go structs representing database schema models
│   │   └── migrations/           # Database migration scripts (SQL files, Go migration tools)
│   │
│   ├── data/                     # Data loading, scraping, and preprocessing
│   │   ├── ingest.go             # Logic for ingesting OHLCV + indicators into DB
│   │   └── preprocess.go         # Data cleaning and feature engineering
│   │
│   ├── model/                    # Machine Learning & Deep Learning model logic
│   │   ├── inference.go          # Logic for running model predictions (e.g., using ONNX runtime)
│   │   ├── trainer.go            # Model training pipeline (Go orchestration, Python execution)
│   │   └── sentiment.go          # FinBERT or LLM sentiment model integration
│   │
│   ├── strategy/                 # Trading strategy definitions
│   │   ├── intraday.go           # Intraday trading strategy logic
│   │   ├── swing.go              # Swing trading strategy logic
│   │   ├── scalping.go           # Scalping strategy logic
│   │   └── selector.go           # Logic to choose/switch strategies based on market conditions
│   │
│   ├── execution/                # Order execution and monitoring logic
│   │   ├── order.go              # Logic for placing, modifying, and canceling orders
│   │   └── monitor.go            # Position and order monitoring, trailing stop, exit logic
│   │
│   ├── metrics/                  # Prometheus metrics definitions and collection
│   │   └── metrics.go            # Custom metrics for bot performance and health
│   │
│   ├── utils/                    # Common helper utilities
|   |   ├── config.go             # Configuration loading and management
|   |   ├── logger.go             # Centralized logging utility
|   |   └── errors.go             # Custom error types and handling
|   |
|   └── server/                   # HTTP/WebSocket server for frontend and internal communication
│       ├── broadcast.go          # WebSocket server for real-time data push
│       └── routes.go             # Centralized routing for all HTTP/WebSocket handlers
│
├── configs/                      # YAML configuration files
│   ├── app.yaml                  # General application settings (ports, service names)
│   ├── zerodha.yaml              # Zerodha API configuration (non-sensitive: API key)
│   ├── database.yaml             # Database connection details
│   ├── model.yaml                # Model-specific parameters
│   └── strategy.yaml             # Strategy parameters
│
├── models/                       # Saved ML models, ONNX files, etc.
│   └── sentiment.onnx            # Pre-trained sentiment analysis model
│
├── data/                         # Cached or downloaded datasets
│   ├── nse/                      # Historical NSE market data
│   └── news/                     # Cached news articles
│
├── scripts/                      # Utility scripts (Python, Shell)
│   ├── download_data.py          # Script for bulk historical data download
│   ├── retrain_sentiment.py      # Script for retraining sentiment models
│   ├── performance_report.py     # Script for generating backtest/live performance reports
│   └── db_migrate.sh             # Script to run database migrations
│
├── notebooks/                    # Jupyter notebooks for Exploratory Data Analysis (EDA) and experiments
│   └── EDA_intraday.ipynb        # Example notebook for intraday data analysis
│
├── .env                          # Environment variables (API keys, secrets - DO NOT COMMIT!)
├── go.mod                        # Go module dependency file
├── requirements.txt              # Python dependency file
├── Dockerfile                    # Docker build instructions for the bot
├── docker-compose.yaml           # Docker Compose setup for services (bot, DB, Prometheus, Grafana)
└── README.md                     # Project description & setup guide
```

---

## Detailed System Flow: Ultra-Fast, Fault-Tolerant, Multi-User Trading Bot

This flow emphasizes **concurrency, error handling, and modularity** in Go, which are key for achieving speed, fault tolerance, and multi-user support.

### 1. System Initialization & Configuration

At startup, the bot orchestrates its environment and essential services.

* **Entry Point (`cmd/main.go`):**
    * Loads all application settings from **YAML configuration files** (`configs/app.yaml`, `zerodha.yaml`, `database.yaml`, etc.) via `internal/utils/config.go`.
    * **Securely loads sensitive secrets** (e.g., Zerodha API Secret, database passwords, JWT signing keys) **from environment variables** (e.g., `.env` file), ensuring these critical credentials are never hardcoded or exposed in source control.
    * Initializes the **logging system** (`internal/utils/logger.go`) with configurable levels and output destinations for comprehensive operational insights.
    * Sets up **Prometheus metrics** (`internal/metrics/metrics.go`), registering key performance indicators like API call latencies, trade executions, and WebSocket connections for real-time monitoring.
    * Establishes a **connection pool to PostgreSQL** (`internal/db/postgres.go`), ensuring efficient and concurrent database interactions.
    * Optionally initializes a **Redis client pool** for high-speed caching and real-time Pub/Sub messaging.
    * Launches the primary communication servers: the **HTTP server** (`internal/server/http.go`) for REST APIs and metrics, and the **WebSocket server** (`internal/server/websocket.go`) for real-time frontend updates.
    * Starts dedicated **Go goroutines** for all long-running, independent processes such as market data ingestion, strategy execution loops, order monitoring, and news fetching, ensuring concurrent operation.

### 2. User Management & Secure API Key Handling

This is critical for multi-user support and securely managing user-specific broker credentials.

* **Database Schema (`internal/db/models.go`, `internal/db/migrations/`):**
    * **`users` table:** Stores core user information (ID, hashed passwords, email).
    * **`user_broker_accounts` table:** Crucial for multi-broker support, storing linked broker account details per user.
        * `user_id` (Foreign Key)
        * `broker_type` (e.g., "ZERODHA")
        * `api_key` (broker's public API key)
        * **`access_token` (Encrypted at rest):** The highly sensitive access token is securely encrypted in the database and only decrypted on demand for live trading sessions.
        * `public_token`, `request_token`, `session_expiry`, `is_active` status.
    * **`user_strategies` table:** Manages which strategies each user has enabled and their specific configuration parameters.
* **Authentication & Authorization (`internal/auth/auth.go`):**
    * Manages user registration, secure password hashing, and login processes.
    * Generates and validates **JSON Web Tokens (JWTs)** for secure user sessions with the frontend.
    * Implements **middleware for HTTP routes** to enforce authenticated access and authorization rules.
* **User API Handlers (`internal/auth/handlers.go`, `internal/api/handlers/`, `internal/server/router.go`):**
    * **`/signup` (POST):** Endpoint for new user registration.
    * **`/login` (POST):** Authenticates users and issues JWTs.
    * **`/user/broker/zerodha/connect` (GET/POST):** Initiates the **Zerodha OAuth flow**, redirecting users for secure authorization.
        * **Callback URL (`/api/zerodha/callback`):** Handled by `internal/api/handlers/`, it receives the `request_token` from Zerodha. It then exchanges this for the `access_token` and `public_token` via `internal/api/zerodha.go`. These tokens are **encrypted and stored securely** in the `user_broker_accounts` table, linked to the user.
    * **`/user/strategies` (GET/POST):** Allows users to manage their trading strategies and parameters.
    * **`/user/dashboard` (WebSocket Upgrade):** Frontend clients connect here for real-time updates tailored to their account.

### 3. Data Ingestion Pipeline (Real-time & Historical)

This concurrent and fault-tolerant pipeline ensures the bot has access to the most up-to-date market and news data.

* **Instrument Management (`internal/api/instruments.go`):**
    * Fetches and periodically refreshes the latest tradable instrument list from Zerodha, storing it in the PostgreSQL `instruments` table.
* **Real-time Market Data (`internal/api/ticker.go`, `internal/data/ingest.go`):**
    * A **dedicated goroutine pool** manages WebSocket connections to Zerodha Kite Ticker for **each active user's subscribed instruments**, ensuring individual session management and proper rate limiting.
    * `internal/api/ticker.go` receives raw tick/quote data and pipes it into a **Go channel**.
    * `internal/data/ingest.go` (running concurrently across multiple goroutines) processes this data:
        * Aggregates raw ticks into standard OHLCV bars (e.g., 1-minute).
        * Calculates various **technical indicators** (SMA, RSI, MACD, etc.).
        * **Persists** the processed OHLCV and indicator data into the **PostgreSQL `market_data` hypertable (TimescaleDB)**.
        * **Publishes** the latest bar/indicator data to a **Redis Pub/Sub channel** (`market_data:<instrument_id>`) for ultra-fast consumption by strategies and frontend broadcasting.
        * Includes robust **error handling** with retries for API failures and graceful WebSocket disconnections/reconnections.
* **Historical Data Ingestion (`internal/data/ingest.go`, `scripts/download_data.py`):**
    * `scripts/download_data.py` (or a Go equivalent) handles initial bulk downloads.
    * `internal/data/ingest.go` also supports backfilling missing historical data from Zerodha's historical API, storing it in the `market_data` hypertable.
* **News & Sentiment (`internal/api/newsapi.go`, `internal/api/marketwatch.go`, `internal/model/sentiment.go`, `internal/data/ingest.go`):**
    * Dedicated goroutines continuously fetch news articles from configured sources.
    * `internal/model/sentiment.go` (via `internal/model/inference.go`) performs **real-time sentiment analysis** on new articles using pre-trained ML models.
    * `internal/data/ingest.go` stores news content and sentiment scores in the `news_sentiment` table and can publish updates to Redis Pub/Sub for immediate strategy input.

### 4. Strategy & Execution Engine

This is the intelligent core that drives trading decisions and executes orders with precision and robust risk management.

* **Strategy Selection (`internal/strategy/selector.go`):**
    * A central service dynamically selects the most appropriate trading strategy (`intraday.go`, `swing.go`, `scalping.go`) for each active user. This selection is based on current market conditions, time of day, and the user's enabled strategies from the `user_strategies` table.
* **Strategy Execution (`internal/strategy/*.go`):**
    * A **dedicated goroutine** runs the logic for each active user's selected strategy.
    * These goroutines **subscribe to Redis Pub/Sub channels** for real-time market data and sentiment updates, ensuring minimal latency for decision-making.
    * Strategies evaluate entry and exit conditions based on the latest data and user-defined parameters.
    * Upon generating a **trade signal**, it's passed to `internal/execution/order.go`.
* **Order Management (`internal/execution/order.go`):**
    * Receives trade signals (e.g., BUY/SELL, instrument, quantity, target, stop-loss).
    * **Crucially, performs stringent risk management checks:**
        * Verifies available capital for the specific user's account.
        * Enforces max daily loss limits configured for the user.
        * Calculates optimal position sizing.
        * Checks against existing open positions to avoid overexposure.
    * Constructs the precise order payload required by the Zerodha API.
    * Sends the order request to `internal/api/zerodha.go` for placement.
    * Logs detailed order information (including `user_id` and `strategy_name`) to the **PostgreSQL `orders` table**.
* **Order & Position Monitoring (`internal/execution/monitor.go`):**
    * A **dedicated goroutine for each active user** continuously monitors their open orders and positions via the Zerodha API.
    * Updates the `orders` and `positions` tables in PostgreSQL in real-time.
    * **Implements sophisticated exit logic:**
        * Detects target profit or stop-loss level hits.
        * Manages dynamic trailing stops.
        * Handles end-of-day square-offs for intraday positions.
    * If an exit condition is met, it signals `internal/execution/order.go` to place a closing order.
    * Updates the `trades` table upon successful order fills.

### 5. Monitoring & Observability

Comprehensive monitoring is vital for maintaining bot health, identifying issues, and tracking performance.

* **Prometheus Exporter (`internal/metrics/metrics.go`, `internal/server/http.go`):**
    * `internal/metrics/metrics.go` defines and exposes custom Prometheus metrics (e.g., `go_goroutines_total`, `db_queries_total`, `api_calls_total`, `strategy_signals_total`, `order_fills_total`, `user_pnl_current`).
    * All key application components increment/set these metrics.
    * The `internal/server/http.go` exposes a `/metrics` endpoint that **Prometheus** can scrape.
* **Prometheus (`docker-compose.yaml`):**
    * Runs as a separate Docker container, configured to periodically scrape metrics from the bot.
    * Stores the collected time-series metric data for historical analysis.
* **Grafana (`docker-compose.yaml`):**
    * Runs as a separate Docker container, connecting to Prometheus as its data source.
    * Provides **dynamic dashboards** to visualize bot health, real-time trade activity, P&L, API usage, and system resource utilization.
    * Enables the setup of **alerts** for critical events (e.g., high memory usage, API errors, bot downtime, significant user drawdown).
* **Centralized Logging (`internal/utils/logger.go`):**
    * Logs all critical events, errors, warnings, and informational messages in a **structured format (e.g., JSON)** for easy parsing and analysis by log aggregation tools (like Grafana Loki, if integrated).

### 6. Real-time Data Broadcasting

The bot pushes real-time updates to connected user frontends, creating a dynamic dashboard experience.

* **WebSocket Server (`internal/server/websocket.go`):**
    * Maintains active WebSocket connections with multiple frontend clients.
    * Authenticates users upon connection using JWTs (`internal/auth/auth.go`) and associates connections with their `user_id`.
* **Data Flow to Frontend (via Redis Pub/Sub):**
    * The WebSocket server **subscribes to relevant Redis Pub/Sub channels** (e.g., `market_data:<instrument_id>`, `user_pnl:<user_id>`, `order_updates:<user_id>`).
    * As new data is published to these channels (by the ingestion, execution, and monitoring modules), the WebSocket server:
        * **Filters the data** to ensure it's relevant only to the specific connected user.
        * **Pushes this real-time data** (latest OHLCV, current P&L, order fills, strategy signals, alerts) down to the user's frontend via their WebSocket connection.
    * This **decoupled architecture (Redis Pub/Sub)** ensures highly scalable and efficient real-time broadcasting without overloading the core bot logic.

### 7. Fault Tolerance & High Performance Considerations

Built with resilience and speed at its core.

* **Concurrency (Go Goroutines & Channels):** Leveraging Go's lightweight goroutines for independent tasks (e.g., one goroutine per user's WebSocket connection, each active strategy, each API polling task) ensures parallelism. Channels provide safe and efficient communication between these concurrent operations.
* **Graceful Shutdowns:** Implemented using Go's signal handling (`os.Interrupt`, `syscall.SIGTERM`) in `main.go` to ensure all open connections are closed and in-flight data is flushed before the bot gracefully exits.
* **Robust Error Handling & Retries:**
    * Comprehensive error handling for all external API calls and database operations.
    * **Exponential backoff and retry mechanisms** for transient network or API errors.
    * **Circuit breakers** for external services to prevent cascading failures during outages.
* **Database Resilience:**
    * PostgreSQL is a robust and highly reliable database.
    * Utilizes **connection pooling** (`internal/db/postgres.go`) to manage database connections efficiently, reducing overhead.
    * Proper **indexing** on critical tables (`market_data`, `orders`, `trades`, `positions`) ensures fast query performance.
    * Regular **database backups** are essential for data recovery.
* **Rate Limiting:** Implements client-side rate limiting for all outgoing API calls to Zerodha to comply with their usage limits and prevent temporary blocks.
* **Idempotency:** Operations are designed to be idempotent where possible (e.g., ensuring that placing an order multiple times due to network issues doesn't result in duplicate orders if the first attempt was successful).
* **Stateless Services:** Where applicable, services are designed to be stateless, pushing state management to the database or Redis. This improves scalability and simplifies recovery.
* **Containerization (Docker):** The entire application is containerized using `Dockerfile` and orchestrated with `docker-compose.yaml`, ensuring consistent environments, easy deployment, and simplified management of services (bot, PostgreSQL, Prometheus, Grafana).
* **Hardware Monitoring:** Continuous system-level monitoring (e.g., macOS Activity Monitor, Grafana dashboards) tracks CPU, RAM, and disk I/O to proactively identify and mitigate any hardware bottlenecks.

---

## Tech Stack

Our bot is built using a powerful and modern tech stack for performance, scalability, and maintainability.

### Core Application & Backend
* **Primary Programming Language:** **Go (Golang)**
* **Web Framework (Go):** Standard library `net/http` (or a lightweight framework like **Gin/Echo** for more complex routing)
* **Concurrency Model:** **Go Goroutines & Channels**

### Databases & Data Storage
* **Primary Database:** **PostgreSQL**
* **Time-Series Extension:** **TimescaleDB** (PostgreSQL extension)
* **In-Memory Cache / Message Broker:** **Redis**

### Machine Learning & Data Science
* **ML/DL Framework (Python):** **PyTorch / TensorFlow** (for model training)
* **ML Model Inference Format:** **ONNX**
    * **ONNX Runtime:** `microsoft/onnxruntime v1.21.0` (for Go inference)
* **Python Libraries for Data/ML:**
    * **Pandas / NumPy:** Data manipulation
    * **Hugging Face Transformers:** For FinBERT/LLM models
    * **Scikit-learn:** General ML utilities
    * **Matplotlib / Seaborn:** Data visualization
* **Exploratory Data Analysis (EDA):** **Jupyter Notebooks**

### API Integrations
* **Broker API:** **Zerodha Kite Connect API**
* **News APIs:** **NewsAPI**, **MarketWatch API** (or similar)
* **Go HTTP Client:** Standard `net/http` package
* **Go WebSocket Client:** (e.g., `gorilla/websocket`)

### Configuration & Environment Management
* **Configuration Files:** **YAML**
* **Environment Variables:** (`.env` file)

### Monitoring & Observability
* **Metrics Collection:** **Prometheus**
* **Visualization & Alerting:** **Grafana**
* **Logging:** **Go's `log` package / Custom Logger** (e.g., `logrus` or `zap`)

### Deployment & Operations
* **Containerization:** **Docker**
* **Container Orchestration:** **Docker Compose**
* **Version Control:** **Git**

### Security Aspects
* **Authentication:** **JSON Web Tokens (JWT)**
* **Encryption:** **Go's `crypto` packages** (for data at rest)

---

## Phase-wise Development Plan

This outlines our planned approach to building the bot, broken down into manageable phases.

| Phase                       | Status        | Start Date      | Notes                                                               |
| :-------------------------- | :------------ | :-------------- | :------------------------------------------------------------------ |
| **1: Zerodha Integration** | 🚧 In Progress | ✅ May 27, 2025 | Enable bot to connect to live markets, stream data, and place orders. |
| **2: Data Ingestion & Feature Engineering** | ⏳ Next       | TBD             | Bring in real-time & historical data to feed models and strategies.   |
| **3: Strategy Framework** | ⏳             | TBD             | Define rules & logic to decide buy/sell/hold per strategy.          |
| **4: ML/DL Predictions (Price Forecasting)** | ⏳             | TBD             | Predict future price movement using ML/DL models.                   |
| **5: Sentiment Analysis (FinBERT / NewsAPI)** | 💤 Last Phase | TBD             | Use financial news to detect bullish/bearish bias.                  |
| **6: Execution Engine** | ⏳             | TBD             | Fire orders live on signals from model + strategy.                  |
| **7: Backtesting & Logging**| ⏳             | TBD             | Evaluate how the bot performs on historical data.                   |
| **8: Deployment & Automation**| 🧾 Final Polish | TBD             | Let bot run on autopilot with monitoring and alerts.                |

---

## Testing WebSockets (for local development)

You can easily test the WebSocket server using `wscat` from your terminal:

1.  **Install `wscat`:**
    ```bash
    npm i -g wscat
    ```
2.  **Connect to your WebSocket server:**
    ```bash
    wscat -c ws://localhost:8000/ws
    ```

---


This project structure introduces a sophisticated flow for an equity trading bot, moving beyond a single-user, file-based system to a multi-user, database-driven, and observable architecture.

Let's break down the usage, flow, and execution of each file and component group.

---

## **Detailed Flow and Execution of the Equity Trading Bot**

### **I. Initial Setup and Infrastructure (Prior to Application Start)**

1.  **`configs/` (`app.yaml`, `zerodha.yaml`, `database.yaml`, `model.yaml`, `strategy.yaml`):**
    * **Usage:** Define application parameters, API keys (non-sensitive), database connection strings, model paths, strategy parameters, etc.
    * **Flow:** These YAML files are read by `internal/utils/config.go` at application startup.
    * **Execution:** Static configuration, loaded once.
2.  **`.env`:**
    * **Usage:** Stores sensitive environment variables (e.g., `ZERODHA_API_KEY`, `ZERODHA_API_SECRET`, `JWT_SECRET_KEY`, `DATA_ENCRYPTION_KEY`, database credentials).
    * **Flow:** Loaded by `internal/utils/config.go` which then populates the `Config` struct.
    * **Execution:** Loaded once at startup. **Crucially, this file is never committed to version control.**
3.  **`docker-compose.yaml`:**
    * **Usage:** Defines and runs the multi-container Docker application environment (PostgreSQL, Redis, Prometheus, Grafana, and the Go bot itself).
    * **Flow:** `docker-compose up -d` starts all services.
    * **Execution:** Orchestrates the entire application stack.
4.  **`internal/db/migrations/` (`.sql` files):**
    * **Usage:** Define the database schema (tables for `users`, `user_broker_accounts`, `instruments`, `market_data`, `orders`, `trades`, `positions`, `news_sentiment`, etc.).
    * **Flow:** Executed by a migration tool (e.g., `migrate`) via `scripts/db_migrate.sh`.
    * **Execution:** Run once during initial setup or whenever schema changes are made.
5.  **`scripts/db_migrate.sh`:**
    * **Usage:** A shell script to apply database schema migrations.
    * **Flow:** Calls the `migrate` command-line tool with the necessary database connection details and migration paths.
    * **Execution:** Manually executed (e.g., `sh scripts/db_migrate.sh up`) by a developer/admin to initialize or update the database schema.
6.  **`scripts/download_data.py`:**
    * **Usage:** Python script for initial or bulk historical data download (e.g., EOD data for backtesting or initial `market_data` population).
    * **Flow:** Authenticates with a broker API (e.g., Zerodha), fetches historical data, and saves it to `data/nse/` or directly inserts into `internal/db/postgres.go` via an adapter.
    * **Execution:** Can be run manually or scheduled via cron job within a Docker container.

---

### **II. Application Startup (`cmd/main.go`)**

1.  **`cmd/main.go` (Entry Point):**
    * **Usage:** Orchestrates the entire bot's lifecycle, from initialization to graceful shutdown.
    * **Flow:**
        1.  **`internal/utils/config.go` (`LoadConfig`):** Reads `configs/` YAML files and `.env` variables to create a unified `Config` struct.
        2.  **`internal/utils/logger.go` (`InitLogger`):** Initializes a structured logger based on `Config` settings (level, output). All subsequent logging uses this utility.
        3.  **`internal/metrics/metrics.go` (`InitMetrics`):** Registers Prometheus custom metrics.
        4.  **`internal/db/postgres.go` (`NewPostgresDB`):** Establishes a connection pool to the PostgreSQL database using settings from `Config`.
        5.  **`github.com/redis/go-redis/v9`:** Establishes a connection to the Redis server.
        6.  **Service Initialization:** Creates instances of various core services (`auth`, `api`, `data`, `strategy`, `execution`), injecting necessary dependencies (e.g., `Config`, `db.DB`, `redis.Client`).
        7.  **`internal/server/router.go` (`NewRouter`):** Initializes the main HTTP router, registering all API and WebSocket endpoints, and applying middleware (like CORS, JWT authentication).
        8.  **`internal/server/http.go` (implicitly `http.ListenAndServe`):** Starts the HTTP server, listening on the configured port.
        9.  **Background Goroutines:** Launches long-running goroutines for:
            * `internal/data/ingest.go` (real-time market data ingestion).
            * `internal/execution/monitor.go` (order and position monitoring).
            * `internal/strategy/` (strategy execution loops).
            * Other periodic tasks (e.g., instrument master data updates).
        10. **Graceful Shutdown:** Sets up signal handling (`os.Signal`) to gracefully shut down the HTTP server and other resources (DB, Redis connections) on `SIGINT` or `SIGTERM`.
    * **Execution:** This is the primary executable that runs 24/7 (or during market hours) as the core bot application.

---

### **III. User Authentication and Broker Connection Flow**

1.  **User Signup/Login:**
    * **Frontend Action:** User interacts with a web interface (not part of this Go backend, but assumes its existence).
    * **`POST /signup`:**
        * **Flow:** Frontend -> `internal/server/router.go` -> `internal/auth/handlers.Signup`.
        * **Execution:** `Signup` handler uses `internal/auth/auth.go` to hash the password and `internal/db/postgres.go` to save the `User` record to the `users` table.
    * **`POST /login`:**
        * **Flow:** Frontend -> `internal/server/router.go` -> `internal/auth/handlers.Login`.
        * **Execution:** `Login` handler uses `internal/auth/auth.go` to verify password, then `internal/auth/auth.go` to generate a JWT. The JWT is returned to the frontend.
2.  **Connecting Zerodha Broker Account:**
    * **Frontend Action:** Authenticated user navigates to "Connect Broker" in the web interface.
    * **`GET /user/broker/zerodha/connect` (Authenticated):**
        * **Flow:** Frontend (with JWT) -> `internal/server/router.go` -> `internal/auth/auth.JWTAuthMiddleware` (validates JWT, extracts `userID`) -> `internal/auth/handlers.ConnectZerodha`.
        * **Execution:** `ConnectZerodha` handler constructs the Zerodha OAuth login URL using `ZERODHA_API_KEY` and `ZERODHA_REDIRECT_URL` from `Config`, and redirects the user's browser to Zerodha.
    * **Zerodha OAuth Callback (`GET /api/zerodha/callback`):**
        * **Flow:** Zerodha's server (after user authorization) redirects back to this public endpoint.
        * **Execution:** `internal/api/handlers/zerodha.ZerodhaOAuthCallback` handler:
            1.  Receives `request_token` from query parameters.
            2.  Calls `internal/api/zerodha.GenerateSession` (using `gokiteconnect`) to exchange `request_token` for `access_token` and `public_token`.
            3.  **`internal/utils/encryption.go`:** Encrypts the obtained `access_token` and `public_token` using the `DATA_ENCRYPTION_KEY`.
            4.  **`internal/db/postgres.go`:** Stores the encrypted tokens, `userID`, and `broker_type` into the `user_broker_accounts` table.
            5.  Redirects the user back to the frontend dashboard, possibly with a success/error status.

---

### **IV. Real-time Market Data Ingestion and Streaming Flow**

This is a critical background process and how the frontend receives updates.

1.  **User ZerodhaClient Initialization (Background):**
    * **Flow:** A background goroutine (e.g., part of a `MarketDataService` or `internal/data/ingest.go`'s startup logic) periodically queries `internal/db/postgres.go` for active `UserBrokerAccounts`.
    * **Execution:** For each active Zerodha account:
        1.  The encrypted `access_token` and `public_token` are retrieved.
        2.  **`internal/utils/encryption.go`:** Decrypts these tokens.
        3.  **`internal/api/zerodha.go` (`NewZerodhaClient`):** A new `ZerodhaClient` instance is created for this specific user.
        4.  **`internal/api/ticker.go` (`TickerManager.AddUserTicker`):** This `ZerodhaClient` is passed to the `TickerManager`.
2.  **Market Data Subscription & Tick Reception:**
    * **Flow:** `internal/api/ticker.go`'s `TickerManager` manages the `kiteticker.Ticker` for each user:
        1.  It establishes the WebSocket connection to Zerodha's ticker feed.
        2.  It sets up `OnConnect`, `OnTick`, `OnError`, `OnClose` callbacks.
        3.  It subscribes to a predefined set of instruments (e.g., from `Config` or `user_strategy` settings) using `TickerManager.SubscribeUserToInstruments`.
    * **Execution:** When a tick arrives from Zerodha:
        1.  The `OnTick` callback in `internal/api/ticker.go` fires.
        2.  It passes the raw `kitemodels.Tick` data (along with the `userID`) to `internal/data/ingest.go` (`ProcessTick` method).
3.  **Data Ingestion and Processing (`internal/data/ingest.go`):**
    * **Flow:** `ProcessTick(userID, tick)` receives the raw tick.
    * **Execution:**
        1.  **`internal/db/postgres.go`:** Looks up the `InstrumentInfo` (from the `instruments` table) using the `InstrumentToken` in the tick.
        2.  **OHLCV Aggregation:** Aggregates ticks into minute/period OHLCV bars.
        3.  **`internal/data/preprocess.go`:** (Or directly in `ingest.go`) Calculates technical indicators (RSI, MACD, Moving Averages) on the OHLCV data.
        4.  **`internal/db/postgres.go`:** Persists the processed OHLCV and indicator data into the `market_data` hypertable (TimescaleDB) for historical analysis and strategy backtesting.
        5.  **`github.com/redis/go-redis/v9` (Publish):** Publishes the *processed* market data (and potentially derived signals) to specific Redis Pub/Sub channels (e.g., `market_data:<instrument_token>`).
        6.  **Strategy Trigger:** Notifies `internal/strategy/selector.go` or specific strategy modules that new market data is available.
        7.  **P&L Calculation:** If the tick impacts a user's open position, it might trigger a P&L recalculation, which is then also published to Redis.
4.  **Real-time Frontend Streaming (`internal/server/websocket.go`):**
    * **Frontend Action:** Authenticated user opens the dashboard and establishes a WebSocket connection.
    * **`GET /ws` (Authenticated WebSocket Endpoint):**
        * **Flow:** Frontend (with JWT) -> `internal/server/router.go` -> `internal/auth/auth.JWTAuthMiddleware` (validates JWT, extracts `userID`) -> `internal/server/websocket.NewWebSocketHandler`.
        * **Execution:** `NewWebSocketHandler`:
            1.  Upgrades the HTTP connection to a WebSocket.
            2.  Retrieves the `userID` from the request context.
            3.  **`github.com/redis/go-redis/v9` (Subscribe):** Subscribes this specific WebSocket connection to Redis Pub/Sub channels relevant to the `userID` (e.g., `market_data:<instrument_token>` for subscribed instruments, `user_pnl:<user_id>`, `order_updates:<user_id>`, `news_alerts`).
            4.  A dedicated goroutine is spawned for each WebSocket connection to continuously read messages from its subscribed Redis channels.
            5.  When a message is received from Redis, it's sent directly to the client's WebSocket connection (`conn.WriteMessage`).
            6.  The `ReadMessage` loop listens for client pings/disconnects.

---

### **V. Trading Strategy Execution Flow**

1.  **Strategy Initialization & Selection:**
    * **Flow:** `internal/strategy/selector.go` is initialized at startup or on user configuration change. It loads user-specific strategy settings from `internal/db/postgres.go` (`users` table, `user_strategy` relationship).
    * **Execution:** `selector.go` determines which strategies (`intraday.go`, `swing.go`, etc.) are active for a given user/instrument.
2.  **Signal Generation:**
    * **Flow:** Active strategies (e.g., `internal/strategy/intraday.go`) are typically triggered:
        * Upon new OHLCV bar completion (from `internal/data/ingest.go`).
        * On specific market events (e.g., significant price change).
        * On a timed interval.
    * **Execution:** Strategies:
        1.  Query `internal/db/postgres.go` for `market_data` and current `positions`.
        2.  May use `internal/model/inference.go` to get sentiment from news.
        3.  Apply their trading logic based on technical indicators, sentiment, and user-defined parameters.
        4.  If a buy/sell signal is generated, they pass an `OrderRequest` to `internal/execution/order.go` (`PlaceOrder` method).
3.  **Order Placement (`internal/execution/order.go`):**
    * **Flow:** Receives `OrderRequest` from a strategy or a manual user action via `/api/user/order/place`.
    * **Execution:**
        1.  Retrieves the user's `ZerodhaClient` (and decrypts tokens) from `internal/db/postgres.go`.
        2.  Uses `internal/api/zerodha.Kite` (from the user's `ZerodhaClient`) to place the order with the broker.
        3.  **`internal/db/postgres.go`:** Records the initial order details in the `orders` table.
        4.  **`github.com/redis/go-redis/v9` (Publish):** Publishes an initial "order placed" notification to the `order_updates:<user_id>` Redis channel.

---

### **VI. Order and Position Monitoring Flow**

1.  **`internal/execution/monitor.go` (Background Task):**
    * **Usage:** Continuously fetches and updates order/position status from the broker.
    * **Flow:**
        1.  Runs as a long-lived goroutine started by `cmd/main.go`.
        2.  Periodically (e.g., every few seconds) iterates through active `users` with connected broker accounts.
        3.  For each user, it retrieves their `ZerodhaClient` (decrypting tokens).
        4.  Uses `internal/api/zerodha.Kite` to fetch the latest order book and position book.
        5.  **`internal/db/postgres.go`:** Updates the `orders`, `trades`, and `positions` tables with the latest status.
        6.  Identifies changes (e.g., order filled, position closed).
        7.  **`github.com/redis/go-redis/v9` (Publish):** Publishes critical updates (e.g., "order filled", "position change", "P&L update") to the `order_updates:<user_id>` and `user_pnl:<user_id>` Redis channels.
    * **Execution:** A vital background component for real-time tracking and triggering further actions (e.g., exit strategies, P&L calculations).

---

### **VII. Data Management and Background Processes**

1.  **`internal/api/instruments.go` (`InstrumentManager`):**
    * **Usage:** Manages the list of tradable instruments.
    * **Flow:** At startup, or periodically as a background goroutine (launched by `cmd/main.go`), it calls `FetchAndStoreInstruments`.
    * **Execution:** `FetchAndStoreInstruments` uses a system-level `ZerodhaClient` to download the latest instrument master data from Zerodha's API and then **`internal/db/postgres.go`** to `UPSERT` this data into the `instruments` table. `FindInstrumentToken` directly queries this database table.
2.  **`internal/data/preprocess.go`:**
    * **Usage:** Provides utility functions for data cleaning, transformation, and technical indicator calculations.
    * **Flow:** Called by `internal/data/ingest.go` during real-time tick processing, or by `internal/model/trainer.go` during model training.
    * **Execution:** Pure functions transforming data.
3.  **`internal/model/` (Inference/Training):**
    * **Usage:** Integrates ML models (e.g., sentiment analysis).
    * **Flow:**
        * `internal/model/inference.go`: Loaded at startup. Used by `internal/data/ingest.go` (for news sentiment) or `internal/strategy/` to make predictions.
        * `internal/model/trainer.go` (mostly Python-orchestrated): Orchestrates model training using `scripts/retrain_sentiment.py`.
    * **Execution:** `inference.go` runs predictions during runtime. `trainer.go` is more for development/scheduled retraining.

---

### **VIII. Monitoring and Observability**

1.  **`internal/metrics/metrics.go`:**
    * **Usage:** Defines and manages custom Prometheus metrics (counters, gauges, histograms) for tracking bot performance, API calls, errors, etc.
    * **Flow:** `InitMetrics()` is called at startup. Various modules make calls like `metrics.OrderCount.Inc()` or `metrics.APILatency.Observe()`.
    * **Execution:** Metrics data is collected in memory by the Go process.
2.  **`GET /metrics`:**
    * **Usage:** Endpoint for Prometheus to scrape metrics data.
    * **Flow:** Prometheus (running in Docker Compose) periodically makes HTTP GET requests to this endpoint.
    * **Execution:** `internal/metrics.PrometheusHandler()` generates the text-based exposition format for Prometheus.
3.  **Prometheus (Docker Compose):**
    * **Usage:** Time-series database that scrapes, stores, and aggregates metrics from `/metrics` endpoint.
    * **Flow:** Configured to scrape `http://bot-app:8000/metrics`.
    * **Execution:** Runs continuously, collecting data.
4.  **Grafana (Docker Compose):**
    * **Usage:** Data visualization tool to create dashboards based on Prometheus data.
    * **Flow:** Connects to Prometheus as a data source.
    * **Execution:** Runs continuously, providing real-time and historical dashboards for bot health, performance, and trading statistics.

---
## Docker start

### 🔐 .env File

Ensure your `.env` file is located in the root of your project (`Go-project/.env`):

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
````

---

## ⚙️ docker-compose.yaml

Path: `configs/docker-compose.yaml`

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: trading_bot_postgres
    environment:
      POSTGRES_DB: ${DB_NAME}
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d ${DB_NAME}"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: trading_bot_redis
    command: redis-server --requirepass ${REDIS_PASSWORD}
    ports:
      - "${REDIS_PORT}:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "INFO"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
```

---

## 🛠️ Usage

### ✅ Start containers with `.env`

```bash
docker compose --env-file .env -f configs/docker-compose.yaml up -d
```

### ❎ Stop and remove containers + volumes

```bash
docker compose --env-file .env -f configs/docker-compose.yaml down -v
```

### 🧪 Validate your config (check if `.env` is loading)

```bash
docker compose --env-file .env -f configs/docker-compose.yaml config
```

You should see actual values in `POSTGRES_DB`, `REDIS_PASSWORD`, etc.

---

## ✅ Tips

* Make sure `.env` is **not named `.env.example`** or anything else.
* Avoid using `export` inside `.env`. Use `KEY=value` format only.
* Use the full path when running from different folders.
* You can test services with tools like `psql` and `redis-cli`.

---

## 🧩 Optional Services (to add later)

Uncomment and add Prometheus, Grafana, or Redis Commander as needed.

```

---

## **Setting Up Your Database**

Before running the main application, you need to set up your PostgreSQL database and enable the TimescaleDB extension. (Tested on PostgreSQL 17.5)

### 1. Enable TimescaleDB Extension

The `market_data` table relies on the TimescaleDB extension for efficient time-series data handling. You need to enable this extension in your PostgreSQL database *before* running any migrations.

**Steps:**

1.  **Ensure `psql` is installed:** Make sure the `psql` command-line client for PostgreSQL is installed and accessible in your terminal's PATH. If you don't have it, you'll need to install the PostgreSQL client tools for your operating system.
2.  **Run the pre-migration script:** Navigate to your project's root directory in the terminal and execute the following command, replacing the placeholders with your actual database connection details (from your `.env`):

    ```bash
    psql -h <your_db_host> -p <your_db_port> -U <your_db_user> -d <your_db_name> -f internal/db/migrations/pre_migration_enable_timescaledb.sql
    ```

    **Example:**
    ```bash
    psql -h localhost -p 5432 -U (admin || postgres) -d trading_bot_db -f internal/db/migrations/pre_migration_enable_timescaledb.sql
    ```
    You'll be prompted for your database password.

### 2. Run Database Migrations

Once TimescaleDB is enabled, you can run the rest of your database migrations to create the necessary tables. We recommend using `golang-migrate/migrate` for this.

**Steps:**

1.  **Install `golang-migrate/migrate`:**
    ```bash
    go install -tags 'postgres' github.com/golang-migrate/migrate/v4/cmd/migrate@latest
    ```
2.  **Set your database URL:** Export your database connection string as an environment variable (replace with your actual details):
    ```bash
    export DATABASE_URL="postgres://<your_db_user>:<your_db_password>@<your_db_host>:<your_db_port>/<your_db_name>?sslmode=disable"

    Example:
    export DATABASE_URL="postgres://admin:secret@localhost:5432/trading_bot_db?sslmode=disable"
    ```
    **Example:**
    ```bash
    export DATABASE_URL="postgres://admin:secret@localhost:5432/trading_bot_db?sslmode=disable"
    ```
3.  **Apply migrations:**
    ```bash
    migrate -path internal/db/migrations -database "$DATABASE_URL" up
    ```

Your database will now be fully set up and ready for your bot!

---

# `db_migrate.sh` Quick Reference Guide

This script uses `golang-migrate/migrate` to manage your PostgreSQL database schema.

## Purpose

Automates applying, reverting, and creating database migrations for the Equity Trading Bot.

## Prerequisites

1.  **`migrate` CLI Tool**: Installed (`go install -tags 'postgres' github.com/golang-migrate/migrate/v4/cmd/migrate@latest`).

2.  **PostgreSQL Running**: Database server is active and accessible.

3.  **Database Exists**: The target database (e.g., `trading_bot_db`) is already created.

4.  **TimescaleDB (if used)**: `CREATE EXTENSION IF NOT EXISTS timescaledb;` run in your database.

5.  **`.env` Configured**: `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` are correctly set in `equity-trading-bot/.env` (no quotes around values).

6.  **Migration Files**: SQL files (`.up.sql`, `.down.sql`) are in `internal/db/migrations/`.

7.  **Script Executable**: `chmod +x scripts/db_migrate.sh`.

## Usage

Navigate to your project's root directory (`equity-trading-bot/`).

### 1. Apply Migrations (`up`)

* **Apply all pending migrations:**

    ```bash
    ./scripts/db_migrate.sh up
    ```

* **Apply next `N` migrations (e.g., next 1):**

    ```bash
    ./scripts/db_migrate.sh up 1
    ```

### 2. Revert Migrations (`down`)

**WARNING: Can cause data loss. Use with extreme caution, especially in production.**

* **Revert all applied migrations:**

    ```bash
    ./scripts/db_migrate.sh down
    ```

* **Revert last `N` migrations (e.g., last 1):**

    ```bash
    ./scripts/db_migrate.sh down 1
    ```

### 3. Create New Migration Files (`create`)

* **Generate new `.up.sql` and `.down.sql` files:**

    ```bash
    ./scripts/db_migrate.sh create your_migration_name
    ```

    (e.g., `./scripts/db_migrate.sh create add_new_table`)

### 4. Show Current Database Version (`version`)

* **Display applied migration version:**

    ```bash
    ./scripts/db_migrate.sh version
    ```

### 5. Force Set Version (`force`)

**EXTREME CAUTION!** Manually sets the database version. Only for recovery from failed migrations.

* **Force version to `N`:**

    ```bash
    ./scripts/db_migrate.sh force N
    ```

## Important Notes

* **Disable GORM `AutoMigrate`**: Once using this script, remove all `db.AutoMigrate()` calls from your Go application.

* **Backup**: Always back up your database before running migrations in production.

* **Environment**: Ensure `.env` is correctly configured for the script's execution environment.