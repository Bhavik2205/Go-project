CREATE TABLE backtest_jobs (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,
    user_id BIGINT NOT NULL,
    strategy_name VARCHAR(255) NOT NULL,
    symbols JSONB NOT NULL DEFAULT '[]',          -- array of symbol strings e.g. ["NSE:RELIANCE"]
    from_time TIMESTAMP WITH TIME ZONE NOT NULL,
    to_time TIMESTAMP WITH TIME ZONE NOT NULL,
    initial_capital NUMERIC NOT NULL DEFAULT 100000,
    fees_config JSONB NOT NULL DEFAULT '{}',      -- brokerageType, brokerageValue, slippagePercent
    parameters JSONB NOT NULL DEFAULT '{}',       -- strategy-specific params
    status VARCHAR(50) NOT NULL DEFAULT 'PENDING', -- PENDING, RUNNING, COMPLETED, FAILED
    result JSONB,                                  -- summary metrics: totalReturn, sharpe, maxDrawdown, etc.
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    CONSTRAINT fk_backtest_jobs_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE backtest_trades (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    backtest_job_id BIGINT NOT NULL,
    instrument_token BIGINT NOT NULL,
    symbol VARCHAR(255) NOT NULL,
    transaction_type VARCHAR(10) NOT NULL,        -- BUY or SELL
    quantity INTEGER NOT NULL,
    price NUMERIC NOT NULL,
    trade_time TIMESTAMP WITH TIME ZONE NOT NULL,
    pnl NUMERIC,
    CONSTRAINT fk_backtest_trades_job_id FOREIGN KEY (backtest_job_id) REFERENCES backtest_jobs(id) ON DELETE CASCADE
);

CREATE TABLE backtest_equity_curve (
    id BIGSERIAL PRIMARY KEY,
    backtest_job_id BIGINT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    equity NUMERIC NOT NULL,
    CONSTRAINT fk_backtest_equity_curve_job_id FOREIGN KEY (backtest_job_id) REFERENCES backtest_jobs(id) ON DELETE CASCADE
);

CREATE INDEX idx_backtest_jobs_user_id ON backtest_jobs (user_id);
CREATE INDEX idx_backtest_jobs_status ON backtest_jobs (status);
CREATE INDEX idx_backtest_trades_job_id ON backtest_trades (backtest_job_id);
CREATE INDEX idx_backtest_equity_curve_job_id ON backtest_equity_curve (backtest_job_id);
