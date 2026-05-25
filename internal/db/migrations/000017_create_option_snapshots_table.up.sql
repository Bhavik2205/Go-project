CREATE TABLE option_snapshots (
    instrument_token INTEGER NOT NULL,      -- references instruments (must be an option)
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Market data from Kite / broker
    last_price NUMERIC NOT NULL,
    bid_price NUMERIC,
    ask_price NUMERIC,
    volume INTEGER,
    open_interest INTEGER,
    implied_volatility NUMERIC,             -- IV in percentage (e.g., 18.5)
    
    -- Greeks (computed by your `internal/options/greeks.go`)
    delta NUMERIC,
    gamma NUMERIC,
    theta NUMERIC,                          -- per day (or per second)
    vega NUMERIC,
    rho NUMERIC,
    
    -- Underlying price at same timestamp (for reference)
    underlying_price NUMERIC,
    
    -- Optional: theoretical price (Black‑Scholes)
    theoretical_price NUMERIC,
    
    PRIMARY KEY (instrument_token, timestamp)
);

-- Convert to TimescaleDB hypertable (time‑series)
SELECT create_hypertable('option_snapshots', 'timestamp',
    chunk_time_interval => INTERVAL '1 hour',
    migrate_data => TRUE,
    if_not_exists => TRUE
);

CREATE INDEX idx_option_snapshots_token_time ON option_snapshots (instrument_token, timestamp DESC);
CREATE INDEX idx_option_snapshots_underlying ON option_snapshots (underlying_price);

ALTER TABLE option_snapshots ADD CONSTRAINT fk_option_snapshots_instrument_token
FOREIGN KEY (instrument_token) REFERENCES instruments(instrument_token) ON DELETE CASCADE;