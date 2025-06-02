-- Note: The TimescaleDB extension 'timescaledb' must be enabled BEFORE running this migration.
-- Use the pre_migration_enable_timescaledb.sql script for that.

CREATE TABLE market_data (
    instrument_token BIGINT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    sma20 DOUBLE PRECISION,
    rsi14 DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_hist DOUBLE PRECISION,
    PRIMARY KEY (instrument_token, timestamp)
);

-- Convert to TimescaleDB hypertable
-- 'timestamp' is the time column, 'instrument_token' is the partitioning key
SELECT create_hypertable('market_data', 'timestamp', chunk_time_interval => INTERVAL '1 day', migrate_data => TRUE, partitioning_column => 'instrument_token', number_partitions => 8);

-- Add foreign key constraint to instruments table
ALTER TABLE market_data ADD CONSTRAINT fk_market_data_instrument_token
FOREIGN KEY (instrument_token) REFERENCES instruments(instrument_token) ON DELETE RESTRICT;

-- Index for efficient time-range and instrument-specific queries
CREATE INDEX idx_market_data_instrument_token_timestamp ON market_data (instrument_token, timestamp DESC);