-- Note: The TimescaleDB extension 'timescaledb' must be enabled BEFORE running this migration.
-- Use the pre_migration_enable_timescaledb.sql script for that.

-- Create table if it doesn't exist (idempotent - safe for production)
CREATE TABLE IF NOT EXISTS market_data (
    -- Composite Primary Keys (aligned with Go's uint32 / int)
    instrument_token INTEGER NOT NULL, -- Go's uint32 maps to INTEGER in PostgreSQL
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    tick_sequence_id INTEGER NOT NULL, -- Custom ID for uniqueness within the same timestamp

    -- Core Tick Data (aligned with kitemodels.Tick)
    last_price NUMERIC NOT NULL,
    last_traded_quantity INTEGER NOT NULL, -- Go's uint32 maps to INTEGER
    volume INTEGER NOT NULL, -- Go's uint32 (from VolumeTraded) maps to INTEGER
    average_trade_price NUMERIC NOT NULL,
    net_change NUMERIC NOT NULL,

    -- Daily OHLC (from kitemodels.Tick.OHLC)
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,

    -- Open Interest (from kitemodels.Tick.OI - will be 0 for equities)
    open_interest INTEGER NOT NULL, -- Go's uint32 maps to INTEGER

    -- Market Depth (Level 1 - Top 5 Bids and Asks, from kitemodels.Tick.Depth)
    -- Prices use NUMERIC for precision; Quantities and Orders use INTEGER (from uint32)

    -- Bid Side (buyers)
    bid_price1 NUMERIC NOT NULL,
    bid_quantity1 INTEGER NOT NULL,
    bid_orders1 INTEGER NOT NULL,

    bid_price2 NUMERIC NOT NULL,
    bid_quantity2 INTEGER NOT NULL,
    bid_orders2 INTEGER NOT NULL,

    bid_price3 NUMERIC NOT NULL,
    bid_quantity3 INTEGER NOT NULL,
    bid_orders3 INTEGER NOT NULL,

    bid_price4 NUMERIC NOT NULL,
    bid_quantity4 INTEGER NOT NULL,
    bid_orders4 INTEGER NOT NULL,

    bid_price5 NUMERIC NOT NULL,
    bid_quantity5 INTEGER NOT NULL,
    bid_orders5 INTEGER NOT NULL,

    -- Ask Side (sellers)
    ask_price1 NUMERIC NOT NULL,
    ask_quantity1 INTEGER NOT NULL,
    ask_orders1 INTEGER NOT NULL,

    ask_price2 NUMERIC NOT NULL,
    ask_quantity2 INTEGER NOT NULL,
    ask_orders2 INTEGER NOT NULL,

    ask_price3 NUMERIC NOT NULL,
    ask_quantity3 INTEGER NOT NULL,
    ask_orders3 INTEGER NOT NULL,

    ask_price4 NUMERIC NOT NULL,
    ask_quantity4 INTEGER NOT NULL,
    ask_orders4 INTEGER NOT NULL,

    ask_price5 NUMERIC NOT NULL,
    ask_quantity5 INTEGER NOT NULL,
    ask_orders5 INTEGER NOT NULL,

    -- Other aggregated quantities (from kitemodels.Tick)
    total_buy_quantity INTEGER NOT NULL, -- Go's uint32 maps to INTEGER
    total_sell_quantity INTEGER NOT NULL, -- Go's uint32 maps to INTEGER
    data_source VARCHAR(20) NOT NULL,
    
    -- Define the composite primary key
    PRIMARY KEY (instrument_token, timestamp, tick_sequence_id)
);

-- Convert to TimescaleDB hypertable (idempotent with if_not_exists)
-- 'timestamp' is the time column, 'instrument_token' is the partitioning key
-- Consider adjusting chunk_time_interval and number_partitions based on your data rate and query patterns.
SELECT create_hypertable('market_data', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day', 
    migrate_data => TRUE, 
    partitioning_column => 'instrument_token', 
    number_partitions => 8,
    if_not_exists => TRUE
);

-- Add foreign key constraint to instruments table
-- This assumes an 'instruments' table exists with 'instrument_token' as a unique column.
ALTER TABLE market_data ADD CONSTRAINT fk_market_data_instrument_token
FOREIGN KEY (instrument_token) REFERENCES instruments(instrument_token) ON DELETE RESTRICT;

-- Index for efficient time-range and instrument-specific queries
-- This index is crucial for performance when querying historical data.
CREATE INDEX IF NOT EXISTS idx_market_data_instrument_token_timestamp ON market_data (instrument_token, timestamp DESC);