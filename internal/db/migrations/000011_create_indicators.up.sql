CREATE TABLE IF NOT EXISTS ohlcv_candles (
    instrument_token INTEGER NOT NULL,
    interval VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    trade_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (instrument_token, interval, timestamp)
);

-- Convert ohlcv_candles to a TimescaleDB hypertable
-- This needs to be run *after* the table is created and the timescaledb extension is enabled.
SELECT create_hypertable('ohlcv_candles', 'timestamp', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

-- Add index for querying candles by instrument and interval
CREATE INDEX IF NOT EXISTS idx_ohlcv_candles_instrument_interval ON ohlcv_candles (instrument_token, interval, timestamp DESC);

---
-- Table for Simple Moving Average (SMA)
---
CREATE TABLE IF NOT EXISTS indicator_smas (
    instrument_token INTEGER NOT NULL,
    interval VARCHAR(10) NOT NULL,
    period INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    value NUMERIC NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (instrument_token, interval, period, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_indicator_smas_instrument_interval_period ON indicator_smas (instrument_token, interval, period, timestamp DESC);

---
-- Table for Exponential Moving Average (EMA)
---
CREATE TABLE IF NOT EXISTS indicator_emas (
    instrument_token INTEGER NOT NULL,
    interval VARCHAR(10) NOT NULL,
    period INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    value NUMERIC NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (instrument_token, interval, period, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_indicator_emas_instrument_interval_period ON indicator_emas (instrument_token, interval, period, timestamp DESC);

---
-- Table for Moving Average Convergence Divergence (MACD)
---
CREATE TABLE IF NOT EXISTS indicator_macds (
    instrument_token INTEGER NOT NULL,
    interval VARCHAR(10) NOT NULL,
    fast_period INTEGER NOT NULL,
    slow_period INTEGER NOT NULL,
    signal_period INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    macd_line NUMERIC NOT NULL,
    signal_line NUMERIC NOT NULL,
    histogram NUMERIC NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (instrument_token, interval, fast_period, slow_period, signal_period, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_indicator_macds_instrument_interval_periods ON indicator_macds (instrument_token, interval, fast_period, slow_period, signal_period, timestamp DESC);

---
-- Table for Average True Range (ATR)
---
CREATE TABLE IF NOT EXISTS indicator_atrs (
    instrument_token INTEGER NOT NULL,
    interval VARCHAR(10) NOT NULL,
    period INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    value NUMERIC NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (instrument_token, interval, period, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_indicator_atrs_instrument_interval_period ON indicator_atrs (instrument_token, interval, period, timestamp DESC);

---
-- Table for Relative Strength Index (RSI)
---
CREATE TABLE IF NOT EXISTS indicator_rsis (
    instrument_token INTEGER NOT NULL,
    interval VARCHAR(10) NOT NULL,
    period INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    value NUMERIC NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (instrument_token, interval, period, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_indicator_rsis_instrument_interval_period ON indicator_rsis (instrument_token, interval, period, timestamp DESC);

---
-- Table for Stochastic Oscillator
---
CREATE TABLE IF NOT EXISTS indicator_stochastics (
    instrument_token INTEGER NOT NULL,
    interval VARCHAR(10) NOT NULL,
    k_period INTEGER NOT NULL,
    d_period INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    k_value NUMERIC NOT NULL,
    d_value NUMERIC NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (instrument_token, interval, k_period, d_period, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_indicator_stochastics_instrument_interval_periods ON indicator_stochastics (instrument_token, interval, k_period, d_period, timestamp DESC);

---
-- Table for Bollinger Bands
---
CREATE TABLE IF NOT EXISTS indicator_bollinger_bands (
    instrument_token INTEGER NOT NULL,
    interval VARCHAR(10) NOT NULL,
    period INTEGER NOT NULL,
    num_std_dev NUMERIC(5,2) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    upper_band NUMERIC NOT NULL,
    middle_band NUMERIC NOT NULL,
    lower_band NUMERIC NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (instrument_token, interval, period, num_std_dev, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_indicator_bollinger_bands_instrument_interval_params ON indicator_bollinger_bands (instrument_token, interval, period, num_std_dev, timestamp DESC);

---
-- Table for On-Balance Volume (OBV)
---
CREATE TABLE IF NOT EXISTS indicator_obvs (
    instrument_token INTEGER NOT NULL,
    interval VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    value NUMERIC NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (instrument_token, interval, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_indicator_obvs_instrument_interval ON indicator_obvs (instrument_token, interval, timestamp DESC);

---
-- Table for Volume Weighted Average Price (VWAP)
---
CREATE TABLE IF NOT EXISTS indicator_vwaps (
    instrument_token INTEGER NOT NULL,
    interval VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    value NUMERIC NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (instrument_token, interval, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_indicator_vwaps_instrument_interval ON indicator_vwaps (instrument_token, interval, timestamp DESC);

---
-- Table for Average Directional Index (ADX)
---
CREATE TABLE IF NOT EXISTS indicator_adxes (
    instrument_token INTEGER NOT NULL,
    interval VARCHAR(10) NOT NULL,
    period INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    adx_value NUMERIC NOT NULL,
    plus_di NUMERIC NOT NULL,
    minus_di NUMERIC NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (instrument_token, interval, period, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_indicator_adxes_instrument_interval_period ON indicator_adxes (instrument_token, interval, period, timestamp DESC);
