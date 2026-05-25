CREATE TABLE instruments (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,
    instrument_token INTEGER NOT NULL UNIQUE,
    exchange VARCHAR(50) NOT NULL,
    tradingsymbol VARCHAR(255) NOT NULL,
    instrument_type VARCHAR(50),
    name VARCHAR(255),
    segment VARCHAR(50),
    tick_size DOUBLE PRECISION,
    lot_size INTEGER,
    expiry TIMESTAMP WITH TIME ZONE,
    strike DOUBLE PRECISION,
    option_type VARCHAR(10),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (tradingsymbol, exchange)
);

-- Add index on instrument_token for fast lookups
CREATE INDEX idx_instruments_instrument_token ON instruments (instrument_token);
CREATE INDEX idx_instruments_tradingsymbol_exchange ON instruments (tradingsymbol, exchange);
CREATE INDEX idx_instruments_instrument_type ON instruments (instrument_type);

-- Optional: Create a separate primary key index if you want to keep id as PK
-- If you prefer instrument_token as PK instead of id, uncomment:
-- ALTER TABLE instruments DROP CONSTRAINT instruments_pkey CASCADE;
-- ALTER TABLE instruments ADD PRIMARY KEY (instrument_token);
-- ALTER TABLE instruments DROP COLUMN id;