CREATE TABLE positions (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,
    user_id BIGINT NOT NULL,
    instrument_token INTEGER NOT NULL,
    trading_symbol VARCHAR(255) NOT NULL,
    product VARCHAR(50) NOT NULL,
    quantity INTEGER NOT NULL,
    average_price DOUBLE PRECISION NOT NULL,
    last_price DOUBLE PRECISION,
    realized_pnl DOUBLE PRECISION DEFAULT 0.0,
    unrealized_pnl DOUBLE PRECISION DEFAULT 0.0,
    CONSTRAINT fk_positions_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE RESTRICT,
    CONSTRAINT fk_positions_instrument_token FOREIGN KEY (instrument_token) REFERENCES instruments(instrument_token) ON DELETE RESTRICT,
    UNIQUE (user_id, instrument_token, product) -- A user has one position per instrument per product type
);

CREATE INDEX idx_positions_user_id ON positions (user_id);
CREATE INDEX idx_positions_instrument_token ON positions (instrument_token);