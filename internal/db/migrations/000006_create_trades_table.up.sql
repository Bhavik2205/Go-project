CREATE TABLE trades (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,
    order_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    instrument_token INTEGER NOT NULL,
    trade_id VARCHAR(255) NOT NULL UNIQUE, -- Broker's trade ID
    transaction_type VARCHAR(10) NOT NULL, -- BUY or SELL
    quantity INTEGER NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    trade_time TIMESTAMP WITH TIME ZONE NOT NULL,
    exchange VARCHAR(50),
    trade_type VARCHAR(10) NOT NULL,   -- <-- NEW COLUMN
    CONSTRAINT fk_trades_order_id FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE RESTRICT,
    CONSTRAINT fk_trades_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE RESTRICT,
    CONSTRAINT fk_trades_instrument_token FOREIGN KEY (instrument_token) REFERENCES instruments(instrument_token) ON DELETE RESTRICT
);

CREATE INDEX idx_trades_order_id ON trades (order_id);
CREATE INDEX idx_trades_user_id ON trades (user_id);
CREATE INDEX idx_trades_trade_time ON trades (trade_time DESC);