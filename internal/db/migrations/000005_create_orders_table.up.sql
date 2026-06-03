CREATE TABLE orders (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,
    user_id BIGINT NOT NULL,
    instrument_token INTEGER NOT NULL,
    broker_order_id VARCHAR(255) NOT NULL UNIQUE,
    strategy_name VARCHAR(255) NOT NULL,
    order_type VARCHAR(50) NOT NULL,
    transaction_type VARCHAR(10) NOT NULL, -- BUY or SELL
    quantity INTEGER NOT NULL,
    price DOUBLE PRECISION,
    trigger_price DOUBLE PRECISION,
    status VARCHAR(50) NOT NULL, -- PENDING, OPEN, FILLED, CANCELLED, REJECTED
    placed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    filled_quantity INTEGER DEFAULT 0,
    filled_price DOUBLE PRECISION DEFAULT 0.0,
    valid_until TIMESTAMP WITH TIME ZONE,
    product VARCHAR(50),
    exchange_order_id VARCHAR(255),
    tag VARCHAR(255),
    CONSTRAINT fk_orders_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE RESTRICT,
    CONSTRAINT fk_orders_instrument_token FOREIGN KEY (instrument_token) REFERENCES instruments(instrument_token) ON DELETE RESTRICT
);

CREATE INDEX idx_orders_user_id_status ON orders (user_id, status);
CREATE INDEX idx_orders_broker_order_id ON orders (broker_order_id);
CREATE INDEX idx_orders_placed_at ON orders (placed_at DESC);