CREATE TABLE watchlists (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,
    user_id BIGINT NOT NULL,
    name VARCHAR(255) NOT NULL DEFAULT 'Default',
    CONSTRAINT fk_watchlists_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE (user_id, name)
);

CREATE TABLE watchlist_items (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,
    watchlist_id BIGINT NOT NULL,
    instrument_token INTEGER NOT NULL,
    symbol VARCHAR(255) NOT NULL,        -- e.g. 'NSE:RELIANCE'
    CONSTRAINT fk_watchlist_items_watchlist_id FOREIGN KEY (watchlist_id) REFERENCES watchlists(id) ON DELETE CASCADE,
    CONSTRAINT fk_watchlist_items_instrument_token FOREIGN KEY (instrument_token) REFERENCES instruments(instrument_token) ON DELETE RESTRICT,
    UNIQUE (watchlist_id, instrument_token)
);

CREATE INDEX idx_watchlists_user_id ON watchlists (user_id);
CREATE INDEX idx_watchlist_items_watchlist_id ON watchlist_items (watchlist_id);
