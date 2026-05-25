CREATE TABLE user_strategies (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,
    user_id BIGINT NOT NULL,
    strategy_name VARCHAR(255) NOT NULL,
    is_enabled BOOLEAN DEFAULT FALSE,
    parameters JSONB, -- JSONB for flexible strategy parameters
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_user_strategies_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE (user_id, strategy_name) -- A user can only have one configuration per strategy
);

CREATE INDEX idx_user_strategies_user_id ON user_strategies (user_id);
CREATE INDEX idx_user_strategies_is_enabled ON user_strategies (is_enabled);