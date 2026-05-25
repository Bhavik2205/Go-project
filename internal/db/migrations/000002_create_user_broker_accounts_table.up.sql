CREATE TABLE user_broker_accounts (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,
    user_id BIGINT NOT NULL,
    broker_type VARCHAR(50) NOT NULL,
    api_key VARCHAR(255) NOT NULL,
    access_token BYTEA NOT NULL, -- BYTEA for encrypted binary data
    public_token BYTEA NOT NULL, -- BYTEA for encrypted binary data
    request_token VARCHAR(255),
    session_expiry TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    account_name VARCHAR(255),
    broker_user_id VARCHAR(255),
    CONSTRAINT fk_user_broker_accounts_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE (user_id, broker_type, broker_user_id) -- Ensures one unique account per user for a given broker_user_id
);

CREATE INDEX idx_user_broker_accounts_user_id ON user_broker_accounts (user_id);
CREATE INDEX idx_user_broker_accounts_broker_type ON user_broker_accounts (broker_type);