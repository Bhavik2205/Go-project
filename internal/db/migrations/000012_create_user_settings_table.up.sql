CREATE TABLE user_settings (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,
    user_id BIGINT NOT NULL,
    section VARCHAR(100) NOT NULL,       -- e.g. 'zerodha', 'notifications', 'general', 'strategy', 'data', 'performance'
    settings_json JSONB NOT NULL DEFAULT '{}',
    CONSTRAINT fk_user_settings_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE (user_id, section)
);

CREATE INDEX idx_user_settings_user_id ON user_settings (user_id);
