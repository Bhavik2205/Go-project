CREATE TABLE notification_channels (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,
    user_id BIGINT NOT NULL,
    channel_type VARCHAR(50) NOT NULL,            -- 'telegram', 'whatsapp'
    is_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    config JSONB NOT NULL DEFAULT '{}',           -- encrypted config: botToken/chatId for telegram, apiUrl/phoneNumber for whatsapp
    CONSTRAINT fk_notification_channels_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE (user_id, channel_type)
);

CREATE TABLE notification_history (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_id BIGINT NOT NULL,
    channel_type VARCHAR(50) NOT NULL,
    event_type VARCHAR(100) NOT NULL,             -- 'TRADE_EXECUTION', 'PNL_THRESHOLD', 'ERROR_ALERT', 'TEST'
    message TEXT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'PENDING', -- PENDING, SENT, FAILED
    provider_message_id VARCHAR(255),
    error_message TEXT,
    sent_at TIMESTAMP WITH TIME ZONE,
    CONSTRAINT fk_notification_history_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX idx_notification_channels_user_id ON notification_channels (user_id);
CREATE INDEX idx_notification_history_user_id ON notification_history (user_id);
CREATE INDEX idx_notification_history_created_at ON notification_history (created_at DESC);
