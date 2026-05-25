CREATE TABLE audit_events (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_id BIGINT,                               -- NULL for system-level events
    event_type VARCHAR(100) NOT NULL,             -- 'LOGIN', 'LOGOUT', 'SIGNUP', 'BROKER_CONNECT', 'BROKER_DISCONNECT', 'ORDER_PLACE', 'ORDER_CANCEL', 'SETTINGS_UPDATE', 'ADMIN_ACTION'
    resource_type VARCHAR(100),                   -- 'order', 'broker_account', 'user_settings', etc.
    resource_id VARCHAR(255),                     -- ID of the affected resource
    action VARCHAR(50) NOT NULL,                  -- 'CREATE', 'UPDATE', 'DELETE', 'READ', 'EXECUTE'
    status VARCHAR(50) NOT NULL DEFAULT 'SUCCESS', -- 'SUCCESS', 'FAILURE'
    ip_address VARCHAR(45),                       -- IPv4 or IPv6
    user_agent TEXT,
    request_id VARCHAR(255),
    metadata JSONB,                               -- additional context: old/new values, broker response, etc.
    error_message TEXT,
    CONSTRAINT fk_audit_events_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX idx_audit_events_user_id ON audit_events (user_id);
CREATE INDEX idx_audit_events_event_type ON audit_events (event_type);
CREATE INDEX idx_audit_events_created_at ON audit_events (created_at DESC);
CREATE INDEX idx_audit_events_resource ON audit_events (resource_type, resource_id);
