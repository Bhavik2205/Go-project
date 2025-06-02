CREATE TABLE metrics (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,
    name VARCHAR(255) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    labels JSONB -- JSONB for additional labels/metadata
);

CREATE INDEX idx_metrics_name ON metrics (name);
CREATE INDEX idx_metrics_timestamp ON metrics (timestamp DESC);