CREATE TABLE news_articles (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,
    source VARCHAR(255) NOT NULL,
    title VARCHAR(512) NOT NULL,
    description TEXT,
    content TEXT,
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    url TEXT NOT NULL UNIQUE, -- Use TEXT for URL if very long, otherwise VARCHAR(2048) or similar
    image_url TEXT,
    sentiment_score DOUBLE PRECISION,
    sentiment_label VARCHAR(50),
    analyzed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_news_articles_published_at ON news_articles (published_at DESC);
CREATE INDEX idx_news_articles_sentiment_label ON news_articles (sentiment_label);