// internal/cache/redis.go
package cache

import (
	"context"
	"fmt"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"github.com/redis/go-redis/v9" // Using context-aware client
	"go.uber.org/zap"
)

// RedisClient represents the Redis client
type RedisClient struct {
	*redis.Client
	context context.Context
}

// NewRedisClient initializes and returns a new Redis client
func NewRedisClient(cfg *utils.RedisConfig) (*RedisClient, error) {
	rdb := redis.NewClient(&redis.Options{
		Addr:     fmt.Sprintf("%s:%s", cfg.Host, cfg.Port),
		Password: cfg.Password, // no password set
		DB:       0,            // use default DB
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err := rdb.Ping(ctx).Result()
	if err != nil {
		zap.L().Error("Failed to connect to Redis", zap.String("host", cfg.Host), zap.String("port", cfg.Port), zap.Error(err))
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	zap.L().Info("Connected to Redis", zap.String("host", cfg.Host), zap.String("port", cfg.Port))
	return &RedisClient{
		Client:  rdb,
		context: context.Background(), // Use a long-lived context for the client
	}, nil
}

// Set stores a key-value pair in Redis with an expiration
func (r *RedisClient) Set(key string, value interface{}, expiration time.Duration) error {
	return r.Client.Set(r.context, key, value, expiration).Err()
}

// Get retrieves a value from Redis
func (r *RedisClient) Get(key string) (string, error) {
	return r.Client.Get(r.context, key).Result()
}

// Delete removes a key from Redis
func (r *RedisClient) Delete(key string) error {
	return r.Client.Del(r.context, key).Err()
}

// Publish publishes a message to a Redis channel
func (r *RedisClient) Publish(channel string, message interface{}) error {
	return r.Client.Publish(r.context, channel, message).Err()
}

// Subscribe subscribes to a Redis channel and returns a PubSub struct
func (r *RedisClient) Subscribe(ctx context.Context, channels ...string) *redis.PubSub {
	return r.Client.Subscribe(ctx, channels...)
}
