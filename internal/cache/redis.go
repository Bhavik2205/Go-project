// internal/cache/redis.go
package cache

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"github.com/redis/go-redis/v9" // Using context-aware client
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
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	log.Println("✅ Connected to Redis.")
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
