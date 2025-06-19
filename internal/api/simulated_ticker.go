package api

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/cache" // Ensure this path is correct
	kitemodels "github.com/zerodha/gokiteconnect/v4/models"
	"go.uber.org/zap"
)

type SimTick struct {
	InstrumentToken    uint32           `json:"InstrumentToken"`
	Timestamp          string           `json:"Timestamp"`
	LastPrice          float64          `json:"LastPrice"`
	OHLC               kitemodels.OHLC  `json:"OHLC"`
	VolumeTraded       uint32           `json:"VolumeTraded"`
	Volume             uint32           `json:"Volume"`
	Depth              kitemodels.Depth `json:"Depth"`
	TotalBuyQuantity   uint32           `json:"TotalBuyQuantity"`
	TotalSellQuantity  uint32           `json:"TotalSellQuantity"`
	LastTradedQuantity uint32           `json:"LastTradedQuantity"` // <-- Added
	AverageTradePrice  float64          `json:"AverageTradePrice"`  // <-- Added
	NetChange          float64          `json:"NetChange"`          // <-- Added
}

// SimulatedZerodhaClient mimics the ZerodhaClient for simulation purposes.
// It doesn't connect to any external API but generates synthetic ticks.
type SimulatedZerodhaClient struct {
	// No actual KiteTicker or external dependencies needed for simulation
}

// NewSimulatedZerodhaClient creates a new instance of SimulatedZerodhaClient.
func NewSimulatedZerodhaClient() *SimulatedZerodhaClient {
	return &SimulatedZerodhaClient{}
}

// SimulateTicks generates and publishes synthetic market ticks to Redis.
// It takes a context for graceful shutdown, a list of instruments to simulate,
// the Redis client for publishing, and a multiplier to control simulation speed.
func (s *SimulatedZerodhaClient) SimulateTicks(ctx context.Context, infos []*InstrumentInfo, redisClient *cache.RedisClient, simulationSpeedMultiplier float64) error {
	// Validate required inputs
	if redisClient == nil {
		return fmt.Errorf("RedisClient is nil, cannot publish simulated ticks")
	}
	if len(infos) == 0 {
		return fmt.Errorf("no instruments provided for simulation")
	}

	zap.L().Info("🚀 Starting market data simulation...")

	// Helper functions for rounding
	round1 := func(val float64) float64 {
		return math.Round(val*10) / 10
	}

	// Maps to store and update simulation data per instrument token
	tokenToLabel := make(map[uint32]string)
	currentPrices := make(map[uint32]float64)
	currentVolumes := make(map[uint32]uint32)
	ohlcData := make(map[uint32]struct {
		Open  float64
		High  float64
		Low   float64
		Close float64 // Represents the LastPrice of the current tick
	})

	// Initialize simulation data for each instrument
	for _, info := range infos {
		tokenToLabel[info.Token] = fmt.Sprintf("%s (%s)", info.Symbol, info.Exchange)

		// Set a reasonable random initial price for the simulation
		basePrice := 100.0 + rand.Float64()*1000 // Example: prices between 100 and 1100
		currentPrices[info.Token] = basePrice
		currentVolumes[info.Token] = uint32(rand.Intn(50000) + 1000) // Initial random volume

		// Initialize OHLC data with the base price
		ohlcData[info.Token] = struct {
			Open  float64
			High  float64
			Low   float64
			Close float64
		}{
			Open:  basePrice,
			High:  basePrice,
			Low:   basePrice,
			Close: basePrice,
		}
	}

	// Define the total simulated market duration and the interval between ticks in simulated time
	simulatedMarketDuration := 6*time.Hour + 15*time.Minute // Mimic 9:15 AM to 3:30 PM (6 hours 15 minutes)
	tickIntervalSimulated := 500 * time.Millisecond         // Generate a new "tick" every 500ms of simulated time

	// Calculate the real-time delay needed between publishing ticks to achieve the desired speed multiplier
	// If simulationSpeedMultiplier is 1.0, 1 simulated second = 1 real second.
	// If simulationSpeedMultiplier is 10.0, 10 simulated seconds = 1 real second, so delay is 1/10th.
	realTimeDelay := time.Duration(float64(tickIntervalSimulated) / simulationSpeedMultiplier)

	// Ensure a minimum delay to prevent the loop from spinning too fast
	if realTimeDelay < time.Millisecond {
		realTimeDelay = time.Millisecond
		zap.L().Warn("Simulation speed multiplier too high, clamping real-time delay to 1ms to prevent busy-loop.")
	}

	zap.L().Info("Simulation parameters:",
		zap.String("simulated_market_duration", simulatedMarketDuration.String()),
		zap.String("tick_interval_simulated", tickIntervalSimulated.String()),
		zap.Float64("simulation_speed_multiplier", simulationSpeedMultiplier),
		zap.String("real_time_delay_between_ticks", realTimeDelay.String()),
	)

	// Start time of the simulation (real-world time when simulation began)
	simulationStartTime := time.Now()

	// Main simulation loop
	for {
		select {
		case <-ctx.Done():
			// Exit the simulation if the context is cancelled (e.g., on SIGINT/SIGTERM)
			zap.L().Info("🚫 Market data simulation stopped by context cancellation.")
			return nil
		default:
			// Calculate the elapsed simulated time based on the real-world elapsed time and the multiplier
			elapsedRealTime := time.Since(simulationStartTime)
			elapsedSimulatedTime := time.Duration(float64(elapsedRealTime) * simulationSpeedMultiplier)

			// Check if the total simulated market duration has been reached
			if elapsedSimulatedTime >= simulatedMarketDuration {
				zap.L().Info("✅ Simulated market duration completed.")
				return nil
			}

			// Generate and publish a tick for each subscribed instrument
			for _, info := range infos {
				token := info.Token
				label := tokenToLabel[token]

				// --- Simulate Price Movement ---
				price := round1(currentPrices[token])
				// Generate a small random change, typically within +/- 0.1% of the current price
				priceChange := (rand.Float64()*2 - 1) * (price * 0.001)
				price += priceChange
				price = round1(price)

				// Ensure price remains positive and within a sensible range for demonstration
				if price < 0.1 {
					price = 0.1 // Prevent price from going too low
				}
				currentPrices[token] = price // Update the current price for the next tick

				// --- Simulate LastTradedQuantity ---
				lastTradedQty := uint32(rand.Intn(500) + 1) // Random trade size

				// --- Simulate Volume Increase ---
				// Add a random amount of volume to simulate trades
				currentVolumes[token] += uint32(rand.Intn(1000) + 50) // Increment volume by 50-1049

				// --- Update OHLC (Open, High, Low, Close) ---
				ohlc := ohlcData[token]
				// Initialize High and Low for the first tick
				if ohlc.High == 0 || price > ohlc.High {
					ohlc.High = price
				}
				if ohlc.Low == 0 || price < ohlc.Low {
					ohlc.Low = price
				}
				ohlc.Close = price // The close price for this tick is its last price
				ohlcData[token] = ohlc

				depth := kitemodels.Depth{}
				var totalBuyQty, totalSellQty uint32
				for i := 0; i < 5; i++ {
					priceOffset := round1(float64(i+1) * 0.05 * price)
					buyQty := uint32(rand.Intn(1000) + 1)
					sellQty := uint32(rand.Intn(1000) + 1)
					depth.Buy[i] = kitemodels.DepthItem{
						Price:    round1(price - priceOffset),
						Quantity: buyQty,
						Orders:   uint32(rand.Intn(10) + 1),
					}
					depth.Sell[i] = kitemodels.DepthItem{
						Price:    round1(price + priceOffset),
						Quantity: sellQty,
						Orders:   uint32(rand.Intn(10) + 1),
					}
					totalBuyQty += buyQty
					totalSellQty += sellQty
				}

				// --- Simulate AverageTradePrice ---
				averageTradePrice := round1((ohlc.Open + ohlc.Close) / 2)

				// --- Simulate NetChange (keep 4 decimals as requested) ---
				netChange := price - ohlc.Open

				// --- Construct the kitemodels.Tick ---
				tick := SimTick{
					InstrumentToken: token,
					Timestamp:       time.Now().Format(time.RFC3339Nano),
					LastPrice:       round1(price),
					OHLC: kitemodels.OHLC{
						Open:  round1(ohlc.Open),
						High:  round1(ohlc.High),
						Low:   round1(ohlc.Low),
						Close: round1(ohlc.Close),
					},
					VolumeTraded:       currentVolumes[token],
					Volume:             currentVolumes[token],
					Depth:              depth,        // <-- Add this line
					TotalBuyQuantity:   totalBuyQty,  // <-- Add this field
					TotalSellQuantity:  totalSellQty, // <-- Add this field
					LastTradedQuantity: lastTradedQty,
					AverageTradePrice:  averageTradePrice,
					NetChange:          netChange,
				}

				// --- Prepare and Publish to Redis ---
				// Create an enriched tick structure similar to the real Zerodha ticker
				enrichedTick := struct {
					Symbol           string      `json:"symbol"`
					ProcessedAtNanos int64       `json:"processed_at_nanos"` // Real-world timestamp of processing
					Tick             interface{} `json:"tick"`
				}{
					Symbol:           label,
					ProcessedAtNanos: time.Now().UnixNano(),
					Tick:             tick,
				}

				if jsonData, err := json.Marshal(enrichedTick); err == nil {
					err := redisClient.Publish(RedisMarketDataChannel, jsonData)
					if err != nil {
						zap.L().Error("❌ Failed to publish simulated tick to Redis",
							zap.Uint32("instrument_token", tick.InstrumentToken),
							zap.Error(err),
						)
					} else {
						// Optionally log published ticks for debugging, comment out for production verbosity
						// zap.L().Debug("Published simulated tick",
						// 	zap.String("symbol", label),
						// 	zap.Float64("price", tick.LastPrice),
						// 	zap.Uint32("volume", tick.VolumeTraded),
						// )
					}
				} else {
					zap.L().Error("❌ Failed to marshal enriched simulated tick data for Redis",
						zap.Uint32("instrument_token", tick.InstrumentToken),
						zap.Error(err),
					)
				}
			}
			// Pause for the calculated real-time delay before the next set of ticks
			time.Sleep(realTimeDelay)
		}
	}
}
