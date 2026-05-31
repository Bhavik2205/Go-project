package api

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/cache" // Ensure this path is correct
	"github.com/Bhavik2205/ML-Bot/internal/marketdata"
	kitemodels "github.com/zerodha/gokiteconnect/v4/models"
	"go.uber.org/zap"
)

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
func (s *SimulatedZerodhaClient) SimulateTicks(ctx context.Context, infos []*InstrumentInfo, redisClient *cache.RedisClient, simulationSpeedMultiplier float64) error { // Validate required inputs
	if redisClient == nil {
		return fmt.Errorf("RedisClient is nil, cannot publish simulated ticks")
	}
	if len(infos) == 0 {
		return fmt.Errorf("no instruments provided for simulation")
	}

	zap.L().Info("🚀 Starting market data simulation...")

	// Helper functions for rounding
	round1 := func(val float64) float64 {
		return math.Round(val*100) / 100 // Round to 2 decimal places for price
	}
	round4 := func(val float64) float64 {
		return math.Round(val*10000) / 10000 // Round to 4 decimal places for percent change
	}

	// Maps to store and update simulation data per instrument token
	tokenToLabel := make(map[uint32]string)
	currentPrices := make(map[uint32]float64)
	currentVolumes := make(map[uint32]uint32)
	// New map to store the 'previous day's close' for each instrument
	previousDayCloses := make(map[uint32]float64)

	ohlcData := make(map[uint32]struct {
		Open  float64
		High  float64
		Low   float64
		Close float64 // Represents the LastPrice of the current tick
	})

	// Initialize simulation data for each instrument
	for _, info := range infos {
		tokenToLabel[info.Token] = fmt.Sprintf("%s:%s", info.Exchange, info.Symbol)

		// Set a reasonable random initial price for the simulation
		basePrice := 100.0 + rand.Float64()*1000 // Initial price for the *current* simulated day

		// Set previous day's close distinct from the current day's open/initial price
		// This creates a potential for non-zero percent change from the very first tick
		// For example, prevClose could be basePrice * (1 + random_change_percentage)
		initialPrevClose := round1(basePrice * (1 + (rand.Float64()-0.5)*0.05)) // +/- 2.5% from base
		if initialPrevClose <= 0 {                                              // Ensure it's never zero or negative
			initialPrevClose = basePrice * 0.98 // Fallback to a valid positive value
		}

		currentPrices[info.Token] = basePrice
		currentVolumes[info.Token] = uint32(rand.Intn(50000) + 1000)
		previousDayCloses[info.Token] = initialPrevClose // Store the initial previous day's close

		// Initialize OHLC data. Open is the first price of the current simulated day.
		ohlcData[info.Token] = struct {
			Open  float64
			High  float64
			Low   float64
			Close float64
		}{
			Open:  basePrice,
			High:  basePrice,
			Low:   basePrice,
			Close: basePrice, // Initial close is same as open for the first tick
		}
	}

	// Define the total simulated market duration and the interval between ticks in simulated time
	simulatedMarketDuration := 6*time.Hour + 15*time.Minute // Mimic 9:15 AM to 3:30 PM (6 hours 15 minutes)
	tickIntervalSimulated := 500 * time.Millisecond         // Generate a new "tick" every 500ms of simulated time

	// Calculate the real-time delay needed between publishing ticks to achieve the desired speed multiplier
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

	// simulationStartTime tracks the real-world start of the current simulated day.
	// It is reset each time a simulated market day completes.
	simulationStartTime := time.Now()

	// Main simulation loop — restarts each simulated market day continuously
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

			// When a simulated market day ends, roll over to the next day instead of stopping
			if elapsedSimulatedTime >= simulatedMarketDuration {
				zap.L().Info("✅ Simulated market day completed — restarting for next day")
				for _, info := range infos {
					previousDayCloses[info.Token] = ohlcData[info.Token].Close
					ohlcData[info.Token] = struct {
						Open  float64
						High  float64
						Low   float64
						Close float64
					}{
						Open:  currentPrices[info.Token],
						High:  currentPrices[info.Token],
						Low:   currentPrices[info.Token],
						Close: currentPrices[info.Token],
					}
				}
				simulationStartTime = time.Now()
				continue
			}

			// Generate and publish a tick for each subscribed instrument
			for _, info := range infos {
				token := info.Token
				label := tokenToLabel[token]

				// --- Simulate Price Movement ---
				price := round1(currentPrices[token])
				// Generate a small random change, typically within +/- 0.1% of the current price
				priceChange := (rand.Float64()*2 - 1) * (price * 0.004) // Small random fluctuation
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
				currentVolumes[token] += uint32(rand.Intn(1000) + 50) // Increment volume by 50-1049

				// --- Update OHLC (Open, High, Low, Close) ---
				ohlc := ohlcData[token]
				// Initialize High and Low for the first tick, or update
				if ohlc.High == 0 || price > ohlc.High {
					ohlc.High = price
				}
				if ohlc.Low == 0 || price < ohlc.Low {
					ohlc.Low = price
				}
				ohlc.Close = price // The close price for this tick is its last price
				ohlcData[token] = ohlc

				// --- Simulate Depth (Bids and Asks) ---
				depth := kitemodels.Depth{}
				var totalBuyQty, totalSellQty uint32
				for i := 0; i < 5; i++ {
					priceOffset := round1(float64(i+1) * 0.05) // Small price difference for bids/asks
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
				averageTradePrice := round1((ohlc.Open + ohlc.Close) / 2) // Simple average for simulation

				// --- Calculate NetChange and PercentChange ---
				// NetChange is LastPrice - PrevClose
				netChange := price - previousDayCloses[token]
				percentChange := 0.0
				if previousDayCloses[token] != 0 {
					percentChange = (netChange / previousDayCloses[token]) * 100.0
				}
				netChange = round1(netChange)         // Round net change to 2 decimals
				percentChange = round4(percentChange) // Round percent change to 4 decimals

				// Build canonical NormalizedTick
				normalized := marketdata.NormalizedTick{
					InstrumentToken:    token,
					Symbol:             label,
					Exchange:           info.Exchange,
					EventTime:          time.Now(), // simulated time – use current time as event time
					IngestTime:         time.Now(),
					LastPrice:          price,
					LastTradedQuantity: lastTradedQty,
					Volume:             currentVolumes[token],
					AverageTradePrice:  averageTradePrice,
					NetChange:          netChange,
					PercentChange:      percentChange,
					PrevClose:          previousDayCloses[token],
					OHLC: kitemodels.OHLC{
						Open:  ohlc.Open,
						High:  ohlc.High,
						Low:   ohlc.Low,
						Close: previousDayCloses[token], // OHLC close is previous close
					},
					Depth:             depth,
					TotalBuyQuantity:  totalBuyQty,
					TotalSellQuantity: totalSellQty,
					OpenInterest:      0,
				}

				// --- Prepare and Publish to Redis ---
				enrichedTick := struct {
					Symbol           string                    `json:"symbol"`
					ProcessedAtNanos int64                     `json:"processed_at_nanos"`
					Tick             marketdata.NormalizedTick `json:"tick"`
				}{
					Symbol:           label,
					ProcessedAtNanos: time.Now().UnixNano(),
					Tick:             normalized,
				}

				jsonData, err := json.Marshal(enrichedTick)
				if err != nil {
					zap.L().Error("❌ Failed to marshal simulated tick data for Redis",
						zap.Uint32("instrument_token", token),
						zap.Error(err),
					)
					continue
				}

				if err := redisClient.Publish(RedisMarketDataChannel, jsonData); err != nil {
					zap.L().Error("❌ Failed to publish simulated tick to Redis",
						zap.Uint32("instrument_token", token),
						zap.Error(err),
					)
				} else {
					zap.L().Debug("Published simulated tick",
						zap.String("symbol", label),
						zap.Float64("ltp", normalized.LastPrice),
						zap.Float64("prev_close", normalized.PrevClose),
						zap.Float64("net_change", normalized.NetChange),
						zap.Float64("percent_change", normalized.PercentChange),
					)
				}
			}

			// Pause for the calculated real-time delay before the next set of ticks
			time.Sleep(realTimeDelay)
		}
	}
}
