// internal/data/candles.go
package data

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api"
	"github.com/Bhavik2205/ML-Bot/internal/cache"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/utils"
	"github.com/gorilla/websocket"
	kitemodels "github.com/zerodha/gokiteconnect/v4/models"
	"go.uber.org/zap"
	"gorm.io/gorm/clause"

	"github.com/Bhavik2205/ML-Bot/internal/indicators"
)

const (
	marketOpenHour    = 9
	marketOpenMinute  = 15
	marketCloseHour   = 15
	marketCloseMinute = 30
	marketTimezone    = "Asia/Kolkata"
)

type CandleData struct {
	InstrumentToken uint32
	Interval        string
	Timestamp       time.Time
	Open            float64
	High            float64
	Low             float64
	Close           float64
	Volume          float64
	TradeCount      uint32
	LastTickTime    time.Time
	mu              sync.Mutex
}

type instrumentCandles struct {
	mu      sync.Mutex
	candles map[string]*CandleData
}

type CandleGenerator struct {
	dbClient                *db.DBClient
	redisClient             *cache.RedisClient
	appCfg                  *utils.AppConfig
	openCandles             map[uint32]*instrumentCandles
	openCandlesMu           sync.Mutex
	candleWsClients         *sync.Map // maps *websocket.Conn to *wsClient
	marketLoc               *time.Location
	indicatorManagerInputCh chan<- indicators.Candle

	candleDBFlushCh chan db.OHLCVCandle

	// Metrics
	ticksProcessed uint64
	dbErrors       uint64
	wsDrops        uint64

	// Monitoring
	monitorStopCh chan struct{}
}

// NewCandleGenerator creates and returns a new instance of CandleGenerator.
func NewCandleGenerator(
	dbC *db.DBClient,
	rC *cache.RedisClient,
	cfg *utils.AppConfig,
	wsClients *sync.Map,
	indicatorManagerInputCh chan<- indicators.Candle,
) *CandleGenerator {
	loc, err := time.LoadLocation(marketTimezone)
	if err != nil {
		zap.L().Error("Failed to load market timezone, defaulting to UTC. Market time-based candle alignment may be incorrect.",
			zap.String("timezone", marketTimezone), zap.Error(err))
		loc = time.UTC
	}
	cg := &CandleGenerator{
		dbClient:                dbC,
		redisClient:             rC,
		appCfg:                  cfg,
		openCandles:             make(map[uint32]*instrumentCandles),
		candleWsClients:         wsClients,
		marketLoc:               loc,
		indicatorManagerInputCh: indicatorManagerInputCh,
		candleDBFlushCh:         make(chan db.OHLCVCandle, cfg.Ingestion.DBFlushChannelSize),
		monitorStopCh:           make(chan struct{}),
	}
	return cg
}

// StartCandleGeneration subscribes to Redis ticks and processes them into candles.
func (cg *CandleGenerator) StartCandleGeneration(ctx context.Context) {
	// Start monitoring goroutine
	go cg.startMonitoring()

	defer func() {
		if r := recover(); r != nil {
			zap.L().Error("Panic recovered in StartCandleGeneration", zap.Any("recover", r))
		}
		// Safe close: monitorStopCh may already be closed if ctx was cancelled inside the loop.
		select {
		case <-cg.monitorStopCh:
			// already closed
		default:
			close(cg.monitorStopCh)
		}
	}()

	pubsub := cg.redisClient.Subscribe(ctx, api.RedisMarketDataChannel)
	if pubsub == nil {
		zap.L().Error("Failed to subscribe to Redis PubSub for candle generation, stopping.")
		return
	}
	defer func() {
		if err := pubsub.Close(); err != nil {
			if !isClosedNetworkError(err) {
				zap.L().Error("Failed to close Redis PubSub connection for candle generator", zap.Error(err))
			}
		}
		zap.L().Info("Redis PubSub subscriber for candle generator closed.")
	}()

	zap.L().Info("✅ Candle generator subscribed to Redis market data channel", zap.String("channel", api.RedisMarketDataChannel))

	ch := pubsub.Channel()
	for {
		select {
		case msg, ok := <-ch:
			if !ok {
				zap.L().Warn("Redis PubSub channel for candle generator closed. Attempting reconnect in 5 seconds...")
				if pubsub != nil {
					_ = pubsub.Close()
					pubsub = nil
				}
				select {
				case <-ctx.Done():
					return
				case <-time.After(5 * time.Second):
				}
				pubsub = cg.redisClient.Subscribe(ctx, api.RedisMarketDataChannel)
				if pubsub == nil {
					zap.L().Error("Failed to resubscribe to Redis PubSub for candle generation, stopping.")
					return
				}
				ch = pubsub.Channel()
				zap.L().Info("Successfully reconnected to Redis PubSub for candle generation.")
				continue
			}

			var enrichedTick struct {
				Symbol           string          `json:"symbol"`
				ProcessedAtNanos int64           `json:"processed_at_nanos"`
				Tick             json.RawMessage `json:"tick"`
			}
			if err := json.Unmarshal([]byte(msg.Payload), &enrichedTick); err != nil {
				zap.L().Error("Failed to unmarshal Redis message payload outer structure for candle generation",
					zap.Error(err),
					zap.String("payload_sample", string(msg.Payload[:min(len(msg.Payload), 200)])))
				continue
			}

			var kiteTick kitemodels.Tick
			if err := json.Unmarshal(enrichedTick.Tick, &kiteTick); err != nil {
				zap.L().Error("Failed to unmarshal raw tick for candle generation",
					zap.Error(err),
					zap.String("tick_payload_sample", string(enrichedTick.Tick[:min(len(enrichedTick.Tick), 100)])))
				continue
			}

			atomic.AddUint64(&cg.ticksProcessed, 1)
			func() {
				defer cg.recoverGoroutine("processTickForCandles")
				cg.processTickForCandles(kiteTick)
			}()

		case <-ctx.Done():
			zap.L().Info("Context cancelled, stopping candle generator Redis subscriber.")
			cg.flushAllOpenCandles()
			return
		}
	}
}

// StartCandleDBWriter batches and writes candles to DB.
func (cg *CandleGenerator) StartCandleDBWriter(ctx context.Context) {
	defer cg.recoverGoroutine("StartCandleDBWriter")
	batchSize := cg.appCfg.Ingestion.MarketDataBatchSize
	batch := make([]db.OHLCVCandle, 0, batchSize)
	ticker := time.NewTicker(time.Duration(cg.appCfg.Ingestion.MarketDataFlushIntervalMS) * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case candle := <-cg.candleDBFlushCh:
			batch = append(batch, candle)
			if len(batch) >= batchSize {
				cg.writeCandleBatch(batch)
				batch = batch[:0]
			}
		case <-ticker.C:
			if len(batch) > 0 {
				cg.writeCandleBatch(batch)
				batch = batch[:0]
			}
		case <-ctx.Done():
			if len(batch) > 0 {
				cg.writeCandleBatch(batch)
			}
			return
		}
	}
}

func (cg *CandleGenerator) writeCandleBatch(batch []db.OHLCVCandle) {
	defer cg.recoverGoroutine("writeCandleBatch")
	result := cg.dbClient.DB.Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "timestamp"}},
		DoUpdates: clause.AssignmentColumns([]string{"high", "low", "close", "volume", "trade_count", "updated_at"}),
	}).CreateInBatches(batch, len(batch))
	if result.Error != nil {
		atomic.AddUint64(&cg.dbErrors, 1)
		zap.L().Error("❌ Failed to batch save/update OHLCVCandles to DB", zap.Error(result.Error))
	}
}

// RegisterWebSocketClient adds a new WebSocket client and starts its write pump.
func (cg *CandleGenerator) RegisterWebSocketClient(conn *websocket.Conn) {
	bufferSize := cg.appCfg.Ingestion.WSBroadcastChannelSize
	client := &wsClient{conn: conn, send: make(chan []byte, bufferSize)}
	cg.candleWsClients.Store(conn, client)
	go cg.writePump(client)
}

// UnregisterWebSocketClient removes a WebSocket client and closes its channel.
func (cg *CandleGenerator) UnregisterWebSocketClient(conn *websocket.Conn) {
	if val, ok := cg.candleWsClients.Load(conn); ok {
		client := val.(*wsClient)
		close(client.send)
		cg.candleWsClients.Delete(conn)
		conn.Close()
	}
}

// writePump writes messages from the channel to the WebSocket with deadlines and periodic pings.
func (cg *CandleGenerator) writePump(client *wsClient) {
	defer func() {
		if r := recover(); r != nil {
			zap.L().Error("Panic recovered in candle writePump", zap.Any("recover", r))
		}
		client.conn.Close()
		cg.candleWsClients.Delete(client.conn)
	}()
	pingTicker := time.NewTicker(wsPingPeriod)
	defer pingTicker.Stop()
	for {
		select {
		case msg, ok := <-client.send:
			if !ok {
				_ = client.conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
				_ = client.conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
				return
			}
			_ = client.conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
			if err := client.conn.WriteMessage(websocket.TextMessage, msg); err != nil {
				return
			}
		case <-pingTicker.C:
			_ = client.conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
			if err := client.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// isMarketOpen checks if the given time falls within the defined market hours.
func (cg *CandleGenerator) isMarketOpen(t time.Time) bool {
	marketTime := t.In(cg.marketLoc)
	marketOpenToday := time.Date(marketTime.Year(), marketTime.Month(), marketTime.Day(),
		marketOpenHour, marketOpenMinute, 0, 0, cg.marketLoc)
	marketCloseToday := time.Date(marketTime.Year(), marketTime.Month(), marketTime.Day(),
		marketCloseHour, marketCloseMinute, 0, 0, cg.marketLoc)
	return !marketTime.Before(marketOpenToday) && !marketTime.After(marketCloseToday)
}

// getCandleStartTime aligns a given tick time to the appropriate candle start time.
func (cg *CandleGenerator) getCandleStartTime(tickTime time.Time, intervalDuration time.Duration) time.Time {
	marketTime := tickTime.In(cg.marketLoc)
	marketOpenToday := time.Date(marketTime.Year(), marketTime.Month(), marketTime.Day(),
		marketOpenHour, marketOpenMinute, 0, 0, cg.marketLoc)
	if marketTime.Before(marketOpenToday) {
		return time.Time{}
	}
	if intervalDuration == time.Hour {
		minutesSinceMarketOpen := marketTime.Sub(marketOpenToday).Minutes()
		if minutesSinceMarketOpen < 0 {
			return time.Time{}
		}
		hourOffset := int(minutesSinceMarketOpen / 60)
		candleStartTime := marketOpenToday.Add(time.Duration(hourOffset) * time.Hour)
		return candleStartTime.In(time.UTC)
	} else if intervalDuration == 24*time.Hour {
		candleStartTime := marketOpenToday
		return candleStartTime.In(time.UTC)
	}
	truncatedLocalTime := marketTime.Truncate(intervalDuration)
	return truncatedLocalTime.In(time.UTC)
}

// processTickForCandles processes an incoming market data tick to update or create OHLCV candles.
func (cg *CandleGenerator) processTickForCandles(tick kitemodels.Tick) {
	instrumentToken := tick.InstrumentToken
	tickTime := tick.Timestamp.Time

	if !cg.isMarketOpen(tickTime) {
		zap.L().Debug("Skipping tick outside market hours",
			zap.Uint32("token", instrumentToken),
			zap.Time("tick_time", tickTime.In(cg.marketLoc)))
		return
	}

	cg.openCandlesMu.Lock()
	ic, ok := cg.openCandles[instrumentToken]
	if !ok {
		ic = &instrumentCandles{candles: make(map[string]*CandleData)}
		cg.openCandles[instrumentToken] = ic
	}
	cg.openCandlesMu.Unlock()

	ic.mu.Lock()
	defer ic.mu.Unlock()

	for _, intervalStr := range cg.appCfg.Candles.Intervals {
		intervalDuration, err := parseInterval(intervalStr)
		if err != nil {
			zap.L().Error("Invalid candle interval configured in app.yaml, skipping candle generation for this interval",
				zap.String("interval", intervalStr),
				zap.Error(err))
			continue
		}

		candleStartTime := cg.getCandleStartTime(tickTime, intervalDuration)
		if candleStartTime.IsZero() {
			zap.L().Warn("Could not determine valid candle start time for tick, skipping",
				zap.Uint32("token", instrumentToken),
				zap.String("interval", intervalStr),
				zap.Time("tick_time", tickTime.In(cg.marketLoc)))
			continue
		}

		currentCandle, candleExists := ic.candles[intervalStr]

		if !candleExists || currentCandle.Timestamp.Before(candleStartTime) {
			if candleExists && currentCandle.Timestamp.Before(candleStartTime) {
				tempCandleToFlush := CandleData{
					InstrumentToken: currentCandle.InstrumentToken,
					Interval:        currentCandle.Interval,
					Timestamp:       currentCandle.Timestamp,
					Open:            currentCandle.Open,
					High:            currentCandle.High,
					Low:             currentCandle.Low,
					Close:           currentCandle.Close,
					Volume:          currentCandle.Volume,
					TradeCount:      currentCandle.TradeCount,
					LastTickTime:    currentCandle.LastTickTime,
				}
				cg.flushCandle(&tempCandleToFlush)
				zap.L().Debug("Flushed completed candle",
					zap.Uint32("token", tempCandleToFlush.InstrumentToken),
					zap.String("interval", tempCandleToFlush.Interval),
					zap.Time("timestamp", tempCandleToFlush.Timestamp.In(cg.marketLoc)),
					zap.Float64("close", tempCandleToFlush.Close))
			}
			newCandle := &CandleData{
				InstrumentToken: instrumentToken,
				Interval:        intervalStr,
				Timestamp:       candleStartTime,
				Open:            tick.LastPrice,
				High:            tick.LastPrice,
				Low:             tick.LastPrice,
				Close:           tick.LastPrice,
				Volume:          float64(tick.LastTradedQuantity),
				TradeCount:      1,
				LastTickTime:    tickTime,
			}
			ic.candles[intervalStr] = newCandle
			zap.L().Debug("Created new candle",
				zap.Uint32("token", instrumentToken),
				zap.String("interval", intervalStr),
				zap.Time("timestamp", candleStartTime.In(cg.marketLoc)),
				zap.Float64("open", tick.LastPrice),
				zap.Time("tick_time", tickTime.In(cg.marketLoc)))
		} else {
			if tick.LastPrice > currentCandle.High {
				currentCandle.High = tick.LastPrice
			}
			if tick.LastPrice < currentCandle.Low {
				currentCandle.Low = tick.LastPrice
			}
			currentCandle.Close = tick.LastPrice
			currentCandle.Volume += float64(tick.LastTradedQuantity)
			currentCandle.TradeCount++
			currentCandle.LastTickTime = tickTime
			zap.L().Debug("Updated existing candle",
				zap.Uint32("token", instrumentToken),
				zap.String("interval", intervalStr),
				zap.Time("timestamp", currentCandle.Timestamp.In(cg.marketLoc)),
				zap.Float64("close", currentCandle.Close),
				zap.Float64("volume", currentCandle.Volume))
		}
	}
}

// flushCandle saves a completed CandleData to the database and sends it for indicator calculation.
func (cg *CandleGenerator) flushCandle(cd *CandleData) {
	if cd == nil {
		zap.L().Debug("Attempted to flush a nil candle, skipping.")
		return
	}

	cd.mu.Lock()
	defer cd.mu.Unlock()

	ohlcvCandle := db.OHLCVCandle{
		InstrumentToken: cd.InstrumentToken,
		Interval:        cd.Interval,
		Timestamp:       cd.Timestamp,
		Open:            cd.Open,
		High:            cd.High,
		Low:             cd.Low,
		Close:           cd.Close,
		Volume:          cd.Volume,
		TradeCount:      cd.TradeCount,
	}

	// Batch insert: send to channel instead of direct DB write
	select {
	case cg.candleDBFlushCh <- ohlcvCandle:
	default:
		atomic.AddUint64(&cg.dbErrors, 1)
		zap.L().Warn("Candle DB flush channel full, dropping candle",
			zap.Uint32("instrument_token", cd.InstrumentToken),
			zap.String("interval", cd.Interval),
			zap.Time("timestamp", cd.Timestamp))
	}

	cg.broadcastCandle(cd)

	if cg.indicatorManagerInputCh != nil {
		select {
		case cg.indicatorManagerInputCh <- indicators.Candle{
			InstrumentToken: cd.InstrumentToken,
			Interval:        cd.Interval,
			Timestamp:       cd.Timestamp,
			Open:            cd.Open,
			High:            cd.High,
			Low:             cd.Low,
			Close:           cd.Close,
			Volume:          cd.Volume,
			TradeCount:      cd.TradeCount,
		}:
			zap.L().Debug("Sent completed candle to IndicatorsManager",
				zap.Uint32("token", cd.InstrumentToken),
				zap.String("interval", cd.Interval),
				zap.Time("timestamp", cd.Timestamp.In(cg.marketLoc)))
		default:
			zap.L().Warn("IndicatorsManager input channel is full; dropping candle for indicator calculation.",
				zap.Uint32("token", cd.InstrumentToken),
				zap.String("interval", cd.Interval),
				zap.Time("timestamp", cd.Timestamp.In(cg.marketLoc)))
		}
	}
}

// broadcastCandle marshals and sends the candle data to all connected WebSocket clients.
func (cg *CandleGenerator) broadcastCandle(cd *CandleData) {
	broadcastData := struct {
		InstrumentToken uint32    `json:"instrument_token"`
		Interval        string    `json:"interval"`
		Timestamp       time.Time `json:"timestamp"`
		Open            float64   `json:"open"`
		High            float64   `json:"high"`
		Low             float64   `json:"low"`
		Close           float64   `json:"close"`
		Volume          float64   `json:"volume"`
		TradeCount      uint32    `json:"trade_count"`
	}{
		InstrumentToken: cd.InstrumentToken,
		Interval:        cd.Interval,
		Timestamp:       cd.Timestamp,
		Open:            cd.Open,
		High:            cd.High,
		Low:             cd.Low,
		Close:           cd.Close,
		Volume:          cd.Volume,
		TradeCount:      cd.TradeCount,
	}

	jsonMessage, err := json.Marshal(broadcastData)
	if err != nil {
		zap.L().Error("Failed to marshal candle data for WebSocket broadcast", zap.Error(err), zap.Uint32("token", cd.InstrumentToken))
		return
	}

	cg.candleWsClients.Range(func(key, value interface{}) bool {
		client, ok := value.(*wsClient)
		if !ok {
			cg.candleWsClients.Delete(key)
			return true
		}
		select {
		case client.send <- jsonMessage:
		default:
			atomic.AddUint64(&cg.wsDrops, 1)
			zap.L().Warn("WebSocket send channel full, dropping candle message")
		}
		return true
	})
	zap.L().Debug("Broadcasted candle to WebSocket clients", zap.Uint32("token", cd.InstrumentToken), zap.String("interval", cd.Interval))
}

func parseInterval(interval string) (time.Duration, error) {
	switch interval {
	case "1m":
		return time.Minute, nil
	case "5m":
		return 5 * time.Minute, nil
	case "15m":
		return 15 * time.Minute, nil
	case "1h":
		return time.Hour, nil
	case "1d":
		return 24 * time.Hour, nil
	default:
		return 0, fmt.Errorf("unsupported interval: %s", interval)
	}
}

func (cg *CandleGenerator) flushAllOpenCandles() {
	zap.L().Info("Flushing all remaining open candles during graceful shutdown...")
	cg.openCandlesMu.Lock()
	defer cg.openCandlesMu.Unlock()

	for _, ic := range cg.openCandles {
		ic.mu.Lock()
		for _, candle := range ic.candles {
			tempCandleToFlush := &CandleData{
				InstrumentToken: candle.InstrumentToken,
				Interval:        candle.Interval,
				Timestamp:       candle.Timestamp,
				Open:            candle.Open,
				High:            candle.High,
				Low:             candle.Low,
				Close:           candle.Close,
				Volume:          candle.Volume,
				TradeCount:      candle.TradeCount,
				LastTickTime:    candle.LastTickTime,
			}
			cg.flushCandle(tempCandleToFlush)
		}
		ic.mu.Unlock()
	}
	zap.L().Info("All open candles flushed successfully.")
}

// recoverGoroutine logs and recovers from panics in goroutines.
func (cg *CandleGenerator) recoverGoroutine(where string) {
	if r := recover(); r != nil {
		zap.L().Error("Panic recovered", zap.String("where", where), zap.Any("recover", r))
	}
}

// startMonitoring launches a goroutine to monitor system usage and candle generator health.
func (cg *CandleGenerator) startMonitoring() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	var lastTicksProcessed uint64
	for {
		select {
		case <-cg.monitorStopCh:
			return
		case <-ticker.C:
			// Candle generator metrics only
			ticks := atomic.LoadUint64(&cg.ticksProcessed)
			dbErrs := atomic.LoadUint64(&cg.dbErrors)
			wsDrops := atomic.LoadUint64(&cg.wsDrops)
			tps := ticks - lastTicksProcessed
			lastTicksProcessed = ticks

			zap.L().Info("CandleGenerator monitoring",
				zap.Uint64("ticks_processed", ticks),
				zap.Uint64("db_errors", dbErrs),
				zap.Uint64("ws_drops", wsDrops),
				zap.Uint64("ticks_per_5s", tps),
			)

			if tps < 10 {
				zap.L().Warn("Low tick processing speed detected", zap.Uint64("ticks_per_5s", tps))
			}
		}
	}
}

// GetWebSocketClientCount returns the number of currently connected WebSocket clients for candles.
func (cg *CandleGenerator) GetWebSocketClientCount() int {
	count := 0
	cg.candleWsClients.Range(func(key, value interface{}) bool {
		count++
		return true
	})
	return count
}
