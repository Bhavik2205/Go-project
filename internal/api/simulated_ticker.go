

package api

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/cache"
	kitemodels "github.com/zerodha/gokiteconnect/v4/models"
	"go.uber.org/zap"
)

// ─────────────────────────────────────────────
//  DATA STRUCTURES
// ─────────────────────────────────────────────

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
	LastTradedQuantity uint32           `json:"LastTradedQuantity"`
	AverageTradePrice  float64          `json:"AverageTradePrice"`
	NetChange          float64          `json:"NetChange"`
	PrevClose          float64          `json:"PrevClose"`
	PercentChange      float64          `json:"PercentChange"`
}

// volatility regime the market is currently in
type VolatilityRegime int

const (
	RegimeRanging  VolatilityRegime = iota // Low drift, low vol
	RegimeTrending                         // Sustained directional drift
	RegimeVolatile                         // High vol, mean-reverting bursts
)

// Sector groupings — stocks within a sector share a common factor
type Sector int

const (
	SectorBanking Sector = iota
	SectorIT
	SectorFMCG
	SectorPharma
	SectorAuto
	SectorEnergy
	SectorMetal
	SectorRealty
)

// per-instrument simulation state
type instrumentState struct {
	token     uint32
	symbol    string
	sector    Sector

	// price model
	lastPrice   float64
	prevClose   float64
	openPrice   float64
	highPrice   float64
	lowPrice    float64
	vwapNum     float64 // running numerator for VWAP
	vwapDen     float64 // running denominator for VWAP

	// GBM parameters (per instrument, sector-tuned)
	annualVol   float64 // annualised volatility (e.g. 0.25 = 25%)
	beta        float64 // sensitivity to the index factor (0.5–1.5)
	drift       float64 // intraday directional drift (reset each regime change)

	// volume model
	totalVolume uint32
	avgTickVol  uint32 // baseline lot size for this stock

	// circuit breaker
	upperCircuit float64
	lowerCircuit float64
	circuitBand  float64 // fraction, e.g. 0.10 = ±10%

	// regime
	regime          VolatilityRegime
	regimeTicksLeft int // ticks remaining in current regime

	// opening auction
	auctionDone bool
}

// ─────────────────────────────────────────────
//  REALISTIC NSE SYMBOL SEED DATA
//  (50 liquid NSE stocks with realistic price ranges)
// ─────────────────────────────────────────────

type symbolSeed struct {
	symbol    string
	sector    Sector
	midPrice  float64 // realistic mid price in INR
	annualVol float64 // realistic annualised vol
	beta      float64 // market beta
	lotSize   uint32  // typical tick trade lot
	circuit   float64 // circuit band fraction
}

var nse50Seeds = []symbolSeed{
	// Banking
	{"HDFCBANK", SectorBanking, 1620.0, 0.22, 1.1, 200, 0.10},
	{"ICICIBANK", SectorBanking, 1120.0, 0.25, 1.2, 300, 0.10},
	{"KOTAKBANK", SectorBanking, 1780.0, 0.20, 1.0, 150, 0.10},
	{"AXISBANK", SectorBanking, 1080.0, 0.28, 1.3, 300, 0.10},
	{"SBIN", SectorBanking, 780.0, 0.30, 1.4, 500, 0.10},
	{"BANKBARODA", SectorBanking, 245.0, 0.38, 1.5, 1500, 0.20},
	{"INDUSINDBK", SectorBanking, 960.0, 0.35, 1.3, 400, 0.10},
	// IT
	{"TCS", SectorIT, 3850.0, 0.18, 0.7, 75, 0.10},
	{"INFY", SectorIT, 1450.0, 0.20, 0.75, 200, 0.10},
	{"WIPRO", SectorIT, 480.0, 0.22, 0.80, 600, 0.10},
	{"HCLTECH", SectorIT, 1380.0, 0.22, 0.78, 200, 0.10},
	{"TECHM", SectorIT, 1560.0, 0.25, 0.85, 200, 0.10},
	{"LTIM", SectorIT, 5200.0, 0.23, 0.80, 50, 0.10},
	// FMCG
	{"HINDUNILVR", SectorFMCG, 2450.0, 0.15, 0.5, 100, 0.10},
	{"ITC", SectorFMCG, 445.0, 0.18, 0.6, 1000, 0.10},
	{"NESTLEIND", SectorFMCG, 2350.0, 0.14, 0.45, 100, 0.10},
	{"BRITANNIA", SectorFMCG, 5100.0, 0.16, 0.5, 50, 0.10},
	{"DABUR", SectorFMCG, 535.0, 0.17, 0.55, 800, 0.10},
	// Pharma
	{"SUNPHARMA", SectorPharma, 1620.0, 0.24, 0.65, 200, 0.10},
	{"DRREDDY", SectorPharma, 6200.0, 0.22, 0.60, 25, 0.10},
	{"CIPLA", SectorPharma, 1480.0, 0.23, 0.65, 200, 0.10},
	{"DIVISLAB", SectorPharma, 3800.0, 0.25, 0.60, 75, 0.10},
	{"AUROPHARMA", SectorPharma, 1150.0, 0.30, 0.70, 300, 0.20},
	// Auto
	{"MARUTI", SectorAuto, 11500.0, 0.20, 0.85, 20, 0.10},
	{"TATAMOTORS", SectorAuto, 920.0, 0.35, 1.3, 400, 0.10},
	{"BAJAJ-AUTO", SectorAuto, 8900.0, 0.18, 0.80, 25, 0.10},
	{"HEROMOTOCO", SectorAuto, 4700.0, 0.19, 0.75, 50, 0.10},
	{"EICHERMOT", SectorAuto, 4600.0, 0.22, 0.85, 50, 0.10},
	{"TVSMOTOR", SectorAuto, 2200.0, 0.28, 0.95, 100, 0.10},
	// Energy
	{"RELIANCE", SectorEnergy, 2940.0, 0.22, 1.0, 100, 0.10},
	{"ONGC", SectorEnergy, 265.0, 0.30, 1.1, 1500, 0.10},
	{"BPCL", SectorEnergy, 315.0, 0.32, 1.15, 1500, 0.10},
	{"IOC", SectorEnergy, 175.0, 0.33, 1.1, 3000, 0.20},
	{"GAIL", SectorEnergy, 215.0, 0.28, 0.90, 2000, 0.10},
	{"POWERGRID", SectorEnergy, 330.0, 0.20, 0.70, 1500, 0.10},
	{"NTPC", SectorEnergy, 375.0, 0.22, 0.75, 1500, 0.10},
	// Metal
	{"TATASTEEL", SectorMetal, 155.0, 0.38, 1.4, 3000, 0.20},
	{"HINDALCO", SectorMetal, 665.0, 0.35, 1.35, 800, 0.20},
	{"JSWSTEEL", SectorMetal, 890.0, 0.36, 1.4, 600, 0.20},
	{"COALINDIA", SectorMetal, 435.0, 0.28, 1.0, 1500, 0.10},
	{"VEDL", SectorMetal, 455.0, 0.40, 1.45, 1000, 0.20},
	// Realty / Infra / Mixed
	{"LT", SectorRealty, 3600.0, 0.23, 1.1, 75, 0.10},
	{"ADANIENT", SectorRealty, 2450.0, 0.45, 1.5, 100, 0.20},
	{"ADANIPORTS", SectorRealty, 1280.0, 0.35, 1.2, 200, 0.10},
	{"ULTRACEMCO", SectorRealty, 10500.0, 0.20, 0.85, 25, 0.10},
	{"SHREECEM", SectorRealty, 27000.0, 0.18, 0.75, 10, 0.10},
	{"DLF", SectorRealty, 860.0, 0.38, 1.3, 600, 0.20},
	{"BAJFINANCE", SectorBanking, 7100.0, 0.30, 1.2, 40, 0.10},
	{"BAJAJFINSV", SectorBanking, 1680.0, 0.28, 1.1, 200, 0.10},
	{"ASIANPAINT", SectorFMCG, 2800.0, 0.17, 0.6, 100, 0.10},
	{"M&M", SectorAuto, 2900.0, 0.26, 1.05, 100, 0.10},
}

// ─────────────────────────────────────────────
//  SIMULATOR
// ─────────────────────────────────────────────

type SimulatedZerodhaClient struct{}

func NewSimulatedZerodhaClient() *SimulatedZerodhaClient {
	return &SimulatedZerodhaClient{}
}

// r2 rounds to 2 decimal places
func r2(v float64) float64 { return math.Round(v*100) / 100 }

// r4 rounds to 4 decimal places
func r4(v float64) float64 { return math.Round(v*10000) / 10000 }

// clamp keeps v within [lo, hi]
func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// sampleRegime returns a regime weighted by time of day.
// Open/close periods are more volatile; midday is ranging.
func sampleRegime(minutesElapsed float64) VolatilityRegime {
	// 0–30 min → volatile open; 30–330 → mix; 330–375 → volatile close
	r := rand.Float64()
	switch {
	case minutesElapsed < 30:
		if r < 0.55 {
			return RegimeVolatile
		} else if r < 0.80 {
			return RegimeTrending
		}
		return RegimeRanging
	case minutesElapsed > 330:
		if r < 0.50 {
			return RegimeVolatile
		} else if r < 0.80 {
			return RegimeTrending
		}
		return RegimeRanging
	default: // midday
		if r < 0.50 {
			return RegimeRanging
		} else if r < 0.80 {
			return RegimeTrending
		}
		return RegimeVolatile
	}
}

// volumeMultiplier returns a scalar ∈ [0.3, 3.0] that mimics the
// U-shaped intraday volume profile (high at open & close, low at noon).
func volumeMultiplier(minutesElapsed float64) float64 {
	// Normalised position in the trading day [0,1]
	t := minutesElapsed / 375.0
	// U-shape: f(t) = 1 + 2*exp(-8*(t-0)^2) + 2*exp(-8*(t-1)^2) - 0.5
	openBump := 2.0 * math.Exp(-18*t*t)
	closeBump := 2.0 * math.Exp(-18*(t-1)*(t-1))
	mid := -0.3
	v := 1.0 + openBump + closeBump + mid
	return clamp(v, 0.3, 4.0)
}

// normalSample returns a standard normal sample via Box-Muller.
func normalSample() float64 {
	u1 := rand.Float64()
	u2 := rand.Float64()
	if u1 < 1e-10 {
		u1 = 1e-10
	}
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

// initState builds an instrumentState from a seed and an InstrumentInfo.
// If the provided InstrumentInfo token doesn't match any seed by index,
// we cycle through the seed list.
func initState(info *InstrumentInfo, seed symbolSeed) *instrumentState {
	// Seed a realistic prevClose with a tiny random perturbation (±0.3%)
	prevClose := r2(seed.midPrice * (1 + (rand.Float64()-0.5)*0.006))
	if prevClose <= 0 {
		prevClose = seed.midPrice
	}

	// Opening gap: pre-market sentiment moves price ±0–2% from prevClose
	// Probability of gap-up vs gap-down is slightly skewed to up (55%)
	gapDir := 1.0
	if rand.Float64() < 0.45 {
		gapDir = -1.0
	}
	gapMag := rand.Float64() * 0.02 // up to 2%
	openPrice := r2(prevClose * (1 + gapDir*gapMag))

	// Circuit band
	upper := r2(prevClose * (1 + seed.circuit))
	lower := r2(prevClose * (1 - seed.circuit))

	return &instrumentState{
		token:           info.Token,
		symbol:          seed.symbol,
		sector:          seed.sector,
		lastPrice:       openPrice,
		prevClose:       prevClose,
		openPrice:       openPrice,
		highPrice:       openPrice,
		lowPrice:        openPrice,
		vwapNum:         openPrice * float64(seed.lotSize),
		vwapDen:         float64(seed.lotSize),
		annualVol:       seed.annualVol,
		beta:            seed.beta,
		drift:           0,
		totalVolume:     seed.lotSize,
		avgTickVol:      seed.lotSize,
		upperCircuit:    upper,
		lowerCircuit:    lower,
		circuitBand:     seed.circuit,
		regime:          RegimeRanging,
		regimeTicksLeft: 0,
		auctionDone:     false,
	}
}

// ─────────────────────────────────────────────
//  MAIN SIMULATION ENTRY POINT
// ─────────────────────────────────────────────

func (s *SimulatedZerodhaClient) SimulateTicks(
	ctx context.Context,
	infos []*InstrumentInfo,
	redisClient *cache.RedisClient,
	simulationSpeedMultiplier float64,
) error {
	if redisClient == nil {
		return fmt.Errorf("RedisClient is nil")
	}
	if len(infos) == 0 {
		return fmt.Errorf("no instruments provided")
	}

	// ── Initialise per-instrument state ──────────────────────────────────
	states := make([]*instrumentState, len(infos))
	for i, info := range infos {
		seed := nse50Seeds[i%len(nse50Seeds)]
		states[i] = initState(info, seed)
	}

	// ── Sector factor map: one shared GBM factor per sector ──────────────
	// Will be updated each tick.
	sectorFactors := make(map[Sector]float64)
	for sec := SectorBanking; sec <= SectorRealty; sec++ {
		sectorFactors[sec] = 0.0
	}

	// ── Index (Nifty-like) factor ─────────────────────────────────────────
	indexFactor := 0.0

	// ── Timing ───────────────────────────────────────────────────────────
	// Tick interval: 500 ms simulated time
	tickIntervalSim := 500 * time.Millisecond
	realDelay := time.Duration(float64(tickIntervalSim) / simulationSpeedMultiplier)
	if realDelay < time.Millisecond {
		realDelay = time.Millisecond
	}

	// dt in years for one tick (used in GBM)
	tradingSecondsPerYear := 375.0 * 60.0 * 252.0 // NSE: 375 min/day, 252 days
	dtSeconds := tickIntervalSim.Seconds()
	dt := dtSeconds / tradingSecondsPerYear

	// Total simulated duration: 9:15 → 15:30 = 375 minutes
	totalSimDuration := 375 * time.Minute
	simulationStart := time.Now()

	// Track regime per sector
	sectorRegimes := make(map[Sector]VolatilityRegime)
	sectorRegimeTicks := make(map[Sector]int)
	for sec := SectorBanking; sec <= SectorRealty; sec++ {
		sectorRegimes[sec] = RegimeRanging
		sectorRegimeTicks[sec] = 0
	}

	zap.L().Info("🚀 NSE Market Simulator started",
		zap.Int("instruments", len(states)),
		zap.Float64("speed_multiplier", simulationSpeedMultiplier),
		zap.String("real_delay_per_tick", realDelay.String()),
	)

	// ─────────────────────────────────────────────────────────────────────
	//  MAIN LOOP
	// ─────────────────────────────────────────────────────────────────────
	for {
		select {
		case <-ctx.Done():
			zap.L().Info("🚫 Simulation stopped.")
			return nil
		default:
		}

		elapsedReal := time.Since(simulationStart)
		elapsedSim := time.Duration(float64(elapsedReal) * simulationSpeedMultiplier)

		if elapsedSim >= totalSimDuration {
			zap.L().Info("✅ Simulated trading day complete.")
			return nil
		}

		minutesElapsed := elapsedSim.Minutes()
		volMult := volumeMultiplier(minutesElapsed)

		// ── Update index factor (shared shock all stocks feel) ────────────
		// Index has its own GBM: ~15% annualised vol
		indexVol := 0.15
		indexDrift := 0.0
		indexFactor = indexVol*math.Sqrt(dt)*normalSample() + indexDrift*dt

		// ── Update sector factors (sector shock on top of index) ──────────
		for sec := SectorBanking; sec <= SectorRealty; sec++ {
			// Advance regime timer
			sectorRegimeTicks[sec]--
			if sectorRegimeTicks[sec] <= 0 {
				sectorRegimes[sec] = sampleRegime(minutesElapsed)
				// Regime lasts 30–300 ticks (15s–2.5min sim time at 500ms/tick)
				sectorRegimeTicks[sec] = rand.Intn(270) + 30
			}

			regime := sectorRegimes[sec]
			secVol := 0.0
			secDrift := 0.0
			switch regime {
			case RegimeRanging:
				secVol = 0.10
				secDrift = 0.0
			case RegimeTrending:
				secVol = 0.15
				dir := 1.0
				if rand.Float64() < 0.5 {
					dir = -1.0
				}
				secDrift = dir * 0.20 // ~20% annualised trend
			case RegimeVolatile:
				secVol = 0.35
				secDrift = 0.0
			}
			sectorFactors[sec] = secVol*math.Sqrt(dt)*normalSample() + secDrift*dt
		}

		// ── Generate tick for each instrument ─────────────────────────────
		for _, st := range states {

			// ── Opening auction (first tick only) ─────────────────────────
			// During the first 5 minutes of sim, restrict tick volume to small lots
			// and don't update depth symmetrically (pre-open is one-sided)
			isPreOpen := minutesElapsed < 5.0
			if isPreOpen && !st.auctionDone {
				// Just publish the opening price tick without regime moves
				publishTick(st, redisClient, volMult, true)
				continue
			}
			if !st.auctionDone {
				st.auctionDone = true
			}

			// ── Per-instrument regime ─────────────────────────────────────
			st.regimeTicksLeft--
			if st.regimeTicksLeft <= 0 {
				st.regime = sampleRegime(minutesElapsed)
				st.regimeTicksLeft = rand.Intn(200) + 20
				// New drift on regime change
				switch st.regime {
				case RegimeTrending:
					dir := 1.0
					if rand.Float64() < 0.5 {
						dir = -1.0
					}
					st.drift = dir * 0.25 * st.annualVol
				default:
					st.drift = 0
				}
			}

			// ── GBM price update ──────────────────────────────────────────
			// dS = S*(mu*dt + sigma*sqrt(dt)*Z_idio + beta*Z_index + beta_sec*Z_sector)
			idioVol := st.annualVol
			switch st.regime {
			case RegimeVolatile:
				idioVol = st.annualVol * 1.8
			case RegimeTrending:
				idioVol = st.annualVol * 0.9
			}

			idioShock := idioVol * math.Sqrt(dt) * normalSample()
			indexShock := st.beta * indexFactor
			sectorShock := 0.6 * sectorFactors[st.sector] // partial sector exposure
			driftTerm := st.drift * dt

			logReturn := driftTerm + idioShock + indexShock + sectorShock
			newPrice := st.lastPrice * math.Exp(logReturn)

			// ── Circuit breaker ───────────────────────────────────────────
			if newPrice >= st.upperCircuit {
				newPrice = st.upperCircuit
				zap.L().Warn("⚡ Upper circuit hit",
					zap.String("symbol", st.symbol),
					zap.Float64("price", newPrice),
				)
			} else if newPrice <= st.lowerCircuit {
				newPrice = st.lowerCircuit
				zap.L().Warn("⚡ Lower circuit hit",
					zap.String("symbol", st.symbol),
					zap.Float64("price", newPrice),
				)
			}

			// Prevent degenerate price
			if newPrice < 0.05 {
				newPrice = 0.05
			}
			newPrice = r2(newPrice)
			st.lastPrice = newPrice

			// ── OHLC update ───────────────────────────────────────────────
			if newPrice > st.highPrice {
				st.highPrice = newPrice
			}
			if newPrice < st.lowPrice {
				st.lowPrice = newPrice
			}

			// ── Volume (U-shaped profile + regime burst) ──────────────────
			baseTickVol := float64(st.avgTickVol)
			switch st.regime {
			case RegimeVolatile:
				baseTickVol *= 2.5
			case RegimeTrending:
				baseTickVol *= 1.4
			}
			tickVol := uint32(baseTickVol * volMult * (0.5 + rand.Float64()))
			if tickVol < 1 {
				tickVol = 1
			}
			st.totalVolume += tickVol

			// ── VWAP update ───────────────────────────────────────────────
			st.vwapNum += newPrice * float64(tickVol)
			st.vwapDen += float64(tickVol)

			publishTick(st, redisClient, volMult, false)
		}

		time.Sleep(realDelay)
	}
}

// ─────────────────────────────────────────────
//  TICK PUBLISHER
// ─────────────────────────────────────────────

func publishTick(st *instrumentState, redisClient *cache.RedisClient, volMult float64, isAuction bool) {
	price := st.lastPrice
	prevClose := st.prevClose

	// ── Bid-Ask Depth ─────────────────────────────────────────────────────
	// Spread widens with volatility (volMult) and is proportional to price level.
	// Use 0.05% base spread, wider during volatile periods.
	baseSpreadFraction := 0.0005 // 0.05% of price
	spreadFraction := baseSpreadFraction * (1 + (volMult-1)*0.4)
	if isAuction {
		spreadFraction *= 3 // pre-open: very wide spread
	}
	spread := r2(price * spreadFraction)
	if spread < 0.05 {
		spread = 0.05 // min tick size
	}

	depth := kitemodels.Depth{}
	var totalBuyQty, totalSellQty uint32

	for i := 0; i < 5; i++ {
		// Price levels step away from mid at increasing multiples of spread
		levelOffset := r2(float64(i+1) * spread)

		// Quantity: larger at inner levels, tapers off
		qtyBase := uint32(float64(st.avgTickVol) * (1.5 - float64(i)*0.2) * volMult * (0.7 + rand.Float64()*0.6))
		if qtyBase < 1 {
			qtyBase = 1
		}
		ordersBase := uint32(rand.Intn(12) + 1)

		buyPrice := r2(price - levelOffset)
		if buyPrice < 0.05 {
			buyPrice = 0.05
		}

		depth.Buy[i] = kitemodels.DepthItem{
			Price:    buyPrice,
			Quantity: qtyBase,
			Orders:   ordersBase,
		}
		depth.Sell[i] = kitemodels.DepthItem{
			Price:    r2(price + levelOffset),
			Quantity: uint32(float64(qtyBase) * (0.8 + rand.Float64()*0.4)), // slight asymmetry
			Orders:   uint32(rand.Intn(12) + 1),
		}
		totalBuyQty += depth.Buy[i].Quantity
		totalSellQty += depth.Sell[i].Quantity
	}

	// ── VWAP as AverageTradePrice ─────────────────────────────────────────
	atp := price
	if st.vwapDen > 0 {
		atp = r2(st.vwapNum / st.vwapDen)
	}

	// ── Net change & percent change ───────────────────────────────────────
	netChange := r2(price - prevClose)
	pctChange := 0.0
	if prevClose != 0 {
		pctChange = r4((netChange / prevClose) * 100.0)
	}

	// ── Last traded quantity (random lot, larger during volatile) ─────────
	ltq := uint32(float64(st.avgTickVol) * (0.5 + rand.Float64()) * volMult * 0.3)
	if ltq < 1 {
		ltq = 1
	}

	tick := SimTick{
		InstrumentToken: st.token,
		Timestamp:       time.Now().Format(time.RFC3339Nano),
		LastPrice:       price,
		OHLC: kitemodels.OHLC{
			Open:  st.openPrice,
			High:  st.highPrice,
			Low:   st.lowPrice,
			Close: prevClose, // Kite convention: OHLC.Close = previous day's close
		},
		VolumeTraded:       st.totalVolume,
		Volume:             st.totalVolume,
		Depth:              depth,
		TotalBuyQuantity:   totalBuyQty,
		TotalSellQuantity:  totalSellQty,
		LastTradedQuantity: ltq,
		AverageTradePrice:  atp,
		NetChange:          netChange,
		PrevClose:          prevClose,
		PercentChange:      pctChange,
	}

	label := fmt.Sprintf("%s (NSE)", st.symbol)
	enriched := struct {
		Symbol           string      `json:"symbol"`
		ProcessedAtNanos int64       `json:"processed_at_nanos"`
		Tick             interface{} `json:"tick"`
	}{
		Symbol:           label,
		ProcessedAtNanos: time.Now().UnixNano(),
		Tick:             tick,
	}

	if data, err := json.Marshal(enriched); err == nil {
		if pubErr := redisClient.Publish(RedisMarketDataChannel, data); pubErr != nil {
			zap.L().Error("❌ Redis publish failed",
				zap.String("symbol", st.symbol),
				zap.Error(pubErr),
			)
		}
	} else {
		zap.L().Error("❌ Marshal failed",
			zap.String("symbol", st.symbol),
			zap.Error(err),
		)
	}
}
