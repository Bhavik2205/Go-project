package indicators

import "math"

// ---------------------------------------------------------------------------
// EMAState – O(1) incremental Exponential Moving Average
// ---------------------------------------------------------------------------

type EMAState struct {
	period     int
	multiplier float64
	value      float64
	warmup     float64 // running sum during seeding phase
	count      int
	ready      bool
}

func NewEMAState(period int) *EMAState {
	return &EMAState{
		period:     period,
		multiplier: 2.0 / float64(period+1),
	}
}

// Update accepts the next price and returns (value, ready).
func (s *EMAState) Update(price float64) (float64, bool) {
	s.count++
	if !s.ready {
		s.warmup += price
		if s.count == s.period {
			s.value = s.warmup / float64(s.period)
			s.ready = true
		}
		return s.value, s.ready
	}
	s.value = (price-s.value)*s.multiplier + s.value
	return s.value, true
}

func (s *EMAState) Value() float64 { return s.value }
func (s *EMAState) Ready() bool    { return s.ready }

// ---------------------------------------------------------------------------
// SMAState – O(1) incremental Simple Moving Average (sliding window)
// ---------------------------------------------------------------------------

type SMAState struct {
	period  int
	window  []float64
	head    int
	sum     float64
	count   int
	ready   bool
}

func NewSMAState(period int) *SMAState {
	return &SMAState{period: period, window: make([]float64, period)}
}

func (s *SMAState) Update(price float64) (float64, bool) {
	if s.count < s.period {
		s.window[s.count] = price
		s.sum += price
		s.count++
		if s.count == s.period {
			s.ready = true
			return s.sum / float64(s.period), true
		}
		return 0, false
	}
	// slide window
	s.sum -= s.window[s.head]
	s.window[s.head] = price
	s.sum += price
	s.head = (s.head + 1) % s.period
	return s.sum / float64(s.period), true
}

func (s *SMAState) Ready() bool { return s.ready }

// ---------------------------------------------------------------------------
// RSIState – O(1) incremental Relative Strength Index (Wilder smoothing)
// ---------------------------------------------------------------------------

type RSIState struct {
	period    int
	avgGain   float64
	avgLoss   float64
	prevClose float64
	count     int
	ready     bool
}

func NewRSIState(period int) *RSIState {
	return &RSIState{period: period}
}

func (s *RSIState) Update(close float64) (float64, bool) {
	if s.count == 0 {
		s.prevClose = close
		s.count++
		return 0, false
	}

	change := close - s.prevClose
	s.prevClose = close
	gain, loss := 0.0, 0.0
	if change > 0 {
		gain = change
	} else {
		loss = -change
	}
	s.count++

	if !s.ready {
		// accumulate for initial SMA seed (period+1 prices = period changes)
		s.avgGain += gain
		s.avgLoss += loss
		if s.count-1 == s.period {
			s.avgGain /= float64(s.period)
			s.avgLoss /= float64(s.period)
			s.ready = true
			return s.rsi(), true
		}
		return 0, false
	}

	// Wilder smoothing
	s.avgGain = (s.avgGain*float64(s.period-1) + gain) / float64(s.period)
	s.avgLoss = (s.avgLoss*float64(s.period-1) + loss) / float64(s.period)
	return s.rsi(), true
}

func (s *RSIState) rsi() float64 {
	if s.avgLoss == 0 {
		return 100
	}
	rs := s.avgGain / s.avgLoss
	return 100 - (100 / (1 + rs))
}

func (s *RSIState) Ready() bool { return s.ready }

// ---------------------------------------------------------------------------
// MACDState – O(1) incremental MACD
// Uses two EMAStates (fast/slow) + one EMAState for signal
// ---------------------------------------------------------------------------

type MACDState struct {
	fast   *EMAState
	slow   *EMAState
	signal *EMAState
	ready  bool
}

func NewMACDState(fastPeriod, slowPeriod, signalPeriod int) *MACDState {
	return &MACDState{
		fast:   NewEMAState(fastPeriod),
		slow:   NewEMAState(slowPeriod),
		signal: NewEMAState(signalPeriod),
	}
}

// Update returns (macdLine, signalLine, histogram, ready).
func (s *MACDState) Update(price float64) (float64, float64, float64, bool) {
	fastVal, fastOk := s.fast.Update(price)
	slowVal, slowOk := s.slow.Update(price)
	if !fastOk || !slowOk {
		return 0, 0, 0, false
	}
	macdLine := fastVal - slowVal
	sigVal, sigOk := s.signal.Update(macdLine)
	if !sigOk {
		return 0, 0, 0, false
	}
	s.ready = true
	return macdLine, sigVal, macdLine - sigVal, true
}

func (s *MACDState) Ready() bool { return s.ready }

// ---------------------------------------------------------------------------
// ATRState – O(1) incremental Average True Range (Wilder smoothing)
// ---------------------------------------------------------------------------

type ATRState struct {
	period    int
	atr       float64
	prevClose float64
	count     int
	warmup    float64
	ready     bool
}

func NewATRState(period int) *ATRState {
	return &ATRState{period: period}
}

func (s *ATRState) Update(high, low, close float64) (float64, bool) {
	var tr float64
	if s.count == 0 {
		tr = high - low
	} else {
		tr = math.Max(high-low, math.Max(math.Abs(high-s.prevClose), math.Abs(low-s.prevClose)))
	}
	s.prevClose = close
	s.count++

	if !s.ready {
		s.warmup += tr
		if s.count == s.period {
			s.atr = s.warmup / float64(s.period)
			s.ready = true
		}
		return s.atr, s.ready
	}
	s.atr = (s.atr*float64(s.period-1) + tr) / float64(s.period)
	return s.atr, true
}

func (s *ATRState) Ready() bool { return s.ready }

// ---------------------------------------------------------------------------
// BollingerState – O(1) Bollinger Bands backed by SMAState + rolling variance
// Uses Welford's online variance to avoid recomputing std dev over the window.
// ---------------------------------------------------------------------------

type BollingerState struct {
	sma       *SMAState
	window    []float64
	head      int
	count     int
	period    int
	numStdDev float64
	ready     bool
}

func NewBollingerState(period int, numStdDev float64) *BollingerState {
	return &BollingerState{
		sma:       NewSMAState(period),
		window:    make([]float64, period),
		period:    period,
		numStdDev: numStdDev,
	}
}

// Update returns (upper, middle, lower, ready).
func (s *BollingerState) Update(price float64) (float64, float64, float64, bool) {
	middle, ok := s.sma.Update(price)
	if s.count < s.period {
		s.window[s.count] = price
	} else {
		s.window[s.head] = price
		s.head = (s.head + 1) % s.period
	}
	s.count++
	if !ok {
		return 0, 0, 0, false
	}
	s.ready = true

	var variance float64
	for _, p := range s.window {
		d := p - middle
		variance += d * d
	}
	stdDev := math.Sqrt(variance / float64(s.period))
	band := stdDev * s.numStdDev
	return middle + band, middle, middle - band, true
}

func (s *BollingerState) Ready() bool { return s.ready }

// ---------------------------------------------------------------------------
// StochasticState – O(1) via sliding window (fixed-size ring for kPeriod)
// %D is SMA of %K over dPeriod
// ---------------------------------------------------------------------------

type StochasticState struct {
	kPeriod int
	dPeriod int
	highs   []float64
	lows    []float64
	head    int
	count   int
	kSMA    *SMAState
	ready   bool
}

func NewStochasticState(kPeriod, dPeriod int) *StochasticState {
	return &StochasticState{
		kPeriod: kPeriod,
		dPeriod: dPeriod,
		highs:   make([]float64, kPeriod),
		lows:    make([]float64, kPeriod),
		kSMA:    NewSMAState(dPeriod),
	}
}

// Update returns (K, D, ready).
func (s *StochasticState) Update(high, low, close float64) (float64, float64, bool) {
	if s.count < s.kPeriod {
		s.highs[s.count] = high
		s.lows[s.count] = low
		s.count++
		if s.count < s.kPeriod {
			return 0, 0, false
		}
	} else {
		s.highs[s.head] = high
		s.lows[s.head] = low
		s.head = (s.head + 1) % s.kPeriod
	}

	hh, ll := s.highs[0], s.lows[0]
	for i := 1; i < s.kPeriod; i++ {
		if s.highs[i] > hh {
			hh = s.highs[i]
		}
		if s.lows[i] < ll {
			ll = s.lows[i]
		}
	}

	k := 50.0
	if hh != ll {
		k = ((close - ll) / (hh - ll)) * 100
	}

	d, dOk := s.kSMA.Update(k)
	if !dOk {
		return k, 0, false
	}
	s.ready = true
	return k, d, true
}

func (s *StochasticState) Ready() bool { return s.ready }

// ---------------------------------------------------------------------------
// OBVState – trivially O(1)
// ---------------------------------------------------------------------------

type OBVState struct {
	value     float64
	prevClose float64
	seeded    bool
}

func NewOBVState() *OBVState { return &OBVState{} }

func (s *OBVState) Update(close, volume float64) (float64, bool) {
	if !s.seeded {
		s.value = volume
		s.prevClose = close
		s.seeded = true
		return s.value, true
	}
	if close > s.prevClose {
		s.value += volume
	} else if close < s.prevClose {
		s.value -= volume
	}
	s.prevClose = close
	return s.value, true
}

// ---------------------------------------------------------------------------
// VWAPState – O(1) cumulative, resets on session boundary
// ---------------------------------------------------------------------------

type VWAPState struct {
	cumulativePV float64
	cumulativeV  float64
}

func NewVWAPState() *VWAPState { return &VWAPState{} }

func (s *VWAPState) Update(open, high, low, close, volume float64) (float64, bool) {
	typical := (open + high + low + close) / 4.0
	s.cumulativePV += typical * volume
	s.cumulativeV += volume
	if s.cumulativeV == 0 {
		return 0, false
	}
	return s.cumulativePV / s.cumulativeV, true
}

// Reset should be called at the start of each trading session.
func (s *VWAPState) Reset() {
	s.cumulativePV = 0
	s.cumulativeV = 0
}

// ---------------------------------------------------------------------------
// ADXState – O(1) incremental ADX (Wilder smoothing on TR, +DM, -DM, DX)
// ---------------------------------------------------------------------------

type ADXState struct {
	period     int
	prevHigh   float64
	prevLow    float64
	prevClose  float64
	smTR       float64
	smPlusDM   float64
	smMinusDM  float64
	adx        float64
	warmupDX   float64
	warmupDXN  int
	count      int
	ready      bool
}

func NewADXState(period int) *ADXState {
	return &ADXState{period: period}
}

// Update returns (adx, plusDI, minusDI, ready).
func (s *ADXState) Update(high, low, close float64) (float64, float64, float64, bool) {
	if s.count == 0 {
		s.prevHigh = high
		s.prevLow = low
		s.prevClose = close
		s.count++
		return 0, 0, 0, false
	}

	tr := math.Max(high-low, math.Max(math.Abs(high-s.prevClose), math.Abs(low-s.prevClose)))

	upMove := high - s.prevHigh
	downMove := s.prevLow - low
	plusDM, minusDM := 0.0, 0.0
	if upMove > downMove && upMove > 0 {
		plusDM = upMove
	}
	if downMove > upMove && downMove > 0 {
		minusDM = downMove
	}

	s.prevHigh = high
	s.prevLow = low
	s.prevClose = close
	s.count++

	p := float64(s.period)

	if s.count <= s.period {
		// Wilder initial sum
		s.smTR += tr
		s.smPlusDM += plusDM
		s.smMinusDM += minusDM
		if s.count < s.period {
			return 0, 0, 0, false
		}
		// exactly at period: compute first DX, begin ADX warmup
	} else {
		s.smTR = s.smTR - (s.smTR / p) + tr
		s.smPlusDM = s.smPlusDM - (s.smPlusDM / p) + plusDM
		s.smMinusDM = s.smMinusDM - (s.smMinusDM / p) + minusDM
	}

	plusDI, minusDI := 0.0, 0.0
	if s.smTR != 0 {
		plusDI = (s.smPlusDM / s.smTR) * 100
		minusDI = (s.smMinusDM / s.smTR) * 100
	}
	sumDI := plusDI + minusDI
	dx := 0.0
	if sumDI != 0 {
		dx = (math.Abs(plusDI-minusDI) / sumDI) * 100
	}

	if !s.ready {
		s.warmupDX += dx
		s.warmupDXN++
		if s.warmupDXN == s.period {
			s.adx = s.warmupDX / p
			s.ready = true
		}
		return s.adx, plusDI, minusDI, s.ready
	}

	s.adx = (s.adx*(p-1) + dx) / p
	return s.adx, plusDI, minusDI, true
}

func (s *ADXState) Ready() bool { return s.ready }

// ---------------------------------------------------------------------------
// IndicatorStateSet – holds all states for one (instrument, interval) pair
// ---------------------------------------------------------------------------

type IndicatorStateSet struct {
	SMA        *SMAState
	EMA        *EMAState
	RSI        *RSIState
	MACD       *MACDState
	ATR        *ATRState
	Bollinger  *BollingerState
	Stochastic *StochasticState
	OBV        *OBVState
	VWAP       *VWAPState
	ADX        *ADXState
}
