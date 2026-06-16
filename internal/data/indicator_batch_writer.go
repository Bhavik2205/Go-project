package data

import (
	"context"
	"sync"
	"time"

	dbmodels "github.com/Bhavik2205/ML-Bot/internal/db"
	"go.uber.org/zap"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

// IndicatorBatchWriter accumulates indicator rows per table and flushes them
// to Postgres in bulk. Each table owns a typed slice so that GORM's
// CreateInBatches receives a concrete type — []db.IndicatorSMA, not
// []interface{} — which is required for reflection-based schema resolution.
type IndicatorBatchWriter struct {
	db        *gorm.DB
	flushSize int
	ticker    *time.Ticker

	sma        typedBuf[dbmodels.IndicatorSMA]
	ema        typedBuf[dbmodels.IndicatorEMA]
	rsi        typedBuf[dbmodels.IndicatorRSI]
	macd       typedBuf[dbmodels.IndicatorMACD]
	atr        typedBuf[dbmodels.IndicatorATR]
	bollinger  typedBuf[dbmodels.IndicatorBollingerBands]
	stochastic typedBuf[dbmodels.IndicatorStochastic]
	obv        typedBuf[dbmodels.IndicatorOBV]
	vwap       typedBuf[dbmodels.IndicatorVWAP]
	adx        typedBuf[dbmodels.IndicatorADX]
}

// typedBuf is a generic per-table accumulator with its own mutex.
type typedBuf[T any] struct {
	mu    sync.Mutex
	rows  []T
}

func (b *typedBuf[T]) add(row T) int {
	b.mu.Lock()
	b.rows = append(b.rows, row)
	n := len(b.rows)
	b.mu.Unlock()
	return n
}

func (b *typedBuf[T]) drain() []T {
	b.mu.Lock()
	if len(b.rows) == 0 {
		b.mu.Unlock()
		return nil
	}
	snap := b.rows
	b.rows = make([]T, 0, len(snap))
	b.mu.Unlock()
	return snap
}

func (b *typedBuf[T]) pending() int {
	b.mu.Lock()
	n := len(b.rows)
	b.mu.Unlock()
	return n
}

// NewIndicatorBatchWriter creates a writer that flushes when any table reaches
// flushSize rows or when flushInterval elapses, whichever comes first.
func NewIndicatorBatchWriter(db *gorm.DB, flushSize int, flushInterval time.Duration) *IndicatorBatchWriter {
	return &IndicatorBatchWriter{
		db:        db,
		flushSize: flushSize,
		ticker:    time.NewTicker(flushInterval),
	}
}

// Run starts the timer-based flush loop. Call in a goroutine.
func (w *IndicatorBatchWriter) Run(ctx context.Context) {
	defer w.ticker.Stop()
	for {
		select {
		case <-w.ticker.C:
			w.FlushAll()
		case <-ctx.Done():
			w.FlushAll()
			return
		}
	}
}

// FlushAll drains and writes every table that has pending rows.
func (w *IndicatorBatchWriter) FlushAll() {
	w.flushSMA()
	w.flushEMA()
	w.flushRSI()
	w.flushMACD()
	w.flushATR()
	w.flushBollinger()
	w.flushStochastic()
	w.flushOBV()
	w.flushVWAP()
	w.flushADX()
}

// PendingCount returns total un-flushed rows across all tables (for metrics).
func (w *IndicatorBatchWriter) PendingCount() int {
	return w.sma.pending() + w.ema.pending() + w.rsi.pending() +
		w.macd.pending() + w.atr.pending() + w.bollinger.pending() +
		w.stochastic.pending() + w.obv.pending() + w.vwap.pending() + w.adx.pending()
}

// ---------------------------------------------------------------------------
// Typed Add methods — one per indicator
// ---------------------------------------------------------------------------

func (w *IndicatorBatchWriter) AddSMA(row dbmodels.IndicatorSMA) {
	if w.sma.add(row) >= w.flushSize {
		w.flushSMA()
	}
}

func (w *IndicatorBatchWriter) AddEMA(row dbmodels.IndicatorEMA) {
	if w.ema.add(row) >= w.flushSize {
		w.flushEMA()
	}
}

func (w *IndicatorBatchWriter) AddRSI(row dbmodels.IndicatorRSI) {
	if w.rsi.add(row) >= w.flushSize {
		w.flushRSI()
	}
}

func (w *IndicatorBatchWriter) AddMACD(row dbmodels.IndicatorMACD) {
	if w.macd.add(row) >= w.flushSize {
		w.flushMACD()
	}
}

func (w *IndicatorBatchWriter) AddATR(row dbmodels.IndicatorATR) {
	if w.atr.add(row) >= w.flushSize {
		w.flushATR()
	}
}

func (w *IndicatorBatchWriter) AddBollinger(row dbmodels.IndicatorBollingerBands) {
	if w.bollinger.add(row) >= w.flushSize {
		w.flushBollinger()
	}
}

func (w *IndicatorBatchWriter) AddStochastic(row dbmodels.IndicatorStochastic) {
	if w.stochastic.add(row) >= w.flushSize {
		w.flushStochastic()
	}
}

func (w *IndicatorBatchWriter) AddOBV(row dbmodels.IndicatorOBV) {
	if w.obv.add(row) >= w.flushSize {
		w.flushOBV()
	}
}

func (w *IndicatorBatchWriter) AddVWAP(row dbmodels.IndicatorVWAP) {
	if w.vwap.add(row) >= w.flushSize {
		w.flushVWAP()
	}
}

func (w *IndicatorBatchWriter) AddADX(row dbmodels.IndicatorADX) {
	if w.adx.add(row) >= w.flushSize {
		w.flushADX()
	}
}

// ---------------------------------------------------------------------------
// Typed flush methods — GORM receives a concrete []T, not []interface{}
// ---------------------------------------------------------------------------

func (w *IndicatorBatchWriter) flushSMA() {
	rows := w.sma.drain()
	if rows == nil {
		return
	}
	if err := w.db.Table("smas").Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "period"}, {Name: "timestamp"}},
		DoUpdates: clause.AssignmentColumns([]string{"value", "updated_at"}),
	}).CreateInBatches(rows, 500).Error; err != nil {
		zap.L().Error("batch flush: smas", zap.Error(err), zap.Int("rows", len(rows)))
	}
}

func (w *IndicatorBatchWriter) flushEMA() {
	rows := w.ema.drain()
	if rows == nil {
		return
	}
	if err := w.db.Table("emas").Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "period"}, {Name: "timestamp"}},
		DoUpdates: clause.AssignmentColumns([]string{"value", "updated_at"}),
	}).CreateInBatches(rows, 500).Error; err != nil {
		zap.L().Error("batch flush: emas", zap.Error(err), zap.Int("rows", len(rows)))
	}
}

func (w *IndicatorBatchWriter) flushRSI() {
	rows := w.rsi.drain()
	if rows == nil {
		return
	}
	if err := w.db.Table("rsis").Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "period"}, {Name: "timestamp"}},
		DoUpdates: clause.AssignmentColumns([]string{"value", "updated_at"}),
	}).CreateInBatches(rows, 500).Error; err != nil {
		zap.L().Error("batch flush: rsis", zap.Error(err), zap.Int("rows", len(rows)))
	}
}

func (w *IndicatorBatchWriter) flushMACD() {
	rows := w.macd.drain()
	if rows == nil {
		return
	}
	if err := w.db.Table("macds").Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "fast_period"}, {Name: "slow_period"}, {Name: "signal_period"}, {Name: "timestamp"}},
		DoUpdates: clause.AssignmentColumns([]string{"macd_line", "signal_line", "histogram", "updated_at"}),
	}).CreateInBatches(rows, 500).Error; err != nil {
		zap.L().Error("batch flush: macds", zap.Error(err), zap.Int("rows", len(rows)))
	}
}

func (w *IndicatorBatchWriter) flushATR() {
	rows := w.atr.drain()
	if rows == nil {
		return
	}
	if err := w.db.Table("atrs").Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "period"}, {Name: "timestamp"}},
		DoUpdates: clause.AssignmentColumns([]string{"value", "updated_at"}),
	}).CreateInBatches(rows, 500).Error; err != nil {
		zap.L().Error("batch flush: atrs", zap.Error(err), zap.Int("rows", len(rows)))
	}
}

func (w *IndicatorBatchWriter) flushBollinger() {
	rows := w.bollinger.drain()
	if rows == nil {
		return
	}
	if err := w.db.Table("bollinger_bands").Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "period"}, {Name: "num_std_dev"}, {Name: "timestamp"}},
		DoUpdates: clause.AssignmentColumns([]string{"upper_band", "middle_band", "lower_band", "updated_at"}),
	}).CreateInBatches(rows, 500).Error; err != nil {
		zap.L().Error("batch flush: bollinger_bands", zap.Error(err), zap.Int("rows", len(rows)))
	}
}

func (w *IndicatorBatchWriter) flushStochastic() {
	rows := w.stochastic.drain()
	if rows == nil {
		return
	}
	if err := w.db.Table("stochastics").Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "k_period"}, {Name: "d_period"}, {Name: "timestamp"}},
		DoUpdates: clause.AssignmentColumns([]string{"k_value", "d_value", "updated_at"}),
	}).CreateInBatches(rows, 500).Error; err != nil {
		zap.L().Error("batch flush: stochastics", zap.Error(err), zap.Int("rows", len(rows)))
	}
}

func (w *IndicatorBatchWriter) flushOBV() {
	rows := w.obv.drain()
	if rows == nil {
		return
	}
	if err := w.db.Table("obvs").Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "timestamp"}},
		DoUpdates: clause.AssignmentColumns([]string{"value", "updated_at"}),
	}).CreateInBatches(rows, 500).Error; err != nil {
		zap.L().Error("batch flush: obvs", zap.Error(err), zap.Int("rows", len(rows)))
	}
}

func (w *IndicatorBatchWriter) flushVWAP() {
	rows := w.vwap.drain()
	if rows == nil {
		return
	}
	if err := w.db.Table("vwaps").Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "timestamp"}},
		DoUpdates: clause.AssignmentColumns([]string{"value", "updated_at"}),
	}).CreateInBatches(rows, 500).Error; err != nil {
		zap.L().Error("batch flush: vwaps", zap.Error(err), zap.Int("rows", len(rows)))
	}
}

func (w *IndicatorBatchWriter) flushADX() {
	rows := w.adx.drain()
	if rows == nil {
		return
	}
	if err := w.db.Table("adxes").Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "instrument_token"}, {Name: "interval"}, {Name: "period"}, {Name: "timestamp"}},
		DoUpdates: clause.AssignmentColumns([]string{"adx_value", "plus_di", "minus_di", "updated_at"}),
	}).CreateInBatches(rows, 500).Error; err != nil {
		zap.L().Error("batch flush: adxes", zap.Error(err), zap.Int("rows", len(rows)))
	}
}
