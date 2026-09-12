package monitor

import (
	"context"
	"sync/atomic"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/db"
	"go.uber.org/zap"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

// TradeRecord is one completed fill plus the position it produced.
type TradeRecord struct {
	BrokerOrderID   string
	UserID          uint
	InstrumentToken uint32
	Symbol          string
	StrategyName    string
	TransactionType string // BUY or SELL
	Quantity        int64
	Price           float64
	Fee             float64
	RealizedPnL     float64
	Position        PaperPosition
	TradeType       string // PAPER or LIVE
	At              time.Time
}

// TradeSink accepts fills for persistence. Record must not block.
type TradeSink interface {
	Record(r TradeRecord)
}

// NopSink discards records, for memory-only paper runs.
type NopSink struct{}

// Record does nothing.
func (NopSink) Record(TradeRecord) {}

// DBTradeSink persists fills to the orders, trades and positions tables on a
// background goroutine. Record is a non-blocking send: if the database falls
// behind, records are dropped and counted rather than stalling execution.
//
// Dropping trade history is a real cost, so the queue is sized generously and
// every drop is logged and counted for alerting.
type DBTradeSink struct {
	db *gorm.DB
	ch chan TradeRecord

	written atomic.Uint64
	dropped atomic.Uint64
	failed  atomic.Uint64
}

// NewDBTradeSink builds a sink with the given queue depth.
func NewDBTradeSink(dbc *db.DBClient, buffer int) *DBTradeSink {
	if buffer <= 0 {
		buffer = 8192
	}
	return &DBTradeSink{db: dbc.DB, ch: make(chan TradeRecord, buffer)}
}

// Record queues a fill. Never blocks.
func (s *DBTradeSink) Record(r TradeRecord) {
	select {
	case s.ch <- r:
	default:
		n := s.dropped.Add(1)
		if n%100 == 1 { // avoid a log storm while the DB is down
			zap.L().Error("trade persistence queue full, dropping record",
				zap.String("broker_order_id", r.BrokerOrderID),
				zap.Uint64("dropped_total", n))
		}
	}
}

// Run drains the queue until ctx is cancelled.
func (s *DBTradeSink) Run(ctx context.Context) {
	zap.L().Info("✅ Trade persistence started")
	for {
		select {
		case r := <-s.ch:
			s.persist(r)
		case <-ctx.Done():
			// Drain what is already queued so a clean shutdown does not lose
			// trades that were accepted.
			for {
				select {
				case r := <-s.ch:
					s.persist(r)
				default:
					zap.L().Info("Trade persistence stopped",
						zap.Uint64("written", s.written.Load()),
						zap.Uint64("dropped", s.dropped.Load()),
						zap.Uint64("failed", s.failed.Load()))
					return
				}
			}
		}
	}
}

// persist writes the order, its trade and the resulting position in one
// transaction so the three tables can never disagree.
func (s *DBTradeSink) persist(r TradeRecord) {
	err := s.db.Transaction(func(tx *gorm.DB) error {
		order := db.Order{
			UserID:          r.UserID,
			InstrumentToken: r.InstrumentToken,
			BrokerOrderID:   r.BrokerOrderID,
			StrategyName:    r.StrategyName,
			OrderType:       "MARKET",
			TransactionType: r.TransactionType,
			Quantity:        int(r.Quantity),
			Price:           r.Price,
			Status:          "FILLED",
			PlacedAt:        r.At,
			FilledQuantity:  int(r.Quantity),
			FilledPrice:     r.Price,
			Product:         "MIS",
			TradeType:       r.TradeType,
		}
		if err := tx.Create(&order).Error; err != nil {
			return err
		}

		trade := db.Trade{
			OrderID:         order.ID,
			UserID:          r.UserID,
			InstrumentToken: r.InstrumentToken,
			TradeID:         r.BrokerOrderID + "-T",
			TransactionType: r.TransactionType,
			Quantity:        int(r.Quantity),
			Price:           r.Price,
			TradeTime:       r.At,
			TradeType:       r.TradeType,
		}
		if err := tx.Create(&trade).Error; err != nil {
			return err
		}

		// positions is unique on (user_id, instrument_token, product), so
		// upsert rather than insert.
		pos := db.Position{
			UserID:          r.UserID,
			InstrumentToken: r.InstrumentToken,
			TradingSymbol:   r.Symbol,
			Product:         "MIS",
			Quantity:        int(r.Position.Quantity),
			AveragePrice:    r.Position.AveragePrice,
			LastPrice:       r.Position.LastPrice,
			RealizedPnL:     r.Position.RealizedPnL,
			UnrealizedPnL:   r.Position.UnrealizedPnL(),
			UpdatedAt:       r.At,
			TradeType:       r.TradeType,
		}
		return tx.Clauses(clause.OnConflict{
			Columns: []clause.Column{{Name: "user_id"}, {Name: "instrument_token"}, {Name: "product"}},
			DoUpdates: clause.AssignmentColumns([]string{
				"quantity", "average_price", "last_price",
				"realized_pnl", "unrealized_pnl", "updated_at", "trading_symbol",
			}),
		}).Create(&pos).Error
	})

	if err != nil {
		n := s.failed.Add(1)
		if n%100 == 1 {
			zap.L().Error("failed to persist trade",
				zap.String("broker_order_id", r.BrokerOrderID),
				zap.Uint64("failed_total", n),
				zap.Error(err))
		}
		return
	}
	s.written.Add(1)
}

// SinkStats reports persistence counters.
type SinkStats struct {
	Written uint64 `json:"written"`
	Dropped uint64 `json:"dropped"`
	Failed  uint64 `json:"failed"`
	Queued  int    `json:"queued"`
}

// Stats returns persistence counters for the monitoring API.
func (s *DBTradeSink) Stats() SinkStats {
	return SinkStats{
		Written: s.written.Load(),
		Dropped: s.dropped.Load(),
		Failed:  s.failed.Load(),
		Queued:  len(s.ch),
	}
}
