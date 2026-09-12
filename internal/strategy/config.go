package strategy

import (
	"fmt"
	"os"
	"time"

	// The session window is defined in exchange local time, so a missing
	// system tzdata must not silently shift trading hours. Embedding the
	// database keeps "Asia/Kolkata" resolvable on scratch/distroless images
	// and on hosts without a zoneinfo directory.
	_ "time/tzdata"

	"gopkg.in/yaml.v3"
)

// Config is the engine section of configs/strategy.yaml.
type Config struct {
	Enabled bool `yaml:"enabled"`

	// SignalBuffer is the depth of the outbound order channel. When it fills,
	// the engine sheds new orders rather than blocking the market data path.
	SignalBuffer int `yaml:"signal_buffer"`

	// Capital is the notional account size used for position sizing.
	Capital float64 `yaml:"capital"`

	// Session bounds the trading window in the exchange timezone. Entries are
	// refused outside it and open positions are force-exited at SessionEnd.
	Session SessionConfig `yaml:"session"`

	Risk RiskConfig `yaml:"risk"`

	Strategies []InstanceConfig `yaml:"strategies"`
}

// SessionConfig describes the tradable window.
type SessionConfig struct {
	Timezone string `yaml:"timezone"` // e.g. "Asia/Kolkata"
	Open     string `yaml:"open"`     // "09:15"
	Close    string `yaml:"close"`    // "15:30"
	// SquareOff is when intraday positions are force-closed, e.g. "15:15".
	SquareOff string `yaml:"square_off"`

	loc                         *time.Location
	openMin, closeMin, sqOffMin int
}

// RiskConfig bounds what the engine is allowed to do.
type RiskConfig struct {
	MaxTradesPerDay              int     `yaml:"max_trades_per_day"`
	MaxLossPerDayPercentage      float64 `yaml:"max_loss_per_day_percentage"`
	TradeSizePercentageOfCapital float64 `yaml:"trade_size_percentage_of_capital"`
	MaxOpenPositions             int     `yaml:"max_open_positions"`
	// DefaultStopLossPercentage / DefaultTargetPercentage apply when a
	// strategy returns an entry without explicit protective levels.
	DefaultStopLossPercentage float64 `yaml:"default_stop_loss_percentage"`
	DefaultTargetPercentage   float64 `yaml:"default_target_percentage"`
	// AllowShort gates short entries (many cash segments disallow them).
	AllowShort bool `yaml:"allow_short"`
}

// InstanceConfig enables one registered strategy over a set of intervals and
// instruments.
type InstanceConfig struct {
	Name    string `yaml:"name"`
	Enabled bool   `yaml:"enabled"`
	// Intervals the strategy runs on, e.g. ["1m","5m"]. Required.
	Intervals []string `yaml:"intervals"`
	// Instruments restricts the strategy to these tokens. Empty means all.
	Instruments []uint32 `yaml:"instruments"`
	// Params are strategy-specific tunables.
	Params Params `yaml:"params"`
}

// fileShape lets us read only the engine section, leaving the legacy
// intraday/swing blocks in strategy.yaml untouched for existing readers.
type fileShape struct {
	Engine Config `yaml:"engine"`
}

// LoadConfig reads the engine config from a strategy YAML file.
func LoadConfig(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read strategy config %s: %w", path, err)
	}
	var f fileShape
	if err := yaml.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("parse strategy config %s: %w", path, err)
	}
	cfg := f.Engine
	if err := cfg.Normalize(); err != nil {
		return nil, fmt.Errorf("strategy config %s: %w", path, err)
	}
	return &cfg, nil
}

// Normalize applies defaults and validates the config. It is exported so
// tests and programmatic callers can build a Config without a file.
func (c *Config) Normalize() error {
	if c.SignalBuffer <= 0 {
		c.SignalBuffer = 4096
	}
	if c.Capital <= 0 {
		c.Capital = 1_000_000
	}

	if c.Risk.MaxTradesPerDay <= 0 {
		c.Risk.MaxTradesPerDay = 10
	}
	if c.Risk.MaxOpenPositions <= 0 {
		c.Risk.MaxOpenPositions = 5
	}
	if c.Risk.TradeSizePercentageOfCapital <= 0 {
		c.Risk.TradeSizePercentageOfCapital = 0.05
	}
	if c.Risk.MaxLossPerDayPercentage <= 0 {
		c.Risk.MaxLossPerDayPercentage = 0.02
	}
	if c.Risk.DefaultStopLossPercentage <= 0 {
		c.Risk.DefaultStopLossPercentage = 0.005
	}
	if c.Risk.DefaultTargetPercentage <= 0 {
		c.Risk.DefaultTargetPercentage = 0.015
	}

	if err := c.Session.normalize(); err != nil {
		return err
	}

	seen := make(map[string]struct{}, len(c.Strategies))
	for i := range c.Strategies {
		s := &c.Strategies[i]
		if s.Name == "" {
			return fmt.Errorf("strategies[%d]: name is required", i)
		}
		if !s.Enabled {
			continue
		}
		if _, dup := seen[s.Name]; dup {
			return fmt.Errorf("strategies[%d]: %q listed twice", i, s.Name)
		}
		seen[s.Name] = struct{}{}

		if !IsRegistered(s.Name) {
			return fmt.Errorf("strategies[%d]: %q is not a registered strategy (known: %v)", i, s.Name, Names())
		}
		if len(s.Intervals) == 0 {
			return fmt.Errorf("strategies[%d] (%s): at least one interval is required", i, s.Name)
		}
		if s.Params == nil {
			s.Params = Params{}
		}
	}
	return nil
}

// EnabledStrategies returns only the instances that are switched on.
func (c *Config) EnabledStrategies() []InstanceConfig {
	out := make([]InstanceConfig, 0, len(c.Strategies))
	for _, s := range c.Strategies {
		if s.Enabled {
			out = append(out, s)
		}
	}
	return out
}

func (s *SessionConfig) normalize() error {
	if s.Timezone == "" {
		s.Timezone = "Asia/Kolkata"
	}
	loc, err := time.LoadLocation(s.Timezone)
	if err != nil {
		// Windows and scratch containers often lack a tzdata database. Falling
		// back to UTC would silently shift the whole trading session, so this
		// is a hard failure.
		return fmt.Errorf("load timezone %q: %w", s.Timezone, err)
	}
	s.loc = loc

	if s.Open == "" {
		s.Open = "09:15"
	}
	if s.Close == "" {
		s.Close = "15:30"
	}
	if s.SquareOff == "" {
		s.SquareOff = "15:15"
	}

	if s.openMin, err = parseHHMM(s.Open); err != nil {
		return fmt.Errorf("session.open: %w", err)
	}
	if s.closeMin, err = parseHHMM(s.Close); err != nil {
		return fmt.Errorf("session.close: %w", err)
	}
	if s.sqOffMin, err = parseHHMM(s.SquareOff); err != nil {
		return fmt.Errorf("session.square_off: %w", err)
	}
	if s.openMin >= s.closeMin {
		return fmt.Errorf("session.open (%s) must be before session.close (%s)", s.Open, s.Close)
	}
	if s.sqOffMin > s.closeMin {
		return fmt.Errorf("session.square_off (%s) must not be after session.close (%s)", s.SquareOff, s.Close)
	}
	return nil
}

// Location returns the session timezone.
func (s *SessionConfig) Location() *time.Location { return s.loc }

// minutesOf converts t into minutes-since-midnight in the session timezone.
func (s *SessionConfig) minutesOf(t time.Time) int {
	lt := t.In(s.loc)
	return lt.Hour()*60 + lt.Minute()
}

// AllowsEntry reports whether a new position may be opened at t.
func (s *SessionConfig) AllowsEntry(t time.Time) bool {
	m := s.minutesOf(t)
	return m >= s.openMin && m < s.sqOffMin
}

// IsSquareOff reports whether t is at or past the intraday square-off time.
func (s *SessionConfig) IsSquareOff(t time.Time) bool {
	return s.minutesOf(t) >= s.sqOffMin
}

func parseHHMM(v string) (int, error) {
	t, err := time.Parse("15:04", v)
	if err != nil {
		return 0, fmt.Errorf("expected HH:MM, got %q", v)
	}
	return t.Hour()*60 + t.Minute(), nil
}
