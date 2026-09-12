// Command seed populates the database with deterministic, realistic-looking
// market data for local development, frontend work and paper-trading runs.
//
// It is deliberately conservative: it never truncates unless asked, it refuses
// to touch a database whose name looks like production unless forced, and
// --dry-run prints exactly what it would write without opening a transaction.
//
// See cmd/seed/README.md for usage.
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/db"
	"golang.org/x/crypto/bcrypt"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	"gorm.io/gorm/logger"
)

// priceScale matches candles.PriceScale: prices are stored as scaled int64.
const priceScale = 10000

type config struct {
	dsn         string
	users       int
	instruments int
	days        int
	intervals   string
	password    string
	rngSeed     int64
	truncate    bool
	dryRun      bool
	force       bool
	batchSize   int
}

func main() {
	cfg := parseFlags()

	if err := run(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "seed: %v\n", err)
		os.Exit(1)
	}
}

func parseFlags() config {
	var c config
	flag.StringVar(&c.dsn, "dsn", "", "Postgres DSN. Defaults to the DB_* environment variables.")
	flag.IntVar(&c.users, "users", 3, "Number of demo users to create")
	flag.IntVar(&c.instruments, "instruments", 10, "Number of instruments to create")
	flag.IntVar(&c.days, "days", 5, "Number of trading days of candles to generate")
	flag.StringVar(&c.intervals, "intervals", "1m,5m", "Comma-separated candle intervals")
	flag.StringVar(&c.password, "password", "Password123!", "Password for every demo user")
	flag.Int64Var(&c.rngSeed, "seed", 42, "RNG seed; the same seed always produces the same data")
	flag.BoolVar(&c.truncate, "truncate", false, "Delete existing seeded rows first")
	flag.BoolVar(&c.dryRun, "dry-run", false, "Print what would be written without writing")
	flag.BoolVar(&c.force, "force", false, "Allow running against a database whose name looks like production")
	flag.IntVar(&c.batchSize, "batch", 1000, "Insert batch size")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: go run ./cmd/seed [flags]\n\nFlags:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  go run ./cmd/seed --dry-run\n")
		fmt.Fprintf(os.Stderr, "  go run ./cmd/seed --users 3 --instruments 10 --days 5\n")
		fmt.Fprintf(os.Stderr, "  go run ./cmd/seed --truncate --days 30 --intervals 1m,5m,15m\n")
	}
	flag.Parse()
	return c
}

func run(c config) error {
	if c.users < 0 || c.instruments < 0 || c.days < 0 {
		return fmt.Errorf("counts must not be negative")
	}
	intervals := splitIntervals(c.intervals)
	if len(intervals) == 0 {
		return fmt.Errorf("at least one interval is required")
	}

	dsn, dbName, err := resolveDSN(c.dsn)
	if err != nil {
		return err
	}

	if looksLikeProduction(dbName) && !c.force {
		return fmt.Errorf("database %q looks like production; re-run with --force if you really mean it", dbName)
	}

	plan := planFor(c, intervals)
	printPlan(dbName, c, intervals, plan)

	if c.dryRun {
		fmt.Println("\n--dry-run: nothing was written.")
		return nil
	}

	gdb, err := gorm.Open(postgres.Open(dsn), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Silent),
	})
	if err != nil {
		return fmt.Errorf("connect: %w", err)
	}
	sqlDB, err := gdb.DB()
	if err != nil {
		return fmt.Errorf("sql handle: %w", err)
	}
	defer sqlDB.Close()

	if err := sqlDB.Ping(); err != nil {
		return fmt.Errorf("ping: %w", err)
	}

	start := time.Now()
	rng := rand.New(rand.NewSource(c.rngSeed))

	if c.truncate {
		if err := truncateSeeded(gdb); err != nil {
			return fmt.Errorf("truncate: %w", err)
		}
		fmt.Println("✓ truncated seeded tables")
	}

	users, err := seedUsers(gdb, c, rng)
	if err != nil {
		return fmt.Errorf("users: %w", err)
	}
	fmt.Printf("✓ %d users\n", len(users))

	instruments, err := seedInstruments(gdb, c)
	if err != nil {
		return fmt.Errorf("instruments: %w", err)
	}
	fmt.Printf("✓ %d instruments\n", len(instruments))

	candles, err := seedCandles(gdb, c, instruments, intervals, rng)
	if err != nil {
		return fmt.Errorf("candles: %w", err)
	}
	fmt.Printf("✓ %d candles across %v\n", candles, intervals)

	fmt.Printf("\nDone in %s.\n", time.Since(start).Round(time.Millisecond))
	fmt.Printf("Log in with: %s / %s\n", demoEmail(0), c.password)
	return nil
}

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------

// resolveDSN builds the DSN from the flag or the DB_* environment variables,
// and returns the database name for the safety check.
func resolveDSN(flagDSN string) (dsn, dbName string, err error) {
	if flagDSN != "" {
		return flagDSN, dbNameFromDSN(flagDSN), nil
	}

	host := envOr("DB_HOST", "localhost")
	port := envOr("DB_PORT", "5432")
	user := os.Getenv("DB_USER")
	pass := os.Getenv("DB_PASSWORD")
	name := os.Getenv("DB_NAME")

	if user == "" || name == "" {
		return "", "", fmt.Errorf("set DB_USER and DB_NAME (or pass --dsn); DB_HOST/DB_PORT default to localhost:5432")
	}

	return fmt.Sprintf(
		"host=%s port=%s user=%s password=%s dbname=%s sslmode=disable TimeZone=Asia/Kolkata",
		host, port, user, pass, name,
	), name, nil
}

func dbNameFromDSN(dsn string) string {
	for _, part := range strings.Fields(dsn) {
		if strings.HasPrefix(part, "dbname=") {
			return strings.TrimPrefix(part, "dbname=")
		}
	}
	// URL form: postgres://user:pass@host:port/dbname?...
	if i := strings.LastIndex(dsn, "/"); i >= 0 && i+1 < len(dsn) {
		rest := dsn[i+1:]
		if j := strings.IndexByte(rest, '?'); j >= 0 {
			rest = rest[:j]
		}
		return rest
	}
	return ""
}

// looksLikeProduction is a guard, not a security control: it catches the
// obvious mistake of pointing the seeder at a live database.
func looksLikeProduction(name string) bool {
	n := strings.ToLower(name)
	for _, marker := range []string{"prod", "production", "live", "main"} {
		if strings.Contains(n, marker) {
			return true
		}
	}
	return false
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// ---------------------------------------------------------------------------
// Plan / reporting
// ---------------------------------------------------------------------------

type plan struct {
	users       int
	instruments int
	candles     int
}

func planFor(c config, intervals []string) plan {
	total := 0
	for _, iv := range intervals {
		total += c.instruments * c.days * barsPerDay(iv)
	}
	return plan{users: c.users, instruments: c.instruments, candles: total}
}

func printPlan(dbName string, c config, intervals []string, p plan) {
	fmt.Printf("Target database : %s\n", dbName)
	fmt.Printf("RNG seed        : %d (same seed -> same data)\n", c.rngSeed)
	fmt.Printf("Truncate first  : %v\n", c.truncate)
	fmt.Println("Will write:")
	fmt.Printf("  users        : %d\n", p.users)
	fmt.Printf("  instruments  : %d\n", p.instruments)
	fmt.Printf("  candles      : ~%d (%d days x %v)\n", p.candles, c.days, intervals)
}

// ---------------------------------------------------------------------------
// Seeding
// ---------------------------------------------------------------------------

// truncateSeeded removes rows in FK-safe order. It deliberately does not touch
// users, because orders and positions reference them.
func truncateSeeded(gdb *gorm.DB) error {
	stmts := []string{
		"TRUNCATE TABLE ohlcv_candles",
		"TRUNCATE TABLE market_data",
	}
	for _, s := range stmts {
		if err := gdb.Exec(s).Error; err != nil {
			// A missing table is fine: migrations may not have created every
			// optional table in this environment.
			if strings.Contains(strings.ToLower(err.Error()), "does not exist") {
				continue
			}
			return fmt.Errorf("%s: %w", s, err)
		}
	}
	return nil
}

func demoEmail(i int) string { return fmt.Sprintf("demo%d@example.com", i+1) }

func seedUsers(gdb *gorm.DB, c config, _ *rand.Rand) ([]db.User, error) {
	hash, err := bcrypt.GenerateFromPassword([]byte(c.password), bcrypt.DefaultCost)
	if err != nil {
		return nil, fmt.Errorf("hash password: %w", err)
	}

	users := make([]db.User, 0, c.users)
	for i := 0; i < c.users; i++ {
		users = append(users, db.User{
			Email:        demoEmail(i),
			PasswordHash: string(hash),
			UserName:     fmt.Sprintf("Demo User %d", i+1),
			IsActive:     true,
		})
	}
	if len(users) == 0 {
		return users, nil
	}

	// Re-running the seeder must not fail on the unique email index.
	err = gdb.Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "email"}},
		DoUpdates: clause.AssignmentColumns([]string{"password_hash", "user_name", "is_active"}),
	}).CreateInBatches(&users, c.batchSize).Error
	return users, err
}

// symbolPool provides realistic NSE names; the seeder cycles through it.
var symbolPool = []struct {
	symbol string
	name   string
	price  float64
}{
	{"RELIANCE", "Reliance Industries Ltd", 2900},
	{"TCS", "Tata Consultancy Services Ltd", 3900},
	{"HDFCBANK", "HDFC Bank Ltd", 1650},
	{"INFY", "Infosys Ltd", 1500},
	{"ICICIBANK", "ICICI Bank Ltd", 1100},
	{"SBIN", "State Bank of India", 820},
	{"BHARTIARTL", "Bharti Airtel Ltd", 1400},
	{"ITC", "ITC Ltd", 430},
	{"LT", "Larsen & Toubro Ltd", 3600},
	{"AXISBANK", "Axis Bank Ltd", 1150},
	{"WIPRO", "Wipro Ltd", 520},
	{"MARUTI", "Maruti Suzuki India Ltd", 12500},
	{"SUNPHARMA", "Sun Pharmaceutical Industries Ltd", 1750},
	{"TATAMOTORS", "Tata Motors Ltd", 980},
	{"HINDUNILVR", "Hindustan Unilever Ltd", 2400},
}

// baseToken keeps seeded instrument tokens clear of real Kite tokens.
const baseToken uint32 = 900000

func seedInstruments(gdb *gorm.DB, c config) ([]db.Instrument, error) {
	out := make([]db.Instrument, 0, c.instruments)
	now := time.Now()

	for i := 0; i < c.instruments; i++ {
		p := symbolPool[i%len(symbolPool)]
		symbol := p.symbol
		// Past one full cycle of the pool, suffix to keep symbols unique.
		if i >= len(symbolPool) {
			symbol = fmt.Sprintf("%s%d", p.symbol, i/len(symbolPool)+1)
		}
		out = append(out, db.Instrument{
			InstrumentToken: baseToken + uint32(i),
			Exchange:        "NSE",
			Tradingsymbol:   symbol,
			InstrumentType:  "EQ",
			Name:            p.name,
			Segment:         "NSECM",
			TickSize:        0.05,
			LotSize:         1,
			LastUpdated:     now,
		})
	}
	if len(out) == 0 {
		return out, nil
	}

	err := gdb.Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "instrument_token"}},
		DoUpdates: clause.AssignmentColumns([]string{"tradingsymbol", "name", "last_updated"}),
	}).CreateInBatches(&out, c.batchSize).Error
	return out, err
}

// barsPerDay returns how many bars of an interval fit in a 375-minute session
// (09:15 to 15:30).
func barsPerDay(interval string) int {
	const sessionMinutes = 375
	m := intervalMinutes(interval)
	if m <= 0 {
		return 0
	}
	return sessionMinutes / m
}

func intervalMinutes(interval string) int {
	d, err := time.ParseDuration(interval)
	if err != nil {
		return 0
	}
	return int(d.Minutes())
}

// seedCandles generates a geometric-random-walk price series per instrument
// and writes OHLCV bars for each interval.
func seedCandles(gdb *gorm.DB, c config, instruments []db.Instrument, intervals []string, rng *rand.Rand) (int, error) {
	if len(instruments) == 0 || c.days == 0 {
		return 0, nil
	}

	loc, err := time.LoadLocation("Asia/Kolkata")
	if err != nil {
		return 0, fmt.Errorf("load timezone: %w", err)
	}

	total := 0
	batch := make([]db.OHLCVCandle, 0, c.batchSize)

	flush := func() error {
		if len(batch) == 0 {
			return nil
		}
		err := gdb.Clauses(clause.OnConflict{
			Columns: []clause.Column{
				{Name: "instrument_token"}, {Name: "interval"}, {Name: "timestamp"},
			},
			DoUpdates: clause.AssignmentColumns([]string{"open", "high", "low", "close", "volume", "trade_count"}),
		}).CreateInBatches(&batch, c.batchSize).Error
		batch = batch[:0]
		return err
	}

	for idx, inst := range instruments {
		basePrice := symbolPool[idx%len(symbolPool)].price

		for _, interval := range intervals {
			mins := intervalMinutes(interval)
			if mins <= 0 {
				continue
			}
			bars := barsPerDay(interval)
			price := basePrice

			// Walk backwards from today so the newest bar is the most recent
			// completed session.
			for d := c.days - 1; d >= 0; d-- {
				day := tradingDay(time.Now().In(loc), d, loc)

				for b := 0; b < bars; b++ {
					ts := day.Add(time.Duration(b*mins) * time.Minute)
					o, h, l, cl, vol := nextBar(&price, mins, rng)

					batch = append(batch, db.OHLCVCandle{
						InstrumentToken: inst.InstrumentToken,
						Interval:        interval,
						Timestamp:       ts,
						Open:            scaled(o),
						High:            scaled(h),
						Low:             scaled(l),
						Close:           scaled(cl),
						Volume:          vol,
						TradeCount:      uint32(vol / 40),
						DataSource:      "seed",
					})
					total++

					if len(batch) >= c.batchSize {
						if err := flush(); err != nil {
							return total, err
						}
					}
				}
			}
		}
	}

	if err := flush(); err != nil {
		return total, err
	}
	return total, nil
}

// tradingDay returns the 09:15 open of the trading day `back` sessions before
// `from`, skipping weekends. Exchange holidays are not modelled.
func tradingDay(from time.Time, back int, loc *time.Location) time.Time {
	d := time.Date(from.Year(), from.Month(), from.Day(), 9, 15, 0, 0, loc)
	for skipped := 0; skipped < back; {
		d = d.AddDate(0, 0, -1)
		if wd := d.Weekday(); wd != time.Saturday && wd != time.Sunday {
			skipped++
		}
	}
	// Ensure the anchor day itself is a weekday.
	for d.Weekday() == time.Saturday || d.Weekday() == time.Sunday {
		d = d.AddDate(0, 0, -1)
	}
	return d
}

// nextBar advances the price by a geometric random walk and returns one OHLCV
// bar. Volatility scales with the square root of time, as it does in reality,
// so 5m bars are wider than 1m bars rather than identical.
func nextBar(price *float64, minutes int, rng *rand.Rand) (o, h, l, c float64, volume int64) {
	const annualVol = 0.25
	// ~98,280 minutes of trading in a year (252 sessions x 375 minutes + slack).
	perMinuteVol := annualVol / math.Sqrt(98_280)
	sigma := perMinuteVol * math.Sqrt(float64(minutes))

	o = *price
	drift := rng.NormFloat64() * sigma
	c = o * math.Exp(drift)

	// Wicks extend beyond the body by a fraction of the same volatility.
	wick := o * sigma * math.Abs(rng.NormFloat64())
	h = math.Max(o, c) + wick
	l = math.Min(o, c) - wick
	if l <= 0 {
		l = math.Min(o, c) * 0.99
	}

	*price = c

	base := 2000 + rng.Intn(8000)
	volume = int64(base * minutes)
	return o, h, l, c, volume
}

func scaled(v float64) int64 { return int64(math.Round(v * priceScale)) }

func splitIntervals(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}
