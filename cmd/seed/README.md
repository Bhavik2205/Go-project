# cmd/seed — database seeder

Populates the database with deterministic, realistic-looking market data for
local development, frontend work and paper-trading runs.

**Nothing here has been run against your database.** Every command below is
yours to execute.

## Before you start

1. **Postgres must be running** and reachable.
2. **Migrations must already be applied.** The seeder inserts rows; it does not
   create tables. Run migrations first:
   ```bash
   ./scripts/db_migrate.sh
   ```
3. **Connection settings** come from the `DB_*` variables in `.env`
   (`DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`), or pass
   `--dsn` explicitly.

## Recommended: seed a scratch database

Keeping seeded data out of your working database means you can wipe and
re-seed freely.

```bash
# 1. create it
createdb -h localhost -U "$DB_USER" mlbot_seed

# 2. point migrations and the seeder at it
DB_NAME=mlbot_seed ./scripts/db_migrate.sh
DB_NAME=mlbot_seed go run ./cmd/seed
```

## Usage

```bash
# See exactly what would be written; connects to nothing.
go run ./cmd/seed --dry-run

# Default seed: 3 users, 10 instruments, 5 trading days of 1m + 5m candles
go run ./cmd/seed

# A heavier set for frontend charts
go run ./cmd/seed --days 30 --intervals 1m,5m,15m --instruments 15

# Wipe previously seeded candles first
go run ./cmd/seed --truncate --days 10

# Explicit DSN instead of the DB_* env vars
go run ./cmd/seed --dsn "host=localhost port=5432 user=postgres password=secret dbname=mlbot_seed sslmode=disable"
```

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dsn` | from `DB_*` env | Postgres DSN |
| `--users` | `3` | Demo users to create |
| `--instruments` | `10` | Instruments to create |
| `--days` | `5` | Trading days of candles (weekends skipped) |
| `--intervals` | `1m,5m` | Comma-separated candle intervals |
| `--password` | `Password123!` | Password for every demo user |
| `--seed` | `42` | RNG seed — the same seed always produces identical data |
| `--truncate` | `false` | Delete existing candle/market data rows first |
| `--dry-run` | `false` | Print the plan and exit without connecting |
| `--force` | `false` | Permit a database whose name looks like production |
| `--batch` | `1000` | Insert batch size |

## Safety behaviour

- **Refuses production-looking names.** A `DB_NAME` containing `prod`,
  `production`, `live` or `main` is rejected unless you pass `--force`.
- **`--truncate` never touches `users`.** Orders and positions reference them,
  so users are upserted rather than deleted.
- **Re-running is safe.** Users, instruments and candles all upsert on their
  unique keys, so a second run updates rather than erroring.
- **`--dry-run` opens no connection at all.**

## What gets created

| Table | Content |
| --- | --- |
| `users` | `demo1@example.com` … `demoN@example.com`, bcrypt-hashed password |
| `instruments` | NSE equities (RELIANCE, TCS, HDFCBANK…), tokens from `900000` |
| `ohlcv_candles` | OHLCV bars per instrument per interval, `data_source = 'seed'` |

Instrument tokens start at **900000** deliberately, so seeded rows never
collide with real Zerodha instrument tokens.

Prices follow a geometric random walk with volatility scaled by the square
root of time, so 5m bars are genuinely wider than 1m bars rather than being
the same series relabelled. Sessions run 09:15–15:30 IST and weekends are
skipped; exchange holidays are not modelled.

Log in to the app afterwards with:

```
email:    demo1@example.com
password: Password123!
```

## Verifying afterwards

```sql
SELECT count(*) FROM users WHERE email LIKE 'demo%@example.com';
SELECT count(*) FROM instruments WHERE instrument_token >= 900000;
SELECT interval, count(*) FROM ohlcv_candles WHERE data_source = 'seed' GROUP BY interval;
```

## Removing seeded data

```sql
DELETE FROM ohlcv_candles WHERE data_source = 'seed';
DELETE FROM instruments  WHERE instrument_token >= 900000;
DELETE FROM users        WHERE email LIKE 'demo%@example.com';
```

Delete in that order — instruments and users are referenced by other tables.
