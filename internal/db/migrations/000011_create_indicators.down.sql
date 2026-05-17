-- 000011_create_indicators.down.sql
-- Drop all tables created in the 000011_create_indicators.up.sql migration.
-- Tables are dropped in reverse dependency order where applicable.

-- Drop indicator tables
DROP TABLE IF EXISTS adxes CASCADE;
DROP TABLE IF EXISTS vwaps CASCADE;
DROP TABLE IF EXISTS obvs CASCADE;
DROP TABLE IF EXISTS bollinger_bands CASCADE;
DROP TABLE IF EXISTS stochastics CASCADE;
DROP TABLE IF EXISTS rsis CASCADE;
DROP TABLE IF EXISTS atrs CASCADE;
DROP TABLE IF EXISTS macds CASCADE;
DROP TABLE IF EXISTS emas CASCADE;
DROP TABLE IF EXISTS smas CASCADE;
DROP TABLE IF EXISTS ohlcv_candles CASCADE;

-- Note: TimescaleDB extension is NOT dropped here because it's a shared dependency.
-- The extension was created by pre_migration_enable_timescaledb.sql and should only
-- be removed manually if absolutely necessary, as other tables may depend on it.