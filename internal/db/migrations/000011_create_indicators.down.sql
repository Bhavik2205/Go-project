-- 000011_create_indicators.down.sql

-- Drop all tables created in the 000011_create_indicators.up.sql migration.
-- Tables are dropped in reverse dependency order where applicable,
-- but CASCADE is generally used for simplicity to handle foreign key constraints
-- if they were implicitly added by GORM. For these specific tables,
-- they generally don't have FKs pointing to each other, so order is less critical.

-- DROP TABLE IF EXISTS indicator_adxes CASCADE;
-- DROP TABLE IF EXISTS indicator_vwaps CASCADE;
-- DROP TABLE IF EXISTS indicator_obvs CASCADE;
-- DROP TABLE IF EXISTS indicator_bollinger_bands CASCADE;
-- DROP TABLE IF EXISTS indicator_stochastics CASCADE;
-- DROP TABLE IF EXISTS indicator_rsis CASCADE;
-- DROP TABLE IF EXISTS indicator_atrs CASCADE;
-- DROP TABLE IF EXISTS indicator_macds CASCADE;
-- DROP TABLE IF EXISTS indicator_emas CASCADE;
-- DROP TABLE IF EXISTS indicator_smas CASCADE;
-- DROP TABLE IF EXISTS ohlcv_candles CASCADE;

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

-- If you explicitly enabled the TimescaleDB extension in your up.sql,
-- and if this is the ONLY place it's being enabled in your entire migration history,
-- you might consider dropping it here. However, it's generally better to
-- enable extensions once at the very beginning of your database setup
-- and not drop them with every migration rollback unless you have a very specific reason.
-- For a trading bot, you'll almost certainly always want TimescaleDB enabled.
-- Uncomment the line below ONLY if you want to remove the extension on rollback.
-- DROP EXTENSION IF EXISTS timescaledb;