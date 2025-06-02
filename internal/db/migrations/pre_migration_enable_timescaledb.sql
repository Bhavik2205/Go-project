-- pre_migration_enable_timescaledb.sql
-- This script enables the TimescaleDB extension in your PostgreSQL database.
-- It should be run BEFORE any migrations that create hypertables.
-- This operation typically requires superuser or rds_superuser privileges.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Optional: You might want to set default chunk time interval for hypertables
-- ALTER DATABASE your_database_name SET timescaledb.default_chunk_interval = '1 day'::interval;
-- Replace 'your_database_name' with the actual name of your trading_bot_db
-- Example: ALTER DATABASE trading_bot_db SET timescaledb.default_chunk_interval = '1 day'::interval;

-- Output for confirmation
SELECT 'TimescaleDB extension enabled successfully or already existed.' AS status;