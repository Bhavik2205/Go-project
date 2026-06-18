-- Migration 000018 rollback: Revert price columns from BIGINT back to NUMERIC.
-- Divides stored scaled integers by 10000 to recover original decimal values.

ALTER TABLE market_data
    ALTER COLUMN last_price          TYPE NUMERIC USING (last_price::NUMERIC / 10000),
    ALTER COLUMN average_trade_price TYPE NUMERIC USING (average_trade_price::NUMERIC / 10000),
    ALTER COLUMN net_change          TYPE NUMERIC USING (net_change::NUMERIC / 10000),
    ALTER COLUMN open                TYPE NUMERIC USING (open::NUMERIC / 10000),
    ALTER COLUMN high                TYPE NUMERIC USING (high::NUMERIC / 10000),
    ALTER COLUMN low                 TYPE NUMERIC USING (low::NUMERIC / 10000),
    ALTER COLUMN close               TYPE NUMERIC USING (close::NUMERIC / 10000),
    ALTER COLUMN bid_price1          TYPE NUMERIC USING (bid_price1::NUMERIC / 10000),
    ALTER COLUMN bid_price2          TYPE NUMERIC USING (bid_price2::NUMERIC / 10000),
    ALTER COLUMN bid_price3          TYPE NUMERIC USING (bid_price3::NUMERIC / 10000),
    ALTER COLUMN bid_price4          TYPE NUMERIC USING (bid_price4::NUMERIC / 10000),
    ALTER COLUMN bid_price5          TYPE NUMERIC USING (bid_price5::NUMERIC / 10000),
    ALTER COLUMN ask_price1          TYPE NUMERIC USING (ask_price1::NUMERIC / 10000),
    ALTER COLUMN ask_price2          TYPE NUMERIC USING (ask_price2::NUMERIC / 10000),
    ALTER COLUMN ask_price3          TYPE NUMERIC USING (ask_price3::NUMERIC / 10000),
    ALTER COLUMN ask_price4          TYPE NUMERIC USING (ask_price4::NUMERIC / 10000),
    ALTER COLUMN ask_price5          TYPE NUMERIC USING (ask_price5::NUMERIC / 10000);

ALTER TABLE ohlcv_candles
    ALTER COLUMN open   TYPE NUMERIC USING (open::NUMERIC / 10000),
    ALTER COLUMN high   TYPE NUMERIC USING (high::NUMERIC / 10000),
    ALTER COLUMN low    TYPE NUMERIC USING (low::NUMERIC / 10000),
    ALTER COLUMN close  TYPE NUMERIC USING (close::NUMERIC / 10000),
    ALTER COLUMN volume TYPE NUMERIC USING (volume::NUMERIC / 10000);
