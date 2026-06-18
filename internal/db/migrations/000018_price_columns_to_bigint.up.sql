-- Migration 000018: Convert price columns in market_data and ohlcv_candles from NUMERIC to BIGINT.
-- Prices are now stored as scaled integers (multiply by 10000 before write, divide on read).

ALTER TABLE market_data
    ALTER COLUMN last_price          TYPE BIGINT USING (last_price * 10000)::BIGINT,
    ALTER COLUMN average_trade_price TYPE BIGINT USING (average_trade_price * 10000)::BIGINT,
    ALTER COLUMN net_change          TYPE BIGINT USING (net_change * 10000)::BIGINT,
    ALTER COLUMN open                TYPE BIGINT USING (open * 10000)::BIGINT,
    ALTER COLUMN high                TYPE BIGINT USING (high * 10000)::BIGINT,
    ALTER COLUMN low                 TYPE BIGINT USING (low * 10000)::BIGINT,
    ALTER COLUMN close               TYPE BIGINT USING (close * 10000)::BIGINT,
    ALTER COLUMN bid_price1          TYPE BIGINT USING (bid_price1 * 10000)::BIGINT,
    ALTER COLUMN bid_price2          TYPE BIGINT USING (bid_price2 * 10000)::BIGINT,
    ALTER COLUMN bid_price3          TYPE BIGINT USING (bid_price3 * 10000)::BIGINT,
    ALTER COLUMN bid_price4          TYPE BIGINT USING (bid_price4 * 10000)::BIGINT,
    ALTER COLUMN bid_price5          TYPE BIGINT USING (bid_price5 * 10000)::BIGINT,
    ALTER COLUMN ask_price1          TYPE BIGINT USING (ask_price1 * 10000)::BIGINT,
    ALTER COLUMN ask_price2          TYPE BIGINT USING (ask_price2 * 10000)::BIGINT,
    ALTER COLUMN ask_price3          TYPE BIGINT USING (ask_price3 * 10000)::BIGINT,
    ALTER COLUMN ask_price4          TYPE BIGINT USING (ask_price4 * 10000)::BIGINT,
    ALTER COLUMN ask_price5          TYPE BIGINT USING (ask_price5 * 10000)::BIGINT;

ALTER TABLE ohlcv_candles
    ALTER COLUMN open   TYPE BIGINT USING (open * 10000)::BIGINT,
    ALTER COLUMN high   TYPE BIGINT USING (high * 10000)::BIGINT,
    ALTER COLUMN low    TYPE BIGINT USING (low * 10000)::BIGINT,
    ALTER COLUMN close  TYPE BIGINT USING (close * 10000)::BIGINT,
    ALTER COLUMN volume TYPE BIGINT USING (volume * 10000)::BIGINT;
