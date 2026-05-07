# Realtime Package Plan

This package will replace ad hoc WebSocket handling with a shared realtime layer.

- Authenticated WebSocket handshake.
- Client subscriptions by topic, user ID, symbol, instrument token, and interval.
- Heartbeat, ping/pong, read/write deadlines, and max message sizes.
- Backpressure handling and per-client send queues.
- Typed event envelopes for ticks, candles, indicators, heatmap, orders, positions, alerts, strategy signals, model jobs, and backtests.
- Reconnect-safe initial snapshot support through matching REST endpoints.

