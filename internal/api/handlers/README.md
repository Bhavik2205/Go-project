# Handler Package Layout

Each handler package should stay thin:

- Decode request.
- Validate request.
- Call application service.
- Encode response through a shared response helper.
- Never contain broker, database, encryption, strategy, or job orchestration logic directly.

Planned packages:

- `auth`
- `broker`
- `settings`
- `runtime`
- `health`
- `market`
- `watchlist`
- `orders`
- `positions`
- `strategies`
- `models`
- `sentiment`
- `backtest`
- `notifications`

