# internal/strategy

Pluggable trading strategies evaluated against finalized candles.

## Where it sits in the pipeline

```
ticks → ingest → candle engine → IndicatorManager.processCandle
                                        │
                                        ├─ persist indicators (batch writer)
                                        ├─ broadcast indicators (WebSocket)
                                        └─ strategy.Engine.OnCandle(frame)   ← here
                                                  │
                                                  ├─ protective exits (stop / target / square-off)
                                                  ├─ strategy evaluation
                                                  ├─ RiskManager sizing + limits
                                                  └─ Sink → OrderExecutor → Broker
```

The engine runs **after** persistence and the WebSocket broadcast, so a slow or
misbehaving strategy can never delay market data. It sits on the
candle-finalization path (once per interval per instrument), not the per-tick
path.

## Adding a strategy

One new file. Nothing else changes.

```go
package strategy

func init() {
    Register("my_breakout", func() Strategy { return &MyBreakout{} })
}

type MyBreakout struct {
    base
    threshold float64
}

func (s *MyBreakout) Name() string { return "my_breakout" }

func (s *MyBreakout) Init(p Params) error {
    s.threshold = p.Float("threshold", 1.5)
    if s.threshold <= 0 {
        return fmt.Errorf("threshold must be positive, got %v", s.threshold)
    }
    s.base = base{warmup: p.Int("warmup", 5), requires: HasATR | HasRSI}
    return nil
}

func (s *MyBreakout) OnCandle(f *Frame, pos *Position) Signal {
    if pos.IsOpen() {
        return None
    }
    if f.Close > f.Open+s.threshold*f.Ind.ATR {
        return Signal{Action: ActionEntry, Side: SideLong, Reason: ReasonTrendUp, Price: f.Close}
    }
    return None
}
```

Then enable it in `configs/strategy.yaml`:

```yaml
engine:
  enabled: true
  strategies:
    - name: my_breakout
      enabled: true
      intervals: ["5m"]
      instruments: []   # empty = every instrument
      params:
        threshold: 1.5
        warmup: 5
```

An unregistered or misconfigured strategy fails at **startup**, not at 09:15 on
a live feed.

## Contract for `OnCandle`

1. **Do not allocate, block, log, or do I/O.** It is on the data path.
2. **Be deterministic** given `(Frame, Position)` and the instance's own state.
3. **No locking needed.** Each `(instrument, interval, strategy)` gets its own
   instance and is never evaluated concurrently.

`Requires()` declares which indicators the strategy needs; the engine skips
evaluation entirely until all of them are present, so strategies never handle
warm-up themselves. `Present` is a bitmask because a warming indicator and a
genuinely-zero indicator are different things.

## What the engine owns, not the strategy

- **Stop loss / take profit** — checked before strategy logic every bar. Within
  one candle we cannot know which level was touched first, so the engine
  assumes the adverse fill.
- **Session square-off** — open positions are force-closed at `square_off`.
- **Position sizing and risk limits** — max trades/day, max open positions,
  daily loss cap, short permission. Exits are never blocked by a risk limit.
- **Panic isolation** — a panicking strategy is recovered and counted in
  `strategy_panics_total`; the pipeline survives.
- **No pyramiding** — an entry signal while already exposed is ignored.

## Safety defaults

- The engine is **disabled** by default (`engine.enabled: false`).
- Shorting is **disabled** by default (`risk.allow_short: false`).
- The executor uses `PaperBroker`, which logs orders and places nothing.
  Going live is an explicit wiring change in `app.initStrategyEngine`.

## Metrics

| Metric | Meaning |
| --- | --- |
| `strategy_evaluations_total` | candles handed to the engine |
| `strategy_signals_total{strategy,action,side}` | signals before risk checks |
| `strategy_orders_total{strategy,action,side}` | orders after risk sizing |
| `strategy_rejects_total{reason}` | signals refused by risk |
| `strategy_panics_total` | recovered strategy panics — **alert on any value > 0** |
| `strategy_order_drops_total` | orders lost to a full queue |
| `strategy_evaluation_duration_microseconds` | engine time per candle |
| `strategy_open_positions`, `strategy_realized_pnl` | current risk state |

## Performance

Measured on an AMD Ryzen 5 3500 (`go test -bench . -benchmem`):

```
BenchmarkEngineOnCandleNoSignal-6   17831191   67.41 ns/op   0 B/op   0 allocs/op
BenchmarkIntradayOnCandle-6        241450440    4.97 ns/op   0 B/op   0 allocs/op
```

The hot path is allocation-free by construction: `Frame` and
`IndicatorSnapshot` are flat value types with no maps, slices, or interfaces,
and signal reasons are `uint8` enums rather than strings.
