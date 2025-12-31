# Trend Following Bot

A minimal, **indicator-pluggable** MA/Donchian strategy template for Hyperliquid candles.

## What you get

- `config.py` — all knobs in one place (symbols, timeframes, indicator choice per layer)
- `data_client.py` — fetch + cache Hyperliquid candleSnapshot to JSONL
- `indicators.py` — MA + Donchian (pure functions)
- `strategy.py` — decision logic (shared by live + backtest)
- `backtest.py` — downloads data if missing, runs backtest, writes results
- `main.py` — live runner stub (prints latest decision)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Backtest

### Single Run

Example:

```bash
python -m bot.backtest --start 2024-01-01 --end 2025-12-20 --symbols BTC,ETH
```

### Parameter Sweep

Run multiple parameter combinations from a JSON file with automatic Cartesian product:

```bash
# Sequential execution (default)
python -m bot.param_sweep --params param.json --start 2024-01-01 --end 2025-12-20 --symbols ETH

# Parallel execution (8 workers)
python -m bot.param_sweep --params param.json --start 2024-01-01 --end 2025-12-20 --symbols ETH --workers 8
```

Parameter file formats:

**Explicit format** (base + sweep):
```json
{
  "base": {
    "trend_existence": {"ma_type": "ema", "slope_k": 3},
    "trend_quality": {"neutral_band_pct": 0.025}
  },
  "sweep": {
    "trend_existence.window": [18, 20, 25, 30],
    "trend_quality.window": [28, 32, 40]
  }
}
```

**Implicit format** (list values in nested dicts):
```json
{
  "trend_existence": {
    "window": [18, 20, 25, 30],
    "ma_type": "ema"
  },
  "trend_quality": {
    "window": [28, 32, 40]
  }
}
```

Outputs:

```
data/backtest_result/{run_id}/{symbol}/summary.json
data/backtest_result/{run_id}/{symbol}/equity_by_day.csv
data/backtest_result/{run_id}/{symbol}/trades.jsonl
```

`summary.json` includes core performance metrics such as total return, max drawdown, Sharpe,
Ulcer Index (RMS of percent drawdowns from peak), and Ulcer Performance Index (total return
divided by ulcer index).

Market data cache (JSONL):

```
data/market_data/{symbol}/{interval}.jsonl
```

## Strategy idea (3 layers)

Default layers in `config.py`:

- **Trend existence** (1D): Donchian breakout (window N) → LONG/SHORT/NO_TREND
- **Trend quality** (1D): MA window M + neutral band → RISK_ON / RISK_NEUTRAL / RISK_OFF
- **Execution rhythm** (4H): MA window K + cooldown + max step size → smooth rebalancing

All indicator choices are swappable via `config.py`.

## Notes

- Backtest is a simple cash + position simulator (no leverage constraints, no funding, no slippage).
- Fee is parameterized via `TAKER_FEE_BPS` in `config.py` (default 0).
- Live runner (`main.py`) is stateless by default; persist state if you want cooldown behavior across runs.
