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

Example:

```bash
python -m hypervault.backtest --start 2024-01-01 --end 2025-12-20 --symbols BTC,ETH
```

Outputs:

```
data/backtest_result/{run_id}/{symbol}/summary.json
data/backtest_result/{run_id}/{symbol}/equity_by_day.csv
data/backtest_result/{run_id}/{symbol}/trades.jsonl
```

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
