# Strategy Semantics Audit (Quant + State Machine)

## 1. Executive Summary
- Strategy decisions are centralized in `bot/strategy.py` (shared by backtest and live), which already provides a strong semantic single source for trend/risk/target logic.
- Indicator computation is mostly centralized via `bot/indicators.py` + `bot/strategy.prepare_features_*`, with no detected duplicated math outside those helpers.
- Live runner (`bot/main.py`) uses the same indicator pipeline but resets state each run, which bypasses cooldown/flip blocks and can diverge from backtest behavior unless state is persisted.
- Reason strings and event schema fields are ad hoc across `bot/strategy.py` and `bot/backtest.py`, raising drift risk for diagnostics and downstream analytics.
- Backtest win-rate is always 0.0 because `metrics.trade_win_rate` looks for `realized_pnl` while `bot/backtest.py` writes `realized_pnl_usdc`.
- Bar alignment is consistent (index is `close_ts` from Hyperliquid candles) and look-ahead is generally avoided (Donchian uses `shift(1)`), but no central “bar alignment” helper exists.
- Range regime logic is explicit and deterministic; neutral band and target_frac mapping are centralized in config.
- No volatility indicator exists; slope uses log-difference of the trend MA.
- Diagnostics counts align with trades.jsonl fields, but some reason checks are substring-based and not centrally enumerated.

## 2. Semantic Map (concept ➜ implementation ➜ callers)

| Semantic concept | Authoritative implementation | Callers / usage |
| --- | --- | --- |
| Price series (close) | `bot/data_client.load_klines_df_from_cache` (`close` from Hyperliquid candleSnapshot) | `bot/backtest.py` (indicator prep), `bot/main.py` (latest decision) |
| Returns series (simple pct) | `bot/metrics.daily_returns_from_equity` | `bot/metrics.sharpe_ratio_from_daily_returns` |
| SMA/EMA | `bot/indicators.moving_average` | `bot/strategy.prepare_features_1d`, `bot/strategy.prepare_features_exec` |
| Log slope | `bot/indicators.log_slope` | `bot/strategy.prepare_features_1d` (trend_log_slope) |
| Donchian channel | `bot/indicators.donchian` | `bot/strategy.prepare_features_1d` (trend_upper/lower) |
| Trend direction (`raw_dir`) | `bot/strategy.decide_trend_existence` | `bot/strategy.decide`; recorded in `trades.jsonl` |
| Risk regime (`RISK_*`) | `bot/strategy.decide_risk_mode` | `bot/strategy.decide`; recorded in `trades.jsonl` |
| Range detection | `bot/strategy.is_range_regime` | `bot/strategy.decide`; reason codes `range_*` |
| Execution MA gate | `bot/strategy.execution_gate_mode` | `bot/strategy.decide` |
| Target fraction mapping | `bot/strategy.compute_desired_target_frac` + `bot/config.MAX_POSITION_FRAC` | `bot/strategy.decide` |
| Target smoothing | `bot/strategy.smooth_target` | `bot/strategy.decide` |
| Flip cooldown | `bot/strategy.decide` state fields (`flip_block_until_exec_bar_idx`) | `bot/backtest.py` state; `bot/main.py` (stateless demo) |
| Decision/event schema | `bot/strategy.make_decision` + `bot/backtest.py` trade record assembly | `bot/backtest.py` writes `trades.jsonl` |

## 3. Findings (HIGH / MED / LOW)

### HIGH
1) **Win-rate is always 0 because PnL field name mismatch**
   - **What**: `metrics.trade_win_rate` looks for `realized_pnl`, but backtest writes `realized_pnl_usdc`.
   - **Where**: `bot/metrics.py:trade_win_rate`, `bot/backtest.py` trade record field `realized_pnl_usdc`.
   - **Symbol(s)**: `BTCTEST` (backtest run), `BTC/ETH/SOL` (config defaults in `bot/config.py`).
   - **Impact**: Reported win_rate is misleading (semantic drift between metric and event schema).
   - **Convergence**: Update win-rate to consume `realized_pnl_usdc` (with backward-compatible fallback).

### MED
2) **Live runner resets state, bypassing cooldown/flip blocks**
   - **What**: `bot/main.py` instantiates a new `StrategyState` each call, so `last_exec_bar_idx` and flip cooldown are never persisted.
   - **Where**: `bot/main.py` (`state = strat.StrategyState()`), cooldown logic in `bot/strategy.decide`.
   - **Symbol(s)**: `BTC/ETH/SOL` (`bot/config.py`), `BTCTEST` if live-like is simulated.
   - **Impact**: Execution gating and flip-block semantics differ between backtest and live usage.
   - **Convergence**: Persist `StrategyState` between runs or store in a state file/DB.

3) **Reason codes are ad hoc strings and not centrally enumerated**
   - **What**: Reasons are hard-coded in `bot/strategy.py` and referenced via substring checks in `bot/backtest.py` diagnostics.
   - **Where**: `bot/strategy.decide` reason strings; `bot/backtest.py` uses `reason.startswith("range_")` and substring checks.
   - **Symbol(s)**: `BTCTEST`, `BTC/ETH/SOL` (reason strings appear in all outputs).
   - **Impact**: Risk of drift/typos and unstable analytics.
   - **Convergence**: Centralize reason codes (enum/const) and expose in a single module.

### LOW
4) **No explicit canonical event schema for trades/signals**
   - **What**: `trades.jsonl` schema is built inline in `bot/backtest.py`, while live runner outputs only the decision dict.
   - **Where**: `bot/backtest.py` trade record assembly; `bot/main.py` output.
   - **Symbol(s)**: `BTCTEST`, `BTC/ETH/SOL`.
   - **Impact**: Downstream consumers may diverge on fields/definitions.
   - **Convergence**: Define an event schema module with versioning (backward-compatible). Provide a single serializer.

## 4. Specific Audit Questions (with evidence)

1) **Indicators computed differently between backtest and live?**\n
   - **Answer**: No. Both paths call `bot/strategy.prepare_features_1d` and `bot/strategy.prepare_features_exec` which in turn use `bot/indicators.moving_average`/`log_slope`/`donchian`.\n
   - **Evidence**: `bot/backtest.py` (`df_1d_feat = strat.prepare_features_1d(...)`; `df_ex_feat = strat.prepare_features_exec(...)`), `bot/main.py` (same calls).\n
   - **Symbols**: `BTCTEST` (backtest), `BTC/ETH/SOL` (live default config in `bot/config.py`).

2) **Bar boundaries/timezones consistent across timeframes?**\n
   - **Answer**: Yes. Both use Hyperliquid candleSnapshot `close_ts` in ms (UTC) as the DataFrame index; no resampling logic exists.\n
   - **Evidence**: `bot/data_client.load_klines_df_from_cache` and `bot/main.py` `candles_to_df` index `close_ts` and record `open_ts/close_ts` directly from API.\n
   - **Symbols**: `BTCTEST`, `BTC/ETH/SOL`.

3) **Is `raw_dir` computed once and reused?**\n
   - **Answer**: Yes. `raw_dir` is computed once in `bot/strategy.decide` via `decide_trend_existence` and passed through the decision payload.\n
   - **Evidence**: `bot/strategy.decide` and `bot/strategy.decide_trend_existence`.\n
   - **Symbols**: `BTCTEST`, `BTC/ETH/SOL`.

4) **Is risk regime computed once and reused?**\n
   - **Answer**: Yes. `decide_risk_mode` is called once per decision and stored in the decision dict.\n
   - **Evidence**: `bot/strategy.decide`, `bot/strategy.decide_risk_mode`.\n
   - **Symbols**: `BTCTEST`, `BTC/ETH/SOL`.

5) **Are execution gates applied consistently and in a single order?**\n
   - **Answer**: The gate order is centralized in `bot/strategy.decide` (Stage E). However, live uses a fresh `StrategyState` per call, which skips cooldown/flip-block persistence.\n
   - **Evidence**: `bot/strategy.decide` (stage order), `bot/main.py` (`state = strat.StrategyState()`), `bot/backtest.py` (state persisted in loop).\n
   - **Symbols**: `BTCTEST`, `BTC/ETH/SOL`.

6) **Are hold reasons mutually exclusive and collectively exhaustive?**\n
   - **Answer**: In `bot/strategy.decide`, each decision returns a single reason and exits early, so reasons are mutually exclusive at the decision level. Diagnostics in `bot/backtest.py` use substring checks, which can overlap if new reason strings are added.\n
   - **Evidence**: `bot/strategy.decide` early returns with a single `reason`; `bot/backtest.py` uses `reason.startswith(\"range_\")` and substring checks.\n
   - **Symbols**: `BTCTEST`, `BTC/ETH/SOL`.

7) **Are `target_frac` transitions deterministic?**\n
   - **Answer**: Yes. The strategy uses `smooth_target` and `eps = 1e-9` for `already_at_target` checks. Diagnostics use `eps = 1e-6` only for reporting histograms.\n
   - **Evidence**: `bot/strategy.decide` (`eps = 1e-9`, `smooth_target`); `bot/backtest.py` (`eps = 1e-6` in `compute_diagnostic_counts`).\n
   - **Symbols**: `BTCTEST`, `BTC/ETH/SOL`.

8) **Are trade/rebalance lifecycle semantics truthful?**\n
   - **Answer**: `trade_count == rebalance_count` by design (rebalance events are treated as trades). The win-rate mismatch was due to a field-name bug; it now uses `realized_pnl_usdc`.\n
   - **Evidence**: `bot/backtest.py` (`trade_count` set to `rebalance_count` in `compute_trade_decision_counts`), `bot/metrics.trade_win_rate` and trade record field `realized_pnl_usdc`.\n
   - **Symbols**: `BTCTEST`, `BTC/ETH/SOL`.

9) **Do diagnostics counters match actual logged events?**\n
   - **Answer**: Yes for core counts (`decision_count == rebalance_count + hold_count`), and `raw_dir_days_covered` matches `days_total` in the BTCTEST run. Reason attribution uses string heuristics and may drift if reason codes change.\n
   - **Evidence**: `bot/backtest.py` `compute_trade_decision_counts`, `compute_diagnostic_counts`; BTCTEST run `summary.json` in `data/backtest_result/20251224T040043Z/BTCTEST/summary.json`.\n
   - **Symbols**: `BTCTEST`, `BTC/ETH/SOL`.

## 5. Semantic Canon (proposed single-source-of-truth definitions)

### A) Indicator canon
- **Module**: `bot/indicators.py` for pure computations (SMA/EMA, log_slope, donchian).
- **Feature assembly**: `bot/strategy.prepare_features_1d` and `bot/strategy.prepare_features_exec`.
- **Alignment/warmup**:
  - Rolling/EMA uses `min_periods=window`, so NaNs until window fill.
  - `log_slope` requires positive inputs and uses `shift(k)` for series alignment.
  - Donchian uses `shift(1)` to avoid look-ahead.

### B) State canon
- **Regime & target**: `bot/strategy.decide_trend_existence`, `decide_risk_mode`, `compute_desired_target_frac`.
- **Execution gate order**: Stages A-F in `bot/strategy.decide` (fetch rows ➜ trend/range ➜ risk/target ➜ flip cooldown ➜ gate ➜ smooth target).
- **Tolerance**: `eps = 1e-9` for “already_at_target”.
- **Cooldown**: `StrategyState.last_exec_bar_idx`, `flip_block_until_exec_bar_idx`.

### C) Event schema canon
- **Decision**: `bot/strategy.make_decision` defines required keys (use as schema baseline).
- **Trade event**: `bot/backtest.py` record fields (`ts_ms`, `action`, `reason`, `position_*`, `realized_pnl_usdc`, etc.).
- **Recommendation**: central module (e.g., `bot/event_schema.py`) with versioned serializer.

## 6. Refactor Plan (minimal, safe)

### Implemented (safe convergence)
- **Fix win-rate field mismatch**: update `metrics.trade_win_rate` to prefer `realized_pnl_usdc` with fallback to `realized_pnl`.
  - **Why**: Aligns analytics with event schema and eliminates misleading 0% win-rate.
  - **Tests**: add unit tests for win-rate calculation.

### Proposed (not implemented)
- **Reason code enum**: centralize reason strings into `bot/reasons.py` and import from `bot/strategy.py` and `bot/backtest.py` diagnostics.
- **State persistence in live**: add a lightweight state store (JSON/SQLite) to keep `StrategyState` across `bot/main.py` runs.
- **Event schema module**: unify `trades.jsonl` schema and decision schema in a versioned helper to prevent drift.

## 7. Test Plan
- **Existing**: `tests/test_indicators.py` (SMA/EMA/log_slope).
- **Added**: `tests/test_metrics.py` to verify `trade_win_rate` uses `realized_pnl_usdc` and fallback.

## 8. Backtest Validation
- **Command**: `python -m bot.backtest --start 2024-01-01 --end 2025-12-20 --symbols BTCTEST`
- **Run**: `data/backtest_result/20251224T040043Z/BTCTEST/summary.json`
- **Key result**: `win_rate` now reports `0.012268518518518519` (no change to decisions; only metric interpretation corrected).

## Patch List (changes made)
- `bot/metrics.py`: `trade_win_rate` now reads `realized_pnl_usdc` first (fallback to `realized_pnl`).
- `tests/test_metrics.py`: new unit tests for win-rate semantics.
- `AUDIT_STRATEGY_SEMANTICS.md`: audit report and convergence plan.
