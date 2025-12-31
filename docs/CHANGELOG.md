# Documentation Versions

## v8 - Trend existence deadband scaling
- strategy: add fast_state deadband confidence scaling to shrink exposure near the trend MA without changing direction; includes decision diagnostics for deadband_conf/deadband_active and strategy version bump.
- backtest params: include trend_existence.fast_state_deadband_pct in param hashing and summary layer snapshots for reproducibility.
- docs: document deadband formulas and position sizing integration.

## Released - Various versions
- rank_runs: allow comma-separated or list `param_hash_prefix` inputs and normalize filters to a list for stable hashing.
- rank_runs: drop `quarterly_stats.csv` reads so profile pct_days_* metrics always follow summary.json.
- backtest: include risk sizing params in `BacktestParams`/`param_hash` and persist snapshots in summaries for reproducibility; existing run outputs remain unchanged, but new runs will produce new hashes when these params differ.
- backtest: add execution_policy trade intent gate with min_trade_notional_pct to reduce small-delta noise, track NOOP_SMALL_DELTA counts, and include trade_intent in trades logs for reproducibility.

## v7 - Final score UI sensitivity adjustment
- Update rank `final` formula to `E * ((mdd_score / UI_eff) ** gamma)` to reduce UI sensitivity without adding parameters.

## v6 - Drawdown scoring soft/hard guards
- Split drawdown guardrails into `mdd_pass_guard=-0.30` and `mdd_hard_guard=-0.60` with linear decay in between.
- Documented the updated drawdown scoring and guard semantics.

## v5 - Rank final always computed
- Ensure rank scoring always computes `final` from the formula, independent of `mdd_pass`.
- Emit `base` (`E/UI_eff`) and `mdd_score` for rank diagnostics.
- Document the evaluation-layer scoring guardrail behavior and diagnostics fields.

## v4 - rank run implement
- Added `bot/rank_runs.py` for ranking backtest runs, plus key metric contract documentation.
- Documented ranking outputs and evaluation-layer contract updates.
- Switched rank output from Markdown to CSV (`rank_results.csv`) for easier table analysis.
- Prefixed rank output directories with UTC timestamps to improve readability and sorting.

## v3 - Quarterly stats annualized return column

**CR (prompt focus)**
- Add `return_annualized` to `quarterly_stats.csv` with compound annualization based on the quarter’s start/end equity dates.

**Summary of changes**
- Updated `bot/quarterly_stats.py` to compute `return_annualized` and place it between `pnl` and `sharpe_annualized`.
- Documented the new quarterly stats metric in `docs/PROJECT_MAP.md` and `docs/STRATEGY_CONTEXT.md`.

## v2 - Backtest experiment contract

**CR (prompt focus)**
- Add a callable backtest entrypoint with centralized params, deterministic `param_hash`/`data_fingerprint`, and JSONL run records, while keeping CLI behavior.

**Summary of changes**
- Added `bot/backtest_params.py` and `bot/backtest_store.py` for parameter hashing, data manifests, and JSONL append.
- Updated `bot/backtest.py` to expose `run_backtest`, enforce start-inclusive/end-exclusive slicing, and generate deterministic run IDs.
- Refreshed `docs/PROJECT_MAP.md` and `docs/STRATEGY_CONTEXT.md` to describe the new contract and modules.

## v1 - Initial project mapping and strategy context

- Baseline documentation in `PROJECT_MAP.md` and `STRATEGY_CONTEXT.md`.
