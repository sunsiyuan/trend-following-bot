# Documentation Versions

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
