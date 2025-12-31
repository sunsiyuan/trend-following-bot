# Documentation Versions

## v12 - Param sweep main-process runs index writes

- backtest: add `write_run_index` toggle so workers can return run index records without writing runs.jsonl.
- param_sweep: use `as_completed` aggregation to upsert runs.jsonl only in the main process (workers never write the index).
- tests: add parallel sweep coverage to assert runs.jsonl includes all run IDs and index writes happen in the main aggregation loop.
- docs: document main-process runs.jsonl writes during parallel sweeps.

## v11 - Backtest index self-heal on skipped runs

- backtest_store: add runs.jsonl upsert with malformed-line tolerance and atomic write.
- backtest: when run_dir matches existing fingerprints, skip recomputation but upsert the run index.
- tests: cover runs.jsonl create/append/replace and malformed line handling.
- docs: document skipped-run index healing in PROJECT_MAP.md and STRATEGY_CONTEXT.md.

## v10 - Parameter sweep parallel execution

- param_sweep: add `--workers` argument for multi-process parallel execution to accelerate large-scale parameter sweeps.
- param_sweep: use `ProcessPoolExecutor` for parallel backtest execution while maintaining deterministic run_id generation.
- docs: update PROJECT_MAP.md to document parallel execution support.

## v9 - Parameter sweep tool

- param_sweep: add `bot/param_sweep.py` for batch parameter scanning from JSON files with Cartesian product support.
- param_sweep: support explicit format (base + sweep with dot-notation paths) and implicit format (list values in nested dicts).
- param_sweep: automatically merge with config defaults and generate deterministic run_id per parameter combination.
- docs: update PROJECT_MAP.md to document param_sweep entry point and parameter file formats.

## v8 - Trend existence deadband scaling
- strategy: add fast_state deadband confidence scaling to shrink exposure near the trend MA without changing direction; includes decision diagnostics for deadband_conf/deadband_active and strategy version bump.
- backtest params: include trend_existence.fast_state_deadband_pct in param hashing and summary layer snapshots for reproducibility.
- docs: document deadband formulas and position sizing integration.

## Released - Various versions
- rank_runs: allow comma-separated or list `param_hash_prefix` inputs and normalize filters to a list for stable hashing.
- rank_runs: drop `quarterly_stats.csv` reads so profile pct_days_* metrics always follow summary.json.
- backtest: include risk sizing params in `BacktestParams`/`param_hash` and persist snapshots in summaries for reproducibility; existing run outputs remain unchanged, but new runs will produce new hashes when these params differ.
- backtest: add execution_policy trade intent gate with min_trade_notional_pct to reduce small-delta noise, track NOOP_SMALL_DELTA counts, and include trade_intent in trades logs for reproducibility.

# v8 - Param sweep strict params + effective_params snapshots
- Enforce strict parameter materialization (`input_params` → `effective_params`) with fail-fast on unapplied keys.
- Strategy/backtest now read parameters from the materialized params object (no silent config overrides), and long-only runs enforce `pct_days_short == 0`.
- `config_snapshot.json`/`summary_all.json` persist `input_params`, `effective_params`, and `unapplied_params` for sweep auditability.

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
