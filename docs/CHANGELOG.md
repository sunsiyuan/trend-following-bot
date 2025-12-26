# Documentation Versions

## v2 - Backtest experiment contract

**CR (prompt focus)**
- Add a callable backtest entrypoint with centralized params, deterministic `param_hash`/`data_fingerprint`, and JSONL run records, while keeping CLI behavior.

**Summary of changes**
- Added `bot/backtest_params.py` and `bot/backtest_store.py` for parameter hashing, data manifests, and JSONL append.
- Updated `bot/backtest.py` to expose `run_backtest`, enforce start-inclusive/end-exclusive slicing, and generate deterministic run IDs.
- Refreshed `docs/PROJECT_MAP.md` and `docs/STRATEGY_CONTEXT.md` to describe the new contract and modules.

## v1 - Initial project mapping and strategy context

- Baseline documentation in `PROJECT_MAP.md` and `STRATEGY_CONTEXT.md`.
