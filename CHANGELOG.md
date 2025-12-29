# Changelog

## Unreleased
- rank_runs: allow comma-separated or list `param_hash_prefix` inputs and normalize filters to a list for stable hashing.
- rank_runs: drop `quarterly_stats.csv` reads so profile pct_days_* metrics always follow summary.json.
- backtest: include risk sizing params in `BacktestParams`/`param_hash` and persist snapshots in summaries for reproducibility; existing run outputs remain unchanged, but new runs will produce new hashes when these params differ.
