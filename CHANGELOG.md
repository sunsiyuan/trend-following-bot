# Changelog

## Unreleased
- strategy: add fast_state deadband+stickiness via TREND_EXISTENCE.fast_state_deadband_pct and expose fast_sign_raw/fast_sign_eff diagnostics (STRATEGY_VERSION v3).
- rank_runs: allow comma-separated or list `param_hash_prefix` inputs and normalize filters to a list for stable hashing.
- rank_runs: drop `quarterly_stats.csv` reads so profile pct_days_* metrics always follow summary.json.
- backtest: include risk sizing params in `BacktestParams`/`param_hash` and persist snapshots in summaries for reproducibility; existing run outputs remain unchanged, but new runs will produce new hashes when these params differ.
- backtest: add execution_policy trade intent gate with min_trade_notional_pct to reduce small-delta noise, track NOOP_SMALL_DELTA counts, and include trade_intent in trades logs for reproducibility.
