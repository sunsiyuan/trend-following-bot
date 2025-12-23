"""
main.py

Live runner (minimal).
- Fetch latest candles for required timeframes
- Compute features
- Call strategy.decide for the latest execution bar
- Print decision JSON (integrate your notifier later)

This is intentionally simple; keep strategy/backtest as the source of truth.
"""

from __future__ import annotations

import json
import logging
from typing import Dict

import pandas as pd

from bot import config
from bot import data_client
from bot import strategy as strat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("main")

def latest_decision_for_symbol(symbol: str) -> Dict[str, object]:
    # Fetch recent candles (enough for indicators)
    trend_tf = config.TIMEFRAMES["trend"]
    exec_tf = config.TIMEFRAMES["execution"]

    # A bit more than needed for rolling windows
    trend_lookback = max(config.TREND_EXISTENCE["window"], config.TREND_QUALITY["window"]) + 20
    exec_lookback = config.EXECUTION["window"] + max(
        config.EXECUTION["build_min_step_bars"],
        config.EXECUTION["reduce_min_step_bars"],
    ) + 50

    candles_1d = data_client.fetch_latest(symbol, trend_tf, lookback_candles=trend_lookback)
    candles_ex = data_client.fetch_latest(symbol, exec_tf, lookback_candles=exec_lookback)

    # Build dfs (same schema as cache loader)
    def candles_to_df(candles):
        if not candles:
            return pd.DataFrame(columns=["open_ts","close_ts","open","high","low","close","volume","trades"]).set_index("close_ts")
        records = [{
            "open_ts": int(c["t"]),
            "close_ts": int(c["T"]),
            "open": float(c["o"]),
            "high": float(c["h"]),
            "low": float(c["l"]),
            "close": float(c["c"]),
            "volume": float(c.get("v", "0")),
            "trades": int(c.get("n", 0)),
        } for c in candles]
        df = pd.DataFrame(records).drop_duplicates(subset=["close_ts"]).sort_values("close_ts").set_index("close_ts")
        return df

    df_1d = candles_to_df(candles_1d)
    df_ex = candles_to_df(candles_ex)

    df_1d_feat = strat.prepare_features_1d(df_1d)
    df_ex_feat = strat.prepare_features_exec(df_ex)

    # Latest execution bar (close_ts)
    ts_ms = int(df_ex_feat.index.max())
    exec_bar_idx = len(df_ex_feat) - 1

    state = strat.StrategyState()  # stateless demo; persist this if you want real cooldown behavior
    decision = strat.decide(ts_ms, exec_bar_idx, df_1d_feat, df_ex_feat, state)
    decision["symbol"] = symbol
    decision["ts_ms"] = ts_ms
    return decision

def main() -> None:
    for sym in config.SYMBOLS:
        d = latest_decision_for_symbol(sym)
        print(json.dumps(d, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
