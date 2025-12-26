"""
data_client.py

Responsibilities:
- Fetch candleSnapshot data from Hyperliquid /info endpoint (public market data).
- Load/save cached market data as JSONL:
    data/market_data/{symbol}/{interval}.jsonl
- Provide helpers to fetch "latest" or a historical range.

Design constraints:
- Only closed candles are used.
- The strategy/backtest should never call the network directly; only through this module.

Hyperliquid candleSnapshot:
POST https://api.hyperliquid.xyz/info
{
  "type":"candleSnapshot",
  "req": { "coin":"BTC", "interval":"4h", "startTime":..., "endTime":... }
}
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from bot import config

log = logging.getLogger(__name__)

# -----------------------------
# Timeframe utilities
# -----------------------------

_INTERVAL_TO_MS: Dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
    # "1M" is ambiguous (calendar month); for simplicity, treat as 30d if needed
    "1M": 30 * 24 * 60 * 60_000,
}

def interval_to_ms(interval: str) -> int:
    if interval not in _INTERVAL_TO_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    return _INTERVAL_TO_MS[interval]

def now_ms() -> int:
    return int(time.time() * 1000)

# -----------------------------
# Hyperliquid data constraints
# -----------------------------

def get_earliest_possible_ts_ms(symbol: str) -> Optional[int]:
    return config.HYPERLIQUID_EARLIEST_KLINES_TS_MS.get(symbol)

def compute_api_window_start_ts_ms(
    timeframe: str,
    end_ts_ms: Optional[int],
    limit: int = config.HYPERLIQUID_KLINE_MAX_LIMIT,
) -> int:
    end_ms = end_ts_ms if end_ts_ms is not None else now_ms()
    timeframe_ms = interval_to_ms(timeframe)
    return int(end_ms - limit * timeframe_ms)

def get_cache_earliest_ts_ms(symbol: str, interval: str) -> Optional[int]:
    # Scan cached jsonl for minimum open timestamp.
    path = market_data_path(symbol, interval)
    rows = read_jsonl(path)
    if not rows:
        return None
    min_open: Optional[int] = None
    for row in rows:
        if "t" not in row:
            continue
        try:
            open_ts = int(row["t"])
        except (TypeError, ValueError):
            continue
        min_open = open_ts if min_open is None else min(min_open, open_ts)
    return min_open

# -----------------------------
# Cache paths / JSONL helpers
# -----------------------------

def market_data_path(symbol: str, interval: str) -> Path:
    return Path(config.MARKET_DATA_DIR) / symbol / f"{interval}.jsonl"

def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n")

def upsert_candles_jsonl(path: Path, new_rows: List[Dict[str, Any]]) -> int:
    """
    Merge new candle rows into existing cache, dedup by open time `t`.
    Returns: number of rows after merge.
    """
    existing = read_jsonl(path)
    by_t: Dict[int, Dict[str, Any]] = {}
    for r in existing:
        if "t" in r:
            by_t[int(r["t"])] = r
    for r in new_rows:
        if "t" not in r:
            continue
        by_t[int(r["t"])] = r

    merged = [by_t[t] for t in sorted(by_t.keys())]
    write_jsonl(path, merged)
    return len(merged)

# -----------------------------
# Hyperliquid API client
# -----------------------------

class HyperliquidAPIError(RuntimeError):
    pass

def _post_info(payload: Dict[str, Any], timeout_sec: int = 20, retries: int = 3) -> Any:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = requests.post(
                config.HL_INFO_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=timeout_sec,
            )
            if r.status_code != 200:
                raise HyperliquidAPIError(f"HTTP {r.status_code}: {r.text[:200]}")
            return r.json()
        except Exception as e:
            last_err = e
            # basic backoff
            time.sleep(0.6 * (attempt + 1))
    raise HyperliquidAPIError(f"Failed after {retries} attempts: {last_err}")

def symbol_to_coin(symbol: str) -> str:
    """
    For perpetuals: coin is typically the symbol itself ("BTC", "ETH", ...).
    For spot: Hyperliquid uses special naming (see Hyperliquid docs). We keep v1 as perp-default.
    """
    if config.MARKET_TYPE == "perp":
        return symbol
    raise NotImplementedError(
        "Spot coin naming on Hyperliquid requires spotMeta mapping; "
        "v1 defaults to MARKET_TYPE='perp'."
    )

def fetch_candle_snapshot(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> List[Dict[str, Any]]:
    """
    Fetch candles (raw dict list) from Hyperliquid candleSnapshot.

    Returns items like:
      { "t": open_ms, "T": close_ms, "o":"..","h":"..","l":"..","c":"..","v":"..","n":int,"i":interval,"s":coin }
    """
    coin = symbol_to_coin(symbol)
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": int(start_ms),
            "endTime": int(end_ms),
        },
    }
    data = _post_info(payload)
    if not isinstance(data, list):
        raise HyperliquidAPIError(f"Unexpected candleSnapshot response: {type(data)}")
    # Drop any candle that is not closed yet (defensive).
    now = now_ms()
    closed = [c for c in data if int(c.get("T", 0)) <= now]
    return closed

def fetch_latest(
    symbol: str,
    interval: str,
    lookback_candles: int = 300,
) -> List[Dict[str, Any]]:
    """
    Fetch latest N candles by using a time window.
    (Hyperliquid candleSnapshot doesn't take an explicit 'limit', so we window by time.)
    """
    ms = interval_to_ms(interval)
    end = now_ms()
    start = end - lookback_candles * ms
    return fetch_candle_snapshot(symbol, interval, start, end)

def download_history_to_cache(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    window_candles: int = 4500,
) -> Path:
    """
    Download a time range and upsert into JSONL cache.
    Uses pagination by advancing startTime based on last returned candle open time.

    Notes:
    - candleSnapshot has a 'most recent candles' limit; if you request too old data, you may get partial/empty responses.
    - window_candles is kept < 5000 for safety.
    """
    path = market_data_path(symbol, interval)
    ms = interval_to_ms(interval)

    cursor = int(start_ms)
    total_new = 0

    # Paginate by time window; advance by last open time + interval.
    while cursor < end_ms:
        window_end = min(int(end_ms), cursor + window_candles * ms)
        batch = fetch_candle_snapshot(symbol, interval, cursor, window_end)
        if not batch:
            log.warning("No candles returned for %s %s in range [%s, %s]", symbol, interval, cursor, window_end)
            break
        total_new += upsert_candles_jsonl(path, batch)
        last_t = int(batch[-1]["t"])
        next_cursor = last_t + ms
        if next_cursor <= cursor:
            # safety to avoid infinite loop
            break
        cursor = next_cursor
        # gentle rate-limit
        time.sleep(0.12)

    log.info("Cached %s %s candles into %s (merged size=%s)", symbol, interval, total_new, path)
    return path

def load_klines_df_from_cache(symbol: str, interval: str) -> pd.DataFrame:
    """
    Load cached jsonl into a DataFrame with numeric columns.

    Index: close_ts (ms)
    Columns: open_ts, close_ts, open, high, low, close, volume, trades
    """
    path = market_data_path(symbol, interval)
    rows = read_jsonl(path)
    if not rows:
        return pd.DataFrame(columns=["open_ts","close_ts","open","high","low","close","volume","trades"]).set_index("close_ts")

    # Parse rows to typed schema; malformed rows are skipped.
    records = []
    for r in rows:
        try:
            records.append({
                "open_ts": int(r["t"]),
                "close_ts": int(r["T"]),
                "open": float(r["o"]),
                "high": float(r["h"]),
                "low": float(r["l"]),
                "close": float(r["c"]),
                "volume": float(r.get("v", "0")),
                "trades": int(r.get("n", 0)),
            })
        except Exception:
            # Skip malformed rows
            continue

    df = pd.DataFrame(records).drop_duplicates(subset=["close_ts"]).sort_values("close_ts")
    if df.empty:
        return df
    df = df.set_index("close_ts")
    return df

def ensure_market_data(symbol: str, interval: str, start_ms: int, end_ms: Optional[int]) -> pd.DataFrame:
    requested_start_ts_ms = int(start_ms)
    requested_end_ts_ms = int(end_ms) if end_ms is not None else now_ms()

    cache_earliest_ts_ms = get_cache_earliest_ts_ms(symbol, interval)
    earliest_fact_ts_ms = get_earliest_possible_ts_ms(symbol)
    api_window_start_ts_ms = compute_api_window_start_ts_ms(interval, requested_end_ts_ms)

    effective_start_ts_ms = requested_start_ts_ms
    skip_download = False

    # Clamp requests to earliest available data and API window limits.
    if interval == "1d" and earliest_fact_ts_ms is not None and requested_start_ts_ms < earliest_fact_ts_ms:
        effective_start_ts_ms = earliest_fact_ts_ms
        log.warning({
            "event": "REQUEST_BEFORE_ABSOLUTE_EARLIEST",
            "level": "WARNING",
            "symbol": symbol,
            "timeframe": interval,
            "requested_start_ts_ms": requested_start_ts_ms,
            "requested_end_ts_ms": requested_end_ts_ms,
            "earliest_fact_ts_ms": earliest_fact_ts_ms,
            "cache_earliest_ts_ms": cache_earliest_ts_ms,
            "effective_start_ts_ms": effective_start_ts_ms,
        })
        if cache_earliest_ts_ms is not None and cache_earliest_ts_ms <= earliest_fact_ts_ms:
            skip_download = True
            log.info({
                "event": "REQUEST_BEFORE_ABSOLUTE_EARLIEST",
                "level": "INFO",
                "symbol": symbol,
                "timeframe": interval,
                "requested_start_ts_ms": requested_start_ts_ms,
                "requested_end_ts_ms": requested_end_ts_ms,
                "earliest_fact_ts_ms": earliest_fact_ts_ms,
                "cache_earliest_ts_ms": cache_earliest_ts_ms,
                "effective_start_ts_ms": effective_start_ts_ms,
                "action": "SKIP_DOWNLOAD_ABSOLUTE_EARLIEST_REACHED",
            })

    if requested_start_ts_ms < api_window_start_ts_ms:
        effective_start_ts_ms = max(effective_start_ts_ms, api_window_start_ts_ms)
        log.warning({
            "event": "REQUEST_BEFORE_API_WINDOW",
            "level": "WARNING",
            "symbol": symbol,
            "timeframe": interval,
            "requested_start_ts_ms": requested_start_ts_ms,
            "requested_end_ts_ms": requested_end_ts_ms,
            "api_window_start_ts_ms": api_window_start_ts_ms,
            "cache_earliest_ts_ms": cache_earliest_ts_ms,
            "effective_start_ts_ms": effective_start_ts_ms,
            "limit": config.HYPERLIQUID_KLINE_MAX_LIMIT,
        })
        if cache_earliest_ts_ms is not None and cache_earliest_ts_ms <= api_window_start_ts_ms:
            skip_download = True
            log.info({
                "event": "REQUEST_BEFORE_API_WINDOW",
                "level": "INFO",
                "symbol": symbol,
                "timeframe": interval,
                "requested_start_ts_ms": requested_start_ts_ms,
                "requested_end_ts_ms": requested_end_ts_ms,
                "api_window_start_ts_ms": api_window_start_ts_ms,
                "cache_earliest_ts_ms": cache_earliest_ts_ms,
                "effective_start_ts_ms": effective_start_ts_ms,
                "limit": config.HYPERLIQUID_KLINE_MAX_LIMIT,
                "action": "SKIP_DOWNLOAD_API_WINDOW_REACHED",
            })

    df = load_klines_df_from_cache(symbol, interval)

    # Download only if cache is missing or does not cover the requested range.
    if not skip_download:
        if df.empty:
            download_history_to_cache(symbol, interval, effective_start_ts_ms, requested_end_ts_ms)
            df = load_klines_df_from_cache(symbol, interval)
        else:
            min_open = int(df["open_ts"].min())
            max_close = int(df.index.max())  # close_ts

            need_download = (min_open > effective_start_ts_ms) or (max_close < requested_end_ts_ms)
            if need_download:
                download_history_to_cache(symbol, interval, effective_start_ts_ms, requested_end_ts_ms)
                df = load_klines_df_from_cache(symbol, interval)

    return df.loc[(df.index >= requested_start_ts_ms) & (df.index <= requested_end_ts_ms)].copy()
