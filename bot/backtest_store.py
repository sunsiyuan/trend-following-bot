from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from bot.backtest_params import stable_json


data_schema_version = 1


def timeframe_to_seconds(tf: str) -> int:
    if not tf:
        raise ValueError("Empty timeframe")
    unit = tf[-1]
    value = int(tf[:-1])
    if unit == "s":
        return value
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 60 * 60
    if unit == "d":
        return value * 60 * 60 * 24
    raise ValueError(f"Unsupported timeframe: {tf}")


def _get_ts(record: Dict[str, Any], keys: Iterable[str]) -> int | None:
    for key in keys:
        if key in record and record[key] is not None:
            try:
                return int(record[key])
            except (TypeError, ValueError):
                return None
    return None


def slice_bars(data: Any, start_ts: int, end_ts: int) -> Any:
    if isinstance(data, pd.DataFrame):
        if data.empty:
            return data.copy()
        if "open_ts" in data.columns:
            open_ts = data["open_ts"].astype(int)
        else:
            open_ts = data.index.astype(int)
        if "close_ts" in data.columns:
            close_ts = data["close_ts"].astype(int)
        else:
            close_ts = data.index.astype(int)
        mask = (open_ts >= int(start_ts)) & (close_ts < int(end_ts))
        return data.loc[mask].copy()

    if isinstance(data, list):
        sliced: List[Dict[str, Any]] = []
        for row in data:
            if not isinstance(row, dict):
                continue
            open_ts = _get_ts(row, ["open_ts", "t", "openTime", "open_time"])
            close_ts = _get_ts(row, ["close_ts", "T", "closeTime", "close_time"])
            if open_ts is None or close_ts is None:
                continue
            if open_ts >= int(start_ts) and close_ts < int(end_ts):
                sliced.append(row)
        return sliced

    raise TypeError(f"Unsupported data type for slice_bars: {type(data)}")


def _extract_bounds(data: Any) -> tuple[int | None, int | None, int]:
    if isinstance(data, pd.DataFrame):
        row_count = int(len(data))
        if row_count == 0:
            return None, None, 0
        if "open_ts" in data.columns:
            actual_first_ts = int(pd.to_numeric(data["open_ts"], errors="coerce").min())
        else:
            actual_first_ts = int(data.index.min())
        if "close_ts" in data.columns:
            actual_last_ts = int(pd.to_numeric(data["close_ts"], errors="coerce").max())
        else:
            actual_last_ts = int(data.index.max())
        return actual_first_ts, actual_last_ts, row_count

    if isinstance(data, list):
        if not data:
            return None, None, 0
        opens: List[int] = []
        closes: List[int] = []
        for row in data:
            if not isinstance(row, dict):
                continue
            open_ts = _get_ts(row, ["open_ts", "t", "openTime", "open_time"])
            close_ts = _get_ts(row, ["close_ts", "T", "closeTime", "close_time"])
            if open_ts is None or close_ts is None:
                continue
            opens.append(int(open_ts))
            closes.append(int(close_ts))
        if not opens or not closes:
            return None, None, 0
        return min(opens), max(closes), len(opens)

    return None, None, 0


def build_data_manifest(
    tf: str,
    requested_start_ts: int,
    requested_end_ts: int,
    sliced_data: Any,
) -> Dict[str, Any]:
    actual_first_ts, actual_last_ts, row_count = _extract_bounds(sliced_data)
    if row_count == 0 or actual_first_ts is None or actual_last_ts is None:
        return {
            "tf": tf,
            "requested_start_ts": int(requested_start_ts),
            "requested_end_ts": int(requested_end_ts),
            "actual_first_ts": None,
            "actual_last_ts": None,
            "row_count": 0,
            "expected_row_count": 0,
        }

    tf_seconds = timeframe_to_seconds(tf)
    requested_start = int(requested_start_ts)
    requested_end = int(requested_end_ts)
    actual_start = max(requested_start, actual_first_ts)
    actual_end = min(requested_end, actual_last_ts)
    if actual_end <= actual_start:
        expected_row_count = 0
    else:
        expected_row_count = int(
            math.ceil((actual_end - actual_start) / (tf_seconds * 1000))
        )
    return {
        "tf": tf,
        "requested_start_ts": requested_start,
        "requested_end_ts": requested_end,
        "actual_first_ts": int(actual_first_ts),
        "actual_last_ts": int(actual_last_ts),
        "row_count": int(row_count),
        "expected_row_count": int(expected_row_count),
    }


def calc_data_fingerprint(manifest_by_tf_list: List[Dict[str, Any]]) -> str:
    payload = stable_json(manifest_by_tf_list).encode("utf-8")
    return sha256(payload).hexdigest()


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = stable_json(record)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
