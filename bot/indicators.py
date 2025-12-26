"""
indicators.py

Pure calculation helpers.
No config reads. No trading decisions. No I/O.
"""

from __future__ import annotations

from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd

SeriesLike = Union[np.ndarray, pd.Series]
MAType = Literal["sma", "ema"]

def moving_average(series: SeriesLike, window: int, ma_type: MAType = "sma") -> SeriesLike:
    """
    Moving average (SMA or EMA) over a series.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if ma_type not in {"sma", "ema"}:
        raise ValueError(f"Unknown ma_type: {ma_type}")

    is_series = isinstance(series, pd.Series)
    ser = series if is_series else pd.Series(series)
    # min_periods=window enforces warmup: initial window-1 values are NaN.
    if ma_type == "sma":
        result = ser.rolling(window, min_periods=window).mean()
    else:
        result = ser.ewm(span=window, adjust=False, min_periods=window).mean()
    return result if is_series else result.to_numpy()

def log_slope(series: SeriesLike, k: int) -> SeriesLike:
    """
    Log slope: (ln(x_t) - ln(x_{t-k})) / k
    """
    if k <= 0:
        raise ValueError("k must be positive")

    is_series = isinstance(series, pd.Series)
    arr = series if is_series else np.asarray(series)
    # log requires strictly positive values; this raises if any non-positive input.
    if np.any(arr <= 0):
        raise ValueError("log_slope requires all values to be positive")

    if is_series:
        log_ser = np.log(series)
        # shift(k) introduces k-bar warmup NaNs.
        slope = (log_ser - log_ser.shift(k)) / float(k)
        return slope

    log_arr = np.log(arr)
    slope = np.full(len(log_arr), np.nan, dtype=float)
    slope[k:] = (log_arr[k:] - log_arr[:-k]) / float(k)
    return slope

def ma(close: pd.Series, window: int) -> pd.Series:
    """
    Simple moving average (SMA) over close.
    """
    return moving_average(close, window, "sma")

def donchian(high: pd.Series, low: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
    """
    Donchian channel (upper/lower) of the PREVIOUS `window` bars.
    Use shift(1) to avoid look-ahead bias and to make breakouts triggerable.
    """
    # rolling(window) with min_periods=window yields NaN until warmup completes.
    upper = high.rolling(window, min_periods=window).max().shift(1)
    lower = low.rolling(window, min_periods=window).min().shift(1)
    return upper, lower

def hlc3(high: pd.Series | None, low: pd.Series | None, close: pd.Series) -> pd.Series:
    """
    HLC3 price: (high + low + close) / 3, with fallback to close when high/low missing.
    """
    if high is None or low is None:
        return close.copy()
    hlc3_val = (high + low + close) / 3.0
    # Preserve close when high/low are missing or NaN for that row.
    valid = high.notna() & low.notna()
    return hlc3_val.where(valid, close)

def quantize_toward_zero(x: pd.Series, q: float) -> pd.Series:
    """
    Quantize toward zero: sign(x) * floor(abs(x)/q) * q
    Preserves NaNs and index alignment.
    """
    if q <= 0:
        raise ValueError("q must be positive")
    # floor(abs(x)/q) quantizes magnitude toward zero.
    values = x.to_numpy(dtype=float)
    abs_vals = np.abs(values)
    floored = np.floor(abs_vals / q) * q
    quantized = np.sign(values) * floored
    return pd.Series(quantized, index=x.index, name=x.name)
