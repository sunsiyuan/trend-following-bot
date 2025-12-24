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
    if np.any(arr <= 0):
        raise ValueError("log_slope requires all values to be positive")

    if is_series:
        log_ser = np.log(series)
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
    upper = high.rolling(window, min_periods=window).max().shift(1)
    lower = low.rolling(window, min_periods=window).min().shift(1)
    return upper, lower
