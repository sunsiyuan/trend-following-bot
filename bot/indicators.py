"""
indicators.py

Pure calculation helpers.
No config reads. No trading decisions. No I/O.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

def ma(close: pd.Series, window: int) -> pd.Series:
    """
    Simple moving average (SMA) over close.
    """
    return close.rolling(window, min_periods=window).mean()

def donchian(high: pd.Series, low: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
    """
    Donchian channel (upper/lower) of the PREVIOUS `window` bars.
    Use shift(1) to avoid look-ahead bias and to make breakouts triggerable.
    """
    upper = high.rolling(window, min_periods=window).max().shift(1)
    lower = low.rolling(window, min_periods=window).min().shift(1)
    return upper, lower
