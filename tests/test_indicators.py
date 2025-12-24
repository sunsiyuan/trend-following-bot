import math

import numpy as np
import pandas as pd

from bot.indicators import log_slope, moving_average


def test_moving_average_sma_matches_pandas():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = series.rolling(3, min_periods=3).mean()
    result = moving_average(series, 3, "sma")
    pd.testing.assert_series_equal(result, expected)


def test_moving_average_ema_matches_pandas():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = series.ewm(span=3, adjust=False, min_periods=3).mean()
    result = moving_average(series, 3, "ema")
    pd.testing.assert_series_equal(result, expected)


def test_log_slope_k_values():
    series = pd.Series([1.0, 2.0, 4.0, 8.0])
    result = log_slope(series, 2)
    expected = pd.Series([np.nan, np.nan, math.log(2.0), math.log(2.0)])
    pd.testing.assert_series_equal(result, expected)


def test_moving_average_defaults_to_sma():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    expected = series.rolling(2, min_periods=2).mean()
    result = moving_average(series, 2)
    pd.testing.assert_series_equal(result, expected)
