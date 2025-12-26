"""
metrics.py

Small helpers to compute backtest metrics.
"""

from __future__ import annotations

import logging
import math
from typing import Dict

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

log = logging.getLogger(__name__)


def _to_equity_array(equity: ArrayLike) -> np.ndarray:
    if isinstance(equity, pd.Series):
        values = equity.to_numpy(dtype="float64", copy=False)
    else:
        values = np.asarray(equity, dtype="float64")
    values = values[np.isfinite(values)]
    return values

def max_drawdown_from_equity(equity: pd.Series) -> float:
    """
    Max drawdown as a fraction (e.g. -0.2 == -20% from peak).
    """
    if equity.empty:
        return 0.0
    # Drawdown computed vs rolling peak of equity.
    running_max = equity.cummax()
    dd = (equity / running_max) - 1.0
    return float(dd.min())


def daily_returns_from_equity(equity: pd.Series) -> pd.Series:
    """
    Convert an equity curve into daily percentage returns.
    """
    # pct_change uses consecutive equity values; first return is dropped.
    return equity.pct_change(fill_method=None).dropna()


def sharpe_ratio_from_daily_returns(
    daily_ret: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = 365,
) -> float:
    """
    Compute the annualized Sharpe ratio from daily returns.
    """
    # rf_daily converts annual risk-free rate to daily equivalent.
    rf_daily = (1 + rf_annual) ** (1 / periods_per_year) - 1
    excess = daily_ret - rf_daily
    std = excess.std(ddof=1)
    if len(excess) < 2 or std <= 1e-12:
        return 0.0
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def equity_returns_with_first_zero(equity: ArrayLike) -> np.ndarray:
    # Return series with first return forced to 0.0 (length matches equity).
    values = _to_equity_array(equity)
    if values.size == 0:
        return np.array([], dtype="float64")
    returns = np.zeros_like(values)
    prev = values[:-1]
    curr = values[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns[1:] = np.where(prev == 0, 0.0, curr / prev - 1.0)
    return returns


def sharpe_annualized_from_returns(
    returns: ArrayLike,
    periods_per_year: int = 365,
) -> float:
    values = _to_equity_array(returns)
    if values.size == 0:
        return float("nan")
    # ddof=0 uses population std for provided return series.
    std = values.std(ddof=0)
    if std == 0:
        return float("nan")
    return float(values.mean() / std * math.sqrt(periods_per_year))


def mdd_and_ulcer_index(equity: ArrayLike) -> tuple[float, float]:
    values = _to_equity_array(equity)
    if values.size == 0:
        return float("nan"), float("nan")
    # mdd uses min drawdown; ui uses RMS of drawdown depth.
    running_max = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_series = values / running_max - 1.0
        dd_depth = 1.0 - values / running_max
    mdd = float(np.min(dd_series)) if dd_series.size else float("nan")
    ui = float(np.sqrt(np.mean(dd_depth**2))) if dd_depth.size else float("nan")
    return mdd, ui


def annualized_return(start: float, end: float, days: int) -> float:
    if days <= 0 or start <= 0:
        return float("nan")
    return float((end / start) ** (365.0 / days) - 1.0)


def ulcer_index(equity: ArrayLike) -> float:
    values = _to_equity_array(equity)
    if values.size < 2:
        return 0.0
    if np.any(values <= 0):
        positives = values[values > 0]
        if positives.size < 2:
            log.warning("Ulcer index requires positive equity values; returning 0.0.")
            return 0.0
        log.warning("Ulcer index skipping non-positive equity values.")
        values = positives
    running_max = np.maximum.accumulate(values)
    # Ulcer index uses RMS of percentage drawdowns (negative values).
    drawdowns = values / running_max - 1.0
    return float(np.sqrt(np.mean(drawdowns**2)))


def ulcer_performance_index(equity: ArrayLike) -> float:
    values = _to_equity_array(equity)
    if values.size < 2:
        return 0.0
    if np.any(values <= 0):
        positives = values[values > 0]
        if positives.size < 2:
            log.warning("Ulcer performance index requires positive equity values; returning 0.0.")
            return 0.0
        log.warning("Ulcer performance index skipping non-positive equity values.")
        values = positives
    # UPI = total_return / ulcer_index (with special-case handling when ui == 0).
    total_ret = values[-1] / values[0] - 1.0
    ui = ulcer_index(values)
    if ui == 0.0:
        if total_ret > 0:
            return float("inf")
        if total_ret < 0:
            return float("-inf")
        return 0.0
    return float(total_ret / ui)


def safe_float_for_json(value: float) -> float | str:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def build_buy_hold_curve(
    dates: pd.Index,
    close_px: pd.Series,
    starting_cash: float,
) -> pd.Series:
    """
    Build a buy-and-hold equity curve aligned to the provided dates.
    """
    aligned_close = pd.to_numeric(close_px.reindex(dates), errors="coerce")
    if aligned_close.empty:
        return pd.Series(index=dates, dtype="float64")
    non_nan = aligned_close.dropna()
    if non_nan.empty:
        return pd.Series(index=dates, dtype="float64")
    aligned_close = aligned_close.ffill().bfill()
    # Buy-and-hold uses the first non-NaN close as entry price.
    entry_px = float(non_nan.iloc[0])
    qty = starting_cash / entry_px
    equity = qty * aligned_close
    return equity


def compute_equity_metrics(
    equity: pd.Series,
    starting_cash: float,
    rf_annual: float = 0.0,
) -> Dict[str, float]:
    """
    Compute standard equity curve metrics.
    """
    # Drops NaNs; if empty, returns flat metrics vs starting cash.
    equity = pd.to_numeric(equity, errors="coerce").dropna()
    if equity.empty:
        return {
            "ending_equity_usdc": float(starting_cash),
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "ulcer_index": 0.0,
            "ulcer_performance_index": 0.0,
        }

    ending_equity = float(equity.iloc[-1])
    total_return = ending_equity / starting_cash - 1
    max_drawdown = max_drawdown_from_equity(equity)
    sharpe_ratio = sharpe_ratio_from_daily_returns(
        daily_returns_from_equity(equity),
        rf_annual=rf_annual,
    )
    ui = ulcer_index(equity)
    upi = ulcer_performance_index(equity)

    return {
        "ending_equity_usdc": ending_equity,
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe_ratio),
        "ulcer_index": float(ui),
        "ulcer_performance_index": float(upi),
    }


def max_drawdown(equity: pd.Series) -> float:
    """
    Backwards-compatible wrapper for max drawdown.
    """
    return max_drawdown_from_equity(equity)


def total_return(equity: pd.Series) -> float:
    """
    Backwards-compatible wrapper for total return.
    """
    if equity.empty:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def total_return_from_equity(equity: ArrayLike) -> float:
    values = _to_equity_array(equity)
    if values.size < 2:
        return 0.0
    return float(values[-1] / values[0] - 1.0)


def equity_basic_metrics(equity: ArrayLike) -> Dict[str, float]:
    values = _to_equity_array(equity)
    if values.size == 0:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "ulcer_index": 0.0,
        }
    total_ret = total_return_from_equity(values)
    mdd, ui = mdd_and_ulcer_index(values)
    return {
        "total_return": float(total_ret),
        "max_drawdown": float(mdd),
        "ulcer_index": float(ui),
    }
