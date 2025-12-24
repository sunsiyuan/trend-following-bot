"""
metrics.py

Small helpers to compute backtest metrics.
"""

from __future__ import annotations

import math
from typing import Dict, List

import pandas as pd


def max_drawdown_from_equity(equity: pd.Series) -> float:
    """
    Max drawdown as a fraction (e.g. -0.2 == -20% from peak).
    """
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = (equity / running_max) - 1.0
    return float(dd.min())


def daily_returns_from_equity(equity: pd.Series) -> pd.Series:
    """
    Convert an equity curve into daily percentage returns.
    """
    return equity.pct_change(fill_method=None).dropna()


def sharpe_ratio_from_daily_returns(
    daily_ret: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = 365,
) -> float:
    """
    Compute the annualized Sharpe ratio from daily returns.
    """
    rf_daily = (1 + rf_annual) ** (1 / periods_per_year) - 1
    excess = daily_ret - rf_daily
    std = excess.std(ddof=1)
    if len(excess) < 2 or std <= 1e-12:
        return 0.0
    return float(excess.mean() / std * math.sqrt(periods_per_year))


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
    equity = pd.to_numeric(equity, errors="coerce").dropna()
    if equity.empty:
        return {
            "ending_equity_usdc": float(starting_cash),
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

    ending_equity = float(equity.iloc[-1])
    total_return = ending_equity / starting_cash - 1
    max_drawdown = max_drawdown_from_equity(equity)
    sharpe_ratio = sharpe_ratio_from_daily_returns(
        daily_returns_from_equity(equity),
        rf_annual=rf_annual,
    )

    return {
        "ending_equity_usdc": ending_equity,
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe_ratio),
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


def trade_win_rate(trades: List[Dict]) -> float:
    """
    Approximate win-rate using per-trade realized PnL snapshots (if provided).
    """
    if not trades:
        return 0.0
    wins = 0
    count = 0
    for t in trades:
        pnl = t.get("realized_pnl_usdc")
        if pnl is None:
            pnl = t.get("realized_pnl")
        if pnl is None:
            continue
        count += 1
        if pnl > 0:
            wins += 1
    return float(wins / count) if count else 0.0
