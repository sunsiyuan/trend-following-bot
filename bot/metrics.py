"""
metrics.py

Small helpers to compute backtest metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

def max_drawdown(equity: pd.Series) -> float:
    """
    Max drawdown as a fraction (e.g. 0.2 == -20% from peak).
    """
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = (equity / running_max) - 1.0
    return float(dd.min())

def total_return(equity: pd.Series) -> float:
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
        pnl = t.get("realized_pnl")
        if pnl is None:
            continue
        count += 1
        if pnl > 0:
            wins += 1
    return float(wins / count) if count else 0.0
