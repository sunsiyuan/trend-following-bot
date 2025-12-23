"""
strategy.py

Strategy logic only (no I/O, no plotting, no networking).

Inputs:
- klines dataframes per timeframe (already loaded from cache in backtests; in live you will fetch latest)
- config layer settings

Outputs:
- desired target position fraction (-1..+1) and action intent

Key design:
- indicators are swappable via config (ma vs donchian for trend existence)
- backtest.py and main.py both call the same decide_* functions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from bot import config
from bot.indicators import ma, donchian

TrendDir = Literal["LONG", "SHORT", "NO_TREND"]
RiskMode = config.RiskMode

@dataclass
class Position:
    """
    Position fraction is relative to equity (notional / equity), signed:
    +0.5 means 50% equity long; -0.5 means 50% equity short.
    """
    frac: float = 0.0

@dataclass
class StrategyState:
    position: Position = field(default_factory=Position)
    last_exec_bar_idx: int = -10**9  # bar index on execution timeframe

def _last_row_at_or_before(df: pd.DataFrame, ts_ms: int) -> Optional[pd.Series]:
    """
    Get last row with index <= ts_ms.
    Index is expected to be close_ts (ms).
    """
    if df.empty:
        return None
    # fast path
    idx = df.index.values
    pos = np.searchsorted(idx, ts_ms, side="right") - 1
    if pos < 0:
        return None
    return df.iloc[int(pos)]

def prepare_features_1d(df_1d: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns needed for 1D decisions, based on config.
    """
    out = df_1d.copy()
    # Trend existence
    te = config.TREND_EXISTENCE
    if te["indicator"] == "ma":
        out["trend_ma"] = ma(out["close"], te["window"])
    elif te["indicator"] == "donchian":
        upper, lower = donchian(out["high"], out["low"], te["window"])
        out["trend_upper"] = upper
        out["trend_lower"] = lower
    else:
        raise ValueError(f"Unknown trend indicator: {te['indicator']}")

    # Trend quality
    tq = config.TREND_QUALITY
    out["quality_ma"] = ma(out["close"], tq["window"])
    return out

def prepare_features_exec(df_exec: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns needed for execution decisions (e.g., 4h MA7).
    """
    out = df_exec.copy()
    ex = config.EXECUTION
    out["exec_ma"] = ma(out["close"], ex["window"])
    return out

def decide_trend_existence(row_1d: pd.Series) -> TrendDir:
    te = config.TREND_EXISTENCE
    close = float(row_1d["close"])

    if te["indicator"] == "ma":
        ref = row_1d.get("trend_ma", np.nan)
        if np.isnan(ref):
            return "NO_TREND"
        if close > float(ref):
            return "LONG"
        if close < float(ref):
            return "SHORT"
        return "NO_TREND"

    # donchian
    upper = row_1d.get("trend_upper", float("nan"))
    lower = row_1d.get("trend_lower", float("nan"))
    if np.isnan(upper) or np.isnan(lower):
        return "NO_TREND"
    if close > float(upper):
        return "LONG"
    if close < float(lower):
        return "SHORT"
    return "NO_TREND"

def decide_risk_mode(row_1d: pd.Series, trend: TrendDir) -> RiskMode:
    """
    Uses 1D MA (quality_ma) with a neutral band to classify risk mode,
    in a trend-direction-aware way.
    """
    close = float(row_1d["close"])
    qma = row_1d.get("quality_ma", np.nan)
    if trend == "NO_TREND" or np.isnan(qma):
        return "RISK_OFF"

    qma = float(qma)
    band = float(config.TREND_QUALITY["neutral_band_pct"])
    upper = qma * (1.0 + band)
    lower = qma * (1.0 - band)

    if trend == "LONG":
        if close >= upper:
            return "RISK_ON"
        if close >= lower:
            return "RISK_NEUTRAL"
        return "RISK_OFF"

    # SHORT
    if close <= lower:
        return "RISK_ON"
    if close <= upper:
        return "RISK_NEUTRAL"
    return "RISK_OFF"

def execution_gate(row_exec: pd.Series, trend: TrendDir, exec_bar_idx: int, state: StrategyState) -> bool:
    """
    Determines whether we are allowed to change position this bar (cooldown + direction filter).
    """
    ex = config.EXECUTION
    if exec_bar_idx - state.last_exec_bar_idx < int(ex["min_step_bars"]):
        return False

    close = float(row_exec["close"])
    ema = row_exec.get("exec_ma", np.nan)
    if np.isnan(ema):
        return False
    ema = float(ema)

    if trend == "LONG":
        return close > ema
    if trend == "SHORT":
        return close < ema
    return False

def compute_desired_target_frac(trend: TrendDir, risk: RiskMode) -> float:
    """
    Desired target position fraction in [-1, 1].
    """
    frac = float(config.MAX_POSITION_FRAC[risk])
    if trend == "LONG":
        return +frac
    if trend == "SHORT":
        return -frac
    return 0.0

def smooth_target(current: float, desired: float, max_delta: float) -> float:
    """
    Limit how much target can change per execution.
    """
    delta = desired - current
    if delta > max_delta:
        delta = max_delta
    elif delta < -max_delta:
        delta = -max_delta
    return current + delta

def decide(
    ts_ms: int,
    exec_bar_idx: int,
    df_1d_feat: pd.DataFrame,
    df_exec_feat: pd.DataFrame,
    state: StrategyState,
) -> Dict[str, object]:
    """
    Main decision function (used by both backtest and live runner).

    Returns a dict:
      {
        "trend": ...,
        "risk": ...,
        "current_frac": ...,
        "desired_frac": ...,
        "target_frac": ...,
        "action": "HOLD"|"REBALANCE"|"EXIT",
        "reason": "...",
        "update_last_exec": bool,
      }
    """
    row_1d = _last_row_at_or_before(df_1d_feat, ts_ms)
    row_exec = _last_row_at_or_before(df_exec_feat, ts_ms)
    if row_1d is None or row_exec is None:
        return {
            "trend": "NO_TREND",
            "risk": "RISK_OFF",
            "current_frac": state.position.frac,
            "desired_frac": 0.0,
            "target_frac": state.position.frac,
            "action": "HOLD",
            "reason": "insufficient_data",
            "update_last_exec": False,
        }

    trend = decide_trend_existence(row_1d)
    risk = decide_risk_mode(row_1d, trend)
    desired = compute_desired_target_frac(trend, risk)
    current = float(state.position.frac)

    # Emergency exit for risk-off or no-trend: allow reduce to 0 immediately.
    if desired == 0.0 and abs(current) > 1e-9:
        return {
            "trend": trend,
            "risk": risk,
            "current_frac": current,
            "desired_frac": desired,
            "target_frac": 0.0,
            "action": "EXIT",
            "reason": "risk_off_or_no_trend",
            "update_last_exec": True,  # treat as an exec event
        }

    allowed = execution_gate(row_exec, trend, exec_bar_idx, state)
    if not allowed:
        return {
            "trend": trend,
            "risk": risk,
            "current_frac": current,
            "desired_frac": desired,
            "target_frac": current,
            "action": "HOLD",
            "reason": "execution_gate_blocked",
            "update_last_exec": False,
        }

    max_delta = float(config.EXECUTION["max_delta_frac"])  # fraction of equity
    target = smooth_target(current, desired, max_delta)

    if abs(target - current) < 1e-9:
        action = "HOLD"
        reason = "already_at_target"
        update_last_exec = False
    else:
        action = "REBALANCE"
        reason = "gate_passed"
        update_last_exec = True

    return {
        "trend": trend,
        "risk": risk,
        "current_frac": current,
        "desired_frac": desired,
        "target_frac": target,
        "action": action,
        "reason": reason,
        "update_last_exec": update_last_exec,
    }
