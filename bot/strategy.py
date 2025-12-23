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
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

from bot import config
from bot.indicators import ma, donchian

TrendDir = Literal["LONG", "SHORT"]
MarketState = Literal["LONG", "SHORT", "RANGE"]
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
    flip_block_until_exec_bar_idx: int = -10**9
    flip_blocked_side: Optional[TrendDir] = None

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
    range_cfg = config.RANGE
    out["ma_fast_for_state"] = ma(out["close"], int(range_cfg["ma_fast_window"]))
    out["ma_slow_for_state"] = ma(out["close"], int(range_cfg["ma_slow_window"]))
    out["ma_slow_prev_for_state"] = out["ma_slow_for_state"].shift(1)
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
    out["quality_ma_prev"] = out["quality_ma"].shift(1)
    return out

def prepare_features_exec(df_exec: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns needed for execution decisions (e.g., 4h MA7).
    """
    out = df_exec.copy()
    ex = config.EXECUTION
    out["exec_ma"] = ma(out["close"], ex["window"])
    return out

def decide_trend_existence(row_1d: pd.Series) -> Optional[TrendDir]:
    te = config.TREND_EXISTENCE
    close = float(row_1d["close"])

    if te["indicator"] == "ma":
        ref = row_1d.get("trend_ma", np.nan)
        if np.isnan(ref):
            return None
        return "LONG" if close >= float(ref) else "SHORT"

    # donchian
    upper = row_1d.get("trend_upper", float("nan"))
    lower = row_1d.get("trend_lower", float("nan"))
    if np.isnan(upper) or np.isnan(lower):
        return None
    upper = float(upper)
    lower = float(lower)
    if close >= upper:
        return "LONG"
    if close <= lower:
        return "SHORT"
    mid = (upper + lower) / 2.0
    return "LONG" if close >= mid else "SHORT"

def decide_risk_mode(row_1d: pd.Series, trend: TrendDir) -> RiskMode:
    """
    Uses 1D MA (quality_ma) with a neutral band to classify risk mode,
    in a trend-direction-aware way.
    """
    close = float(row_1d["close"])
    qma = row_1d.get("quality_ma", np.nan)
    if np.isnan(qma):
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

def execution_gate_mode(
    row_exec: pd.Series,
    trend: TrendDir,
    exec_bar_idx: int,
    state: StrategyState,
    min_step_bars: int,
    require_trend_filter: bool,
) -> bool:
    if exec_bar_idx - state.last_exec_bar_idx < int(min_step_bars):
        return False
    if not require_trend_filter:
        return True

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

def is_range_regime(row_1d: pd.Series) -> bool:
    range_cfg = config.RANGE
    if not range_cfg["enabled"]:
        return False

    close = float(row_1d["close"])
    ma_fast = row_1d.get("ma_fast_for_state", np.nan)
    ma_slow = row_1d.get("ma_slow_for_state", np.nan)
    ma_slow_prev = row_1d.get("ma_slow_prev_for_state", np.nan)

    if np.isnan(ma_fast) or np.isnan(ma_slow):
        return False

    price_near_ma = abs(close - float(ma_slow)) / float(ma_slow) <= float(range_cfg["price_band_pct"])
    ma_converged = abs(float(ma_fast) - float(ma_slow)) / float(ma_slow) <= float(range_cfg["ma_band_pct"])

    low_slope = False
    if not np.isnan(ma_slow_prev):
        low_slope = abs(float(ma_slow) - float(ma_slow_prev)) / float(ma_slow) <= float(range_cfg["slope_band_pct"])

    return (price_near_ma and ma_converged) or (price_near_ma and low_slope)

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

def is_reduction(current: float, desired: float) -> bool:
    if abs(desired) < abs(current) - 1e-12:
        return True
    if current * desired < 0:
        return True
    return False

def _side_from_frac(frac: float) -> Literal["LONG", "SHORT", "FLAT"]:
    if abs(frac) < 1e-9:
        return "FLAT"
    return "LONG" if frac > 0 else "SHORT"

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
        "regime": "TREND"|"RANGE",
        "current_frac": ...,
        "desired_frac": ...,
        "target_frac": ...,
        "target_pos_frac": ...,
        "action": "HOLD"|"REBALANCE"|"EXIT",
        "reason": "...",
        "update_last_exec": bool,
      }
    """
    eps = 1e-9
    # Stage A: fetch latest rows.
    row_1d = _last_row_at_or_before(df_1d_feat, ts_ms)
    row_exec = _last_row_at_or_before(df_exec_feat, ts_ms)
    if row_1d is None or row_exec is None:
        return {
            "trend": "INSUFFICIENT_DATA",
            "risk": "RISK_OFF",
            "regime": "TREND",
            "current_frac": state.position.frac,
            "desired_frac": 0.0,
            "target_frac": state.position.frac,
            "target_pos_frac": state.position.frac,
            "action": "HOLD",
            "reason": "insufficient_data",
            "update_last_exec": False,
        }

    current = float(state.position.frac)
    # Stage B: determine market state (LONG/SHORT/RANGE).
    raw_dir = decide_trend_existence(row_1d)
    if raw_dir is None:
        return {
            "trend": "INSUFFICIENT_DATA",
            "risk": "RISK_OFF",
            "regime": "TREND",
            "current_frac": current,
            "desired_frac": 0.0,
            "target_frac": current,
            "target_pos_frac": current,
            "action": "HOLD",
            "reason": "insufficient_data",
            "update_last_exec": False,
        }

    range_regime = is_range_regime(row_1d)
    market_state: MarketState = "RANGE" if range_regime else raw_dir

    # Stage C: decide risk mode and desired target.
    if market_state == "RANGE":
        risk = "RISK_OFF"
        desired = 0.0
    else:
        risk = decide_risk_mode(row_1d, raw_dir)
        desired = compute_desired_target_frac(raw_dir, risk)

    # Stage D: apply flip cooldown logic.
    current_side = _side_from_frac(current)
    desired_side = _side_from_frac(desired)
    ex = config.EXECUTION

    if exec_bar_idx >= state.flip_block_until_exec_bar_idx:
        state.flip_blocked_side = None

    if current_side != "FLAT" and desired_side != "FLAT" and current_side != desired_side:
        # Flip detected: flatten first and start cooldown before re-entry.
        state.flip_blocked_side = desired_side
        state.flip_block_until_exec_bar_idx = exec_bar_idx + int(ex["build_min_step_bars"])
        desired = 0.0
        desired_side = "FLAT"

    if (
        current_side == "FLAT"
        and desired_side != "FLAT"
        and state.flip_blocked_side == desired_side
        and exec_bar_idx < state.flip_block_until_exec_bar_idx
    ):
        desired = 0.0
        desired_side = "FLAT"

    regime = "RANGE" if market_state == "RANGE" else "TREND"

    if abs(desired - current) < eps:
        reason = "range_hold" if market_state == "RANGE" and abs(current) < eps else "already_at_target"
        return {
            "trend": market_state,
            "risk": risk,
            "regime": regime,
            "current_frac": current,
            "desired_frac": desired,
            "target_frac": current,
            "target_pos_frac": current,
            "action": "HOLD",
            "reason": reason,
            "update_last_exec": False,
        }

    # Stage E: choose execution pacing and gate.
    reducing = is_reduction(current, desired)
    if reducing:
        min_step_bars = int(ex["reduce_min_step_bars"])
        max_delta = float(ex["reduce_max_delta_frac"])
        require_trend_filter = False
    else:
        min_step_bars = int(ex["build_min_step_bars"])
        max_delta = float(ex["build_max_delta_frac"])
        require_trend_filter = True

    allowed = execution_gate_mode(row_exec, raw_dir, exec_bar_idx, state, min_step_bars, require_trend_filter)
    if not allowed:
        reason = "range_exit_blocked" if market_state == "RANGE" and reducing else "execution_gate_blocked"
        return {
            "trend": market_state,
            "risk": risk,
            "regime": regime,
            "current_frac": current,
            "desired_frac": desired,
            "target_frac": current,
            "target_pos_frac": current,
            "action": "HOLD",
            "reason": reason,
            "update_last_exec": False,
        }

    # Stage F: compute smoothed target.
    target = smooth_target(current, desired, max_delta)

    if abs(target - current) < eps:
        action = "HOLD"
        reason = "range_hold" if market_state == "RANGE" and abs(current) < eps else "already_at_target"
        update_last_exec = False
    else:
        action = "REBALANCE"
        reason = "range_exit" if market_state == "RANGE" and reducing else "gate_passed"
        update_last_exec = True

    return {
        "trend": market_state,
        "trend_dir": raw_dir,
        "risk": risk,
        "regime": regime,
        "current_frac": current,
        "desired_frac": desired,
        "target_frac": target,
        "target_pos_frac": target,
        "action": action,
        "reason": reason,
        "update_last_exec": update_last_exec,
    }
