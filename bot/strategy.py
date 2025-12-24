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
from bot.indicators import donchian, log_slope, moving_average

TrendDir = Literal["LONG", "SHORT"]
MarketState = Literal["LONG", "SHORT"]
RiskMode = config.RiskMode
DirectionMode = config.DirectionMode

DECISION_KEY_DEFAULTS: Dict[str, object] = {
    "raw_dir": None,
    "trend_slope": None,
    "trend_is_weak": False,
    "market_state": None,
    "risk_mode": None,
    "desired_side": None,
    "desired_pos_frac": None,
    "target_pos_frac": None,
    "regime": None,
    "action": None,
    "reason": None,
    "update_last_exec": None,
    "direction_mode": None,
    "effective_dir": None,
    "build_allowed": True,
}

def make_decision(**kwargs: object) -> Dict[str, object]:
    payload = dict(DECISION_KEY_DEFAULTS)
    payload.update(kwargs)
    return payload

DECISION_KEYS = tuple(sorted(make_decision().keys()))

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
    # Trend existence
    te = config.TREND_EXISTENCE
    if te["indicator"] == "ma":
        ma_type = te.get("ma_type", "sma")
        slope_k = int(te.get("slope_k", 2))
        out["trend_ma"] = moving_average(out["close"], te["window"], ma_type)
        out["trend_log_slope"] = log_slope(out["trend_ma"], slope_k)
    elif te["indicator"] == "donchian":
        upper, lower = donchian(out["high"], out["low"], te["window"])
        out["trend_upper"] = upper
        out["trend_lower"] = lower
    else:
        raise ValueError(f"Unknown trend indicator: {te['indicator']}")

    # Trend quality
    tq = config.TREND_QUALITY
    quality_ma_type = tq.get("ma_type", "sma")
    out["quality_ma"] = moving_average(out["close"], tq["window"], quality_ma_type)
    out["quality_ma_prev"] = out["quality_ma"].shift(1)
    return out

def prepare_features_exec(df_exec: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns needed for execution decisions (e.g., 4h MA7).
    """
    out = df_exec.copy()
    ex = config.EXECUTION
    ma_type = ex.get("ma_type", "sma")
    out["exec_ma"] = moving_average(out["close"], ex["window"], ma_type)
    return out

def decide_trend_existence(row_1d: pd.Series) -> Optional[TrendDir]:
    te = config.TREND_EXISTENCE
    close = float(row_1d["close"])

    if te["indicator"] == "ma":
        slope = row_1d.get("trend_log_slope", np.nan)
        if np.isnan(slope):
            return None
        slope = float(slope)
        if slope > 0:
            return "LONG"
        if slope < 0:
            return "SHORT"
        return None

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
    exec_ma = row_exec.get("exec_ma", np.nan)
    if np.isnan(exec_ma):
        return False
    exec_ma = float(exec_ma)

    if trend == "LONG":
        return close > exec_ma
    if trend == "SHORT":
        return close < exec_ma
    return False

def _apply_direction_mode(trend: TrendDir, direction_mode: DirectionMode) -> TrendDir:
    if direction_mode == "both_side":
        return trend
    if direction_mode == "long_only" and trend == "SHORT":
        return "NO_TREND"
    if direction_mode == "short_only" and trend == "LONG":
        return "NO_TREND"
    return trend

def compute_desired_target_frac(trend: TrendDir, risk: RiskMode) -> float:
    """
    Desired target position fraction in [-1, 1].
    """
    trend = _apply_direction_mode(trend, config.DIRECTION_MODE)
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

def _is_weak_trend(slope: float, db_pos: float, db_neg: float) -> bool:
    return (-db_neg <= slope <= db_pos)

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
        "raw_dir": ...,
        "market_state": ...,
        "risk_mode": ...,
        "desired_side": ...,
        "desired_pos_frac": ...,
        "target_pos_frac": ...,
        "regime": "TREND",
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
        current = float(state.position.frac)
        return make_decision(
            raw_dir=None,
            trend_slope=None,
            trend_is_weak=False,
            market_state=None,
            risk_mode="RISK_OFF",
            desired_side=_side_from_frac(0.0),
            desired_pos_frac=0.0,
            target_pos_frac=current,
            regime="TREND",
            action="HOLD",
            reason="insufficient_data",
            update_last_exec=False,
            direction_mode=config.DIRECTION_MODE,
            effective_dir=None,
            build_allowed=True,
        )

    current = float(state.position.frac)
    # Stage B: determine market state (LONG/SHORT).
    raw_dir = decide_trend_existence(row_1d)
    if raw_dir is None:
        return make_decision(
            raw_dir=None,
            trend_slope=None,
            trend_is_weak=False,
            market_state=None,
            risk_mode="RISK_OFF",
            desired_side=_side_from_frac(0.0),
            desired_pos_frac=0.0,
            target_pos_frac=current,
            regime="TREND",
            action="HOLD",
            reason="insufficient_data",
            update_last_exec=False,
            direction_mode=config.DIRECTION_MODE,
            effective_dir=None,
            build_allowed=True,
        )

    market_state: MarketState = raw_dir

    # Stage C: decide risk mode and desired target.
    te = config.TREND_EXISTENCE
    slope_val = row_1d.get("trend_log_slope", np.nan) if te["indicator"] == "ma" else np.nan
    trend_slope = float(slope_val) if not np.isnan(slope_val) else None
    deadband_pos = float(te.get("deadband_pos", 0.0))
    deadband_neg = float(te.get("deadband_neg", 0.0))
    weak = bool(trend_slope is not None and _is_weak_trend(trend_slope, deadband_pos, deadband_neg))
    current_side = _side_from_frac(current)

    effective_dir: TrendDir = raw_dir
    build_allowed = True
    if weak:
        build_allowed = False
        if current_side != "FLAT":
            effective_dir = current_side

    if weak and current_side == "FLAT":
        risk = "RISK_OFF"
        desired = 0.0
    else:
        risk = decide_risk_mode(row_1d, effective_dir)
        desired = compute_desired_target_frac(effective_dir, risk)

    # Stage D: apply flip cooldown logic.
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

    regime = "TREND"

    if abs(desired - current) < eps:
        reason = "already_at_target"
        return make_decision(
            raw_dir=raw_dir,
            trend_slope=trend_slope,
            trend_is_weak=weak,
            market_state=market_state,
            risk_mode=risk,
            desired_side=desired_side,
            desired_pos_frac=desired,
            target_pos_frac=current,
            regime=regime,
            action="HOLD",
            reason=reason,
            update_last_exec=False,
            direction_mode=config.DIRECTION_MODE,
            effective_dir=effective_dir,
            build_allowed=build_allowed,
        )

    # Stage E: choose execution pacing and gate.
    reducing = is_reduction(current, desired)
    if (not build_allowed) and (not reducing):
        reason = "deadband_no_build"
        return make_decision(
            raw_dir=raw_dir,
            trend_slope=trend_slope,
            trend_is_weak=weak,
            market_state=market_state,
            risk_mode=risk,
            desired_side=desired_side,
            desired_pos_frac=desired,
            target_pos_frac=current,
            regime=regime,
            action="HOLD",
            reason=reason,
            update_last_exec=False,
            direction_mode=config.DIRECTION_MODE,
            effective_dir=effective_dir,
            build_allowed=build_allowed,
        )
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
        reason = "execution_gate_blocked"
        return make_decision(
            raw_dir=raw_dir,
            trend_slope=trend_slope,
            trend_is_weak=weak,
            market_state=market_state,
            risk_mode=risk,
            desired_side=desired_side,
            desired_pos_frac=desired,
            target_pos_frac=current,
            regime=regime,
            action="HOLD",
            reason=reason,
            update_last_exec=False,
            direction_mode=config.DIRECTION_MODE,
            effective_dir=effective_dir,
            build_allowed=build_allowed,
        )

    # Stage F: compute smoothed target.
    target = smooth_target(current, desired, max_delta)

    if abs(target - current) < eps:
        action = "HOLD"
        reason = "already_at_target"
        update_last_exec = False
    else:
        action = "REBALANCE"
        reason = "gate_passed"
        update_last_exec = True

    return make_decision(
        raw_dir=raw_dir,
        trend_slope=trend_slope,
        trend_is_weak=weak,
        market_state=market_state,
        risk_mode=risk,
        desired_side=desired_side,
        desired_pos_frac=desired,
        target_pos_frac=target,
        regime=regime,
        action=action,
        reason=reason,
        update_last_exec=update_last_exec,
        direction_mode=config.DIRECTION_MODE,
        effective_dir=effective_dir,
        build_allowed=build_allowed,
    )
