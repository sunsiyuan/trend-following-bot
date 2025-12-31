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
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd

from bot import config
from bot.indicators import donchian, hlc3, log_slope, moving_average, quantize_toward_zero

STRATEGY_VERSION = "v3"
# NOTE: Any structural strategy change (components added/removed, signal definitions,
# or position-sizing logic changes) must bump STRATEGY_VERSION.

TrendDir = Literal["LONG", "SHORT"]
MarketState = Literal["LONG", "SHORT"]
DirectionMode = config.DirectionMode

DECISION_KEY_DEFAULTS: Dict[str, object] = {
    "raw_dir": None,
    "market_state": None,
    "risk_mode": None,
    "fast_dir": None,
    "fast_sign": None,
    "slow_dir": None,
    "slow_sign": None,
    "hlc3": None,
    "ema_fast": None,
    "ema_slow": None,
    "fast_state": None,
    "slow_state": None,
    "s_fast": None,
    "s_slow": None,
    "align": None,
    "fast_state_deadband_pct": None,
    "deadband_conf": None,
    "deadband_active": None,
    "sigma_price": None,
    "sigma_mismatch_mean": None,
    "z": None,
    "zq": None,
    "z_dir": None,
    "penalty": None,
    "penalty_q": None,
    "desired_side": None,
    "desired_pos_frac": None,
    "target_pos_frac": None,
    "regime": None,
    "action": None,
    "reason": None,
    "update_last_exec": None,
    "direction_mode": None,
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

def _risk_value(params: Any, name: str, fallback: Any) -> Any:
    if params is None:
        return fallback
    return getattr(params, name, fallback)

def _vol_window_from_fast_window(
    w_fast: int,
    vol_window_div: float,
    vol_window_min: int,
    vol_window_max: int,
) -> int:
    n = int(round(w_fast / vol_window_div))
    return max(vol_window_min, min(vol_window_max, n))

def compute_deadband_conf(
    fast_state: pd.Series,
    deadband_pct: float,
) -> tuple[pd.Series, pd.Series]:
    deadband_pct = float(deadband_pct)
    if np.isnan(deadband_pct) or deadband_pct <= 0:
        conf = pd.Series(1.0, index=fast_state.index)
        active = pd.Series(False, index=fast_state.index)
        return conf, active
    eps = np.log1p(deadband_pct)
    abs_fast = fast_state.abs()
    conf = (abs_fast / eps).clip(lower=0.0, upper=1.0)
    conf = conf.where(~fast_state.isna(), 1.0)
    active = (abs_fast < eps) & fast_state.notna()
    return conf, active

def prepare_features_1d(df_1d: pd.DataFrame, params: Any | None = None) -> pd.DataFrame:
    """
    Adds columns needed for 1D decisions, based on config.
    """
    out = df_1d.copy()
    close = out["close"]
    high = out["high"] if "high" in out.columns else None
    low = out["low"] if "low" in out.columns else None
    # HLC3 price proxy: (high + low + close) / 3 with close fallback when high/low missing.
    out["hlc3"] = hlc3(high, low, close)
    price_series = out["hlc3"]
    # Trend existence
    te = config.TREND_EXISTENCE
    if params is not None:
        te = getattr(params, "trend_existence", te) or te
    if te["indicator"] == "ma":
        ma_type = te.get("ma_type", "sma")
        slope_k = int(te.get("slope_k", 2))
        # trend_ma: moving average on hlc3; slope uses k-lag log change per bar.
        out["trend_ma"] = moving_average(price_series, te["window"], ma_type)
        out["trend_log_slope"] = log_slope(out["trend_ma"], slope_k)
    elif te["indicator"] == "donchian":
        # Donchian uses prior-window extremes (shifted by 1 bar in indicator helper).
        upper, lower = donchian(out["high"], out["low"], te["window"])
        out["trend_upper"] = upper
        out["trend_lower"] = lower
    else:
        raise ValueError(f"Unknown trend indicator: {te['indicator']}")

    # Trend quality
    tq = config.TREND_QUALITY
    quality_ma_type = tq.get("ma_type", "sma")
    # quality_ma is always MA-based per config; shift(1) used for any potential lookback use.
    out["quality_ma"] = moving_average(price_series, tq["window"], quality_ma_type)
    out["quality_ma_prev"] = out["quality_ma"].shift(1)
    slope_k = int(config.TREND_EXISTENCE.get("slope_k", 2))
    out["quality_log_slope"] = log_slope(out["quality_ma"], slope_k)

    if te["indicator"] == "ma" and tq["indicator"] == "ma":
        w_fast = int(config.TREND_EXISTENCE["window"])
        vol_window_div = float(_risk_value(params, "vol_window_div", config.VOL_WINDOW_DIV))
        vol_window_min = int(_risk_value(params, "vol_window_min", config.VOL_WINDOW_MIN))
        vol_window_max = int(_risk_value(params, "vol_window_max", config.VOL_WINDOW_MAX))
        n_vol = _vol_window_from_fast_window(w_fast, vol_window_div, vol_window_min, vol_window_max)
        # logret: log(hlc3_t) - log(hlc3_{t-1}); rolling std uses min_periods=n_vol.
        out["logret"] = np.log(price_series).diff()
        # delta: fast-slow log-slope mismatch.
        out["delta"] = out["trend_log_slope"] - out["quality_log_slope"]
        out["sigma_price"] = out["logret"].rolling(n_vol, min_periods=n_vol).std()
        # alpha_f/alpha_s use EMA-equivalent smoothing factors regardless of MA type.
        alpha_f = 2.0 / (float(config.TREND_EXISTENCE["window"]) + 1.0)
        alpha_s = 2.0 / (float(config.TREND_QUALITY["window"]) + 1.0)
        vf = alpha_f / (2.0 - alpha_f) + alpha_s / (2.0 - alpha_s)
        # sigma_mismatch_mean rescales sigma_price by vf and slope_k.
        out["sigma_mismatch_mean"] = out["sigma_price"] * np.sqrt(vf) / np.sqrt(float(slope_k))
        # z-score of slope mismatch, guarded by VOL_EPS.
        vol_eps = float(_risk_value(params, "vol_eps", config.VOL_EPS))
        out["z"] = out["delta"] / np.maximum(
            out["sigma_mismatch_mean"],
            vol_eps,
        )
        # zq: quantized z toward zero (step = ANGLE_SIZING_Q).
        angle_sizing_q = float(_risk_value(params, "angle_sizing_q", config.ANGLE_SIZING_Q))
        out["zq"] = quantize_toward_zero(out["z"], angle_sizing_q)
        # fast_state/slow_state: log-distance of price vs MAs.
        out["fast_state"] = np.log(out["hlc3"]) - np.log(out["trend_ma"])
        out["slow_state"] = np.log(out["hlc3"]) - np.log(out["quality_ma"])
        # fast_sign/slow_sign map state to {+1,0,-1} with NaN preserved.
        out["fast_sign"] = np.where(
            out["fast_state"] > 0,
            1.0,
            np.where(out["fast_state"] < 0, -1.0, np.where(out["fast_state"].isna(), np.nan, 0.0)),
        )
        slow_sign = np.where(
            out["slow_state"] > 0,
            1.0,
            np.where(out["slow_state"] < 0, -1.0, np.where(out["slow_state"].isna(), np.nan, 0.0)),
        )
        out["slow_sign"] = slow_sign
        # z_dir: z aligned to slow_sign; negative implies misalignment.
        out["z_dir"] = slow_sign * out["z"]
        # penalty captures only negative alignment; then quantized to ANGLE_SIZING_Q.
        out["penalty"] = np.maximum(0.0, -out["z_dir"])
        out["penalty_q"] = np.floor(out["penalty"] / angle_sizing_q) * angle_sizing_q
        # align: [0,1] attenuation from penalty_q via tanh (higher penalty -> lower align).
        angle_sizing_a = float(_risk_value(params, "angle_sizing_a", config.ANGLE_SIZING_A))
        out["align"] = 1.0 - np.tanh(out["penalty_q"] / angle_sizing_a)
        out["align"] = np.clip(out["align"], 0.0, 1.0)
        # NaN guard: if any prerequisite is NaN, force align=1.0 (no attenuation).
        nan_mask = out[
            [
                "hlc3",
                "trend_ma",
                "quality_ma",
                "trend_log_slope",
                "quality_log_slope",
                "sigma_price",
                "sigma_mismatch_mean",
                "z",
                "fast_state",
                "slow_state",
                "fast_sign",
                "slow_sign",
            ]
        ].isna().any(axis=1)
        out.loc[nan_mask, "align"] = 1.0
    else:
        # When trend/quality indicators are not MA-based, alignment math is skipped.
        out["logret"] = np.nan
        out["delta"] = np.nan
        out["sigma_price"] = np.nan
        out["sigma_mismatch_mean"] = np.nan
        out["z"] = np.nan
        out["zq"] = np.nan
        out["fast_state"] = np.nan
        out["slow_state"] = np.nan
        out["fast_sign"] = np.nan
        out["slow_sign"] = np.nan
        out["z_dir"] = np.nan
        out["penalty"] = np.nan
        out["penalty_q"] = np.nan
        # Default align to 1.0 when angle sizing inputs are absent.
        out["align"] = 1.0
    deadband_pct = float(te.get("fast_state_deadband_pct", 0.0))
    deadband_conf, deadband_active = compute_deadband_conf(out["fast_state"], deadband_pct)
    out["fast_state_deadband_pct"] = deadband_pct
    out["deadband_conf"] = deadband_conf
    out["deadband_active"] = deadband_active
    return out

def prepare_features_exec(df_exec: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns needed for execution decisions (e.g., 4h MA7).
    """
    out = df_exec.copy()
    ex = config.EXECUTION
    ma_type = ex.get("ma_type", "sma")
    # exec_ma is the execution-layer MA used as a trend filter in gating.
    out["exec_ma"] = moving_average(out["close"], ex["window"], ma_type)
    return out

def decide_trend_existence(row_1d: pd.Series) -> Optional[TrendDir]:
    te = config.TREND_EXISTENCE
    close = float(row_1d.get("hlc3", row_1d["close"]))

    if te["indicator"] == "ma":
        # Use log-slope sign of trend_ma to decide direction.
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
    # Breakout above upper => LONG; below lower => SHORT; else midline decides.
    if close >= upper:
        return "LONG"
    if close <= lower:
        return "SHORT"
    mid = (upper + lower) / 2.0
    return "LONG" if close >= mid else "SHORT"

def decide_slow_dir(row_1d: pd.Series) -> Optional[TrendDir]:
    # Slow direction based on quality_log_slope sign.
    slope = row_1d.get("quality_log_slope", np.nan)
    if np.isnan(slope):
        return None
    slope = float(slope)
    if slope > 0:
        return "LONG"
    if slope < 0:
        return "SHORT"
    return None

def _dir_from_sign(sign: float) -> Optional[TrendDir]:
    # Map numeric sign to LONG/SHORT, preserving NaN as None.
    if np.isnan(sign):
        return None
    if sign > 0:
        return "LONG"
    if sign < 0:
        return "SHORT"
    return None

def execution_gate_mode(
    row_exec: pd.Series,
    trend: TrendDir,
    exec_bar_idx: int,
    state: StrategyState,
    min_step_bars: int,
    require_trend_filter: bool,
) -> bool:
    # Enforce minimum spacing between executions on execution timeframe.
    if exec_bar_idx - state.last_exec_bar_idx < int(min_step_bars):
        return False
    if not require_trend_filter:
        return True

    close = float(row_exec["close"])
    exec_ma = row_exec.get("exec_ma", np.nan)
    if np.isnan(exec_ma):
        return False
    exec_ma = float(exec_ma)

    # Trend filter: only allow if close is above/below exec_ma in the direction of trend.
    if trend == "LONG":
        return close > exec_ma
    if trend == "SHORT":
        return close < exec_ma
    return False

def compute_desired_target_frac(
    fast_sign: float,
    align: float,
    direction_mode: DirectionMode,
    deadband_conf: float = 1.0,
) -> float:
    def apply_pos_scale(desired_raw: float, max_long_frac: float, max_short_frac: float) -> float:
        if desired_raw > 0:
            return desired_raw * max_long_frac
        if desired_raw < 0:
            return desired_raw * max_short_frac
        return 0.0

    # Convert directional sign (+1/-1/0) and alignment scalar into target fraction.
    align = float(np.clip(align, 0.0, 1.0))
    deadband_conf = float(deadband_conf)
    if np.isnan(deadband_conf):
        deadband_conf = 1.0
    deadband_conf = float(np.clip(deadband_conf, 0.0, 1.0))
    if np.isnan(fast_sign) or fast_sign == 0:
        return 0.0
    if direction_mode == "both_side":
        desired_raw = float(fast_sign) * align
        desired_raw *= deadband_conf
        return apply_pos_scale(desired_raw, config.MAX_LONG_FRAC, config.MAX_SHORT_FRAC)
    if direction_mode == "long_only":
        desired_raw = align if fast_sign > 0 else 0.0
        desired_raw *= deadband_conf
        return apply_pos_scale(desired_raw, config.MAX_LONG_FRAC, config.MAX_SHORT_FRAC)
    if direction_mode == "short_only":
        desired_raw = -align if fast_sign < 0 else 0.0
        desired_raw *= deadband_conf
        return apply_pos_scale(desired_raw, config.MAX_LONG_FRAC, config.MAX_SHORT_FRAC)
    return 0.0

def smooth_target(current: float, desired: float, max_delta: float) -> float:
    """
    Limit how much target can change per execution.
    """
    # Clamp target change to +/- max_delta.
    delta = desired - current
    if delta > max_delta:
        delta = max_delta
    elif delta < -max_delta:
        delta = -max_delta
    return current + delta

def is_reduction(current: float, desired: float) -> bool:
    # Reduction if absolute exposure decreases or direction flips.
    if abs(desired) < abs(current) - 1e-12:
        return True
    if current * desired < 0:
        return True
    return False

def _side_from_frac(frac: float) -> Literal["LONG", "SHORT", "FLAT"]:
    # Map signed fraction to side label with tiny epsilon.
    if abs(frac) < 1e-9:
        return "FLAT"
    return "LONG" if frac > 0 else "SHORT"

def decide(
    ts_ms: int,
    exec_bar_idx: int,
    df_1d_feat: pd.DataFrame,
    df_exec_feat: pd.DataFrame,
    state: StrategyState,
    params: Any | None = None,
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
            market_state=None,
            risk_mode=None,
            desired_side=_side_from_frac(0.0),
            desired_pos_frac=0.0,
            target_pos_frac=current,
            regime="TREND",
            action="HOLD",
            reason="insufficient_data",
            update_last_exec=False,
            direction_mode=config.DIRECTION_MODE,
        )

    current = float(state.position.frac)
    # Stage B: determine market state (LONG/SHORT).
    # raw_dir uses trend existence (MA slope or Donchian); market_state follows fast_sign.
    raw_dir = decide_trend_existence(row_1d)
    fast_sign = float(row_1d.get("fast_sign", np.nan))
    slow_sign = float(row_1d.get("slow_sign", np.nan))
    fast_dir = _dir_from_sign(fast_sign)
    slow_dir = _dir_from_sign(slow_sign)
    market_state: Optional[MarketState] = fast_dir

    # Stage C: decide desired target.
    # align is an attenuation factor; if disabled, force align=1.0.
    angle_sizing_enabled = bool(_risk_value(params, "angle_sizing_enabled", config.ANGLE_SIZING_ENABLED))
    align = float(row_1d.get("align", 1.0)) if angle_sizing_enabled else 1.0
    deadband_conf = float(row_1d.get("deadband_conf", 1.0))
    desired = compute_desired_target_frac(
        fast_sign,
        align,
        config.DIRECTION_MODE,
        deadband_conf=deadband_conf,
    )
    deadband_active = bool(row_1d.get("deadband_active", False))
    fast_state_deadband_pct = row_1d.get("fast_state_deadband_pct")

    # Stage D: apply flip cooldown logic.
    current_side = _side_from_frac(current)
    desired_side = _side_from_frac(desired)
    ex = config.EXECUTION

    if exec_bar_idx >= state.flip_block_until_exec_bar_idx:
        state.flip_blocked_side = None

    # If flipping directly, flatten first and block re-entry for build_min_step_bars.
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
            market_state=market_state,
            risk_mode=None,
            fast_dir=fast_dir,
            fast_sign=fast_sign,
            slow_dir=slow_dir,
            slow_sign=slow_sign,
            hlc3=row_1d.get("hlc3"),
            ema_fast=row_1d.get("trend_ma"),
            ema_slow=row_1d.get("quality_ma"),
            fast_state=row_1d.get("fast_state"),
            slow_state=row_1d.get("slow_state"),
            s_fast=row_1d.get("trend_log_slope"),
            s_slow=row_1d.get("quality_log_slope"),
            align=align,
            fast_state_deadband_pct=fast_state_deadband_pct,
            deadband_conf=deadband_conf,
            deadband_active=deadband_active,
            sigma_price=row_1d.get("sigma_price"),
            sigma_mismatch_mean=row_1d.get("sigma_mismatch_mean"),
            z=row_1d.get("z"),
            zq=row_1d.get("zq"),
            z_dir=row_1d.get("z_dir"),
            penalty=row_1d.get("penalty"),
            penalty_q=row_1d.get("penalty_q"),
            desired_side=desired_side,
            desired_pos_frac=desired,
            target_pos_frac=current,
            regime=regime,
            action="HOLD",
            reason=reason,
            update_last_exec=False,
            direction_mode=config.DIRECTION_MODE,
        )

    # Stage E: choose execution pacing and gate.
    # Reductions allow faster cadence and skip trend filter; builds are gated by exec_ma.
    reducing = is_reduction(current, desired)
    if reducing:
        min_step_bars = int(ex["reduce_min_step_bars"])
        max_delta = float(ex["reduce_max_delta_frac"])
        require_trend_filter = False
    else:
        min_step_bars = int(ex["build_min_step_bars"])
        max_delta = float(ex["build_max_delta_frac"])
        require_trend_filter = True

    gate_trend: Optional[TrendDir] = None
    if desired > 0:
        gate_trend = "LONG"
    elif desired < 0:
        gate_trend = "SHORT"

    if require_trend_filter and gate_trend is None:
        allowed = False
    else:
        allowed = execution_gate_mode(
            row_exec,
            gate_trend or "LONG",
            exec_bar_idx,
            state,
            min_step_bars,
            require_trend_filter,
        )
    if not allowed:
        reason = "execution_gate_blocked"
        return make_decision(
            raw_dir=raw_dir,
            market_state=market_state,
            risk_mode=None,
            fast_dir=fast_dir,
            fast_sign=fast_sign,
            slow_dir=slow_dir,
            slow_sign=slow_sign,
            hlc3=row_1d.get("hlc3"),
            ema_fast=row_1d.get("trend_ma"),
            ema_slow=row_1d.get("quality_ma"),
            fast_state=row_1d.get("fast_state"),
            slow_state=row_1d.get("slow_state"),
            s_fast=row_1d.get("trend_log_slope"),
            s_slow=row_1d.get("quality_log_slope"),
            align=align,
            fast_state_deadband_pct=fast_state_deadband_pct,
            deadband_conf=deadband_conf,
            deadband_active=deadband_active,
            sigma_price=row_1d.get("sigma_price"),
            sigma_mismatch_mean=row_1d.get("sigma_mismatch_mean"),
            z=row_1d.get("z"),
            zq=row_1d.get("zq"),
            z_dir=row_1d.get("z_dir"),
            penalty=row_1d.get("penalty"),
            penalty_q=row_1d.get("penalty_q"),
            desired_side=desired_side,
            desired_pos_frac=desired,
            target_pos_frac=current,
            regime=regime,
            action="HOLD",
            reason=reason,
            update_last_exec=False,
            direction_mode=config.DIRECTION_MODE,
        )

    # Stage F: compute smoothed target.
    # Smooth target caps per-step change; may produce HOLD if too small.
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
        market_state=market_state,
        risk_mode=None,
        fast_dir=fast_dir,
        fast_sign=fast_sign,
        slow_dir=slow_dir,
        slow_sign=slow_sign,
        hlc3=row_1d.get("hlc3"),
        ema_fast=row_1d.get("trend_ma"),
        ema_slow=row_1d.get("quality_ma"),
        fast_state=row_1d.get("fast_state"),
        slow_state=row_1d.get("slow_state"),
        s_fast=row_1d.get("trend_log_slope"),
        s_slow=row_1d.get("quality_log_slope"),
        align=align,
        fast_state_deadband_pct=fast_state_deadband_pct,
        deadband_conf=deadband_conf,
        deadband_active=deadband_active,
        sigma_price=row_1d.get("sigma_price"),
        sigma_mismatch_mean=row_1d.get("sigma_mismatch_mean"),
        z=row_1d.get("z"),
        zq=row_1d.get("zq"),
        z_dir=row_1d.get("z_dir"),
        penalty=row_1d.get("penalty"),
        penalty_q=row_1d.get("penalty_q"),
        desired_side=desired_side,
        desired_pos_frac=desired,
        target_pos_frac=target,
        regime=regime,
        action=action,
        reason=reason,
        update_last_exec=update_last_exec,
        direction_mode=config.DIRECTION_MODE,
    )
