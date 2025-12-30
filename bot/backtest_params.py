from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Dict

from bot import config
from bot import strategy as strat

def _normalize_for_json(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return _normalize_for_json(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(key): _normalize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_json(value) for value in obj]
    if isinstance(obj, float):
        return repr(obj)
    return obj


def stable_json(obj: Any) -> str:
    normalized = _normalize_for_json(obj)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def calc_param_hash(hashable_dict: Dict[str, Any]) -> str:
    payload = stable_json(hashable_dict).encode("utf-8")
    return sha256(payload).hexdigest()


@dataclass(frozen=True)
class BacktestParams:
    schema_version: int = 1
    timeframes: Dict[str, str] = field(default_factory=dict)
    trend_existence: Dict[str, Any] = field(default_factory=dict)
    trend_quality: Dict[str, Any] = field(default_factory=dict)
    execution: Dict[str, Any] = field(default_factory=dict)
    angle_sizing_enabled: bool = config.ANGLE_SIZING_ENABLED
    angle_sizing_a: float = config.ANGLE_SIZING_A
    angle_sizing_q: float = config.ANGLE_SIZING_Q
    vol_window_div: float = config.VOL_WINDOW_DIV
    vol_window_min: int = config.VOL_WINDOW_MIN
    vol_window_max: int = config.VOL_WINDOW_MAX
    vol_eps: float = config.VOL_EPS
    direction_mode: str = "long_only"
    max_long_frac: float = 1.0
    max_short_frac: float = 0.25
    starting_cash_usdc_per_symbol: float = 0.0
    taker_fee_bps: float = 0.0
    min_trade_notional_pct: float = config.MIN_TRADE_NOTIONAL_PCT

    def to_hashable_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "strategy_version": strat.STRATEGY_VERSION,
            "timeframes": self.timeframes,
            "trend_existence": self.trend_existence,
            "trend_quality": self.trend_quality,
            "execution": self.execution,
            "angle_sizing_enabled": self.angle_sizing_enabled,
            "angle_sizing_a": self.angle_sizing_a,
            "angle_sizing_q": self.angle_sizing_q,
            "vol_window_div": self.vol_window_div,
            "vol_window_min": self.vol_window_min,
            "vol_window_max": self.vol_window_max,
            "vol_eps": self.vol_eps,
            "direction_mode": self.direction_mode,
            "max_long_frac": self.max_long_frac,
            "max_short_frac": self.max_short_frac,
            "starting_cash_usdc_per_symbol": self.starting_cash_usdc_per_symbol,
            "taker_fee_bps": self.taker_fee_bps,
            "min_trade_notional_pct": self.min_trade_notional_pct,
        }

    def param_hash(self) -> str:
        return calc_param_hash(self.to_hashable_dict())

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BacktestParams":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            timeframes=dict(payload.get("timeframes", {})),
            trend_existence=dict(payload.get("trend_existence", {})),
            trend_quality=dict(payload.get("trend_quality", {})),
            execution=dict(payload.get("execution", {})),
            angle_sizing_enabled=bool(payload.get("angle_sizing_enabled", config.ANGLE_SIZING_ENABLED)),
            angle_sizing_a=float(payload.get("angle_sizing_a", config.ANGLE_SIZING_A)),
            angle_sizing_q=float(payload.get("angle_sizing_q", config.ANGLE_SIZING_Q)),
            vol_window_div=float(payload.get("vol_window_div", config.VOL_WINDOW_DIV)),
            vol_window_min=int(payload.get("vol_window_min", config.VOL_WINDOW_MIN)),
            vol_window_max=int(payload.get("vol_window_max", config.VOL_WINDOW_MAX)),
            vol_eps=float(payload.get("vol_eps", config.VOL_EPS)),
            direction_mode=str(payload.get("direction_mode", "long_only")),
            max_long_frac=float(payload.get("max_long_frac", 1.0)),
            max_short_frac=float(payload.get("max_short_frac", 0.25)),
            starting_cash_usdc_per_symbol=float(payload.get("starting_cash_usdc_per_symbol", 0.0)),
            taker_fee_bps=float(payload.get("taker_fee_bps", 0.0)),
            min_trade_notional_pct=float(payload.get("min_trade_notional_pct", config.MIN_TRADE_NOTIONAL_PCT)),
        )
