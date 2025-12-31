from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Dict, List

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

    def to_effective_dict(self) -> Dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["strategy_version"] = strat.STRATEGY_VERSION
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BacktestParams":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            timeframes=dict(payload.get("timeframes", dict(config.TIMEFRAMES))),
            trend_existence=dict(payload.get("trend_existence", dict(config.TREND_EXISTENCE))),
            trend_quality=dict(payload.get("trend_quality", dict(config.TREND_QUALITY))),
            execution=dict(payload.get("execution", dict(config.EXECUTION))),
            angle_sizing_enabled=bool(payload.get("angle_sizing_enabled", config.ANGLE_SIZING_ENABLED)),
            angle_sizing_a=float(payload.get("angle_sizing_a", config.ANGLE_SIZING_A)),
            angle_sizing_q=float(payload.get("angle_sizing_q", config.ANGLE_SIZING_Q)),
            vol_window_div=float(payload.get("vol_window_div", config.VOL_WINDOW_DIV)),
            vol_window_min=int(payload.get("vol_window_min", config.VOL_WINDOW_MIN)),
            vol_window_max=int(payload.get("vol_window_max", config.VOL_WINDOW_MAX)),
            vol_eps=float(payload.get("vol_eps", config.VOL_EPS)),
            direction_mode=str(payload.get("direction_mode", config.DIRECTION_MODE)),
            max_long_frac=float(payload.get("max_long_frac", config.MAX_LONG_FRAC)),
            max_short_frac=float(payload.get("max_short_frac", config.MAX_SHORT_FRAC)),
            starting_cash_usdc_per_symbol=float(
                payload.get("starting_cash_usdc_per_symbol", config.STARTING_CASH_USDC_PER_SYMBOL)
            ),
            taker_fee_bps=float(payload.get("taker_fee_bps", config.TAKER_FEE_BPS)),
            min_trade_notional_pct=float(
                payload.get("min_trade_notional_pct", config.MIN_TRADE_NOTIONAL_PCT)
            ),
        )

    @staticmethod
    def default_params_dict() -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "timeframes": dict(config.TIMEFRAMES),
            "trend_existence": dict(config.TREND_EXISTENCE),
            "trend_quality": dict(config.TREND_QUALITY),
            "execution": dict(config.EXECUTION),
            "angle_sizing_enabled": config.ANGLE_SIZING_ENABLED,
            "angle_sizing_a": config.ANGLE_SIZING_A,
            "angle_sizing_q": config.ANGLE_SIZING_Q,
            "vol_window_div": config.VOL_WINDOW_DIV,
            "vol_window_min": config.VOL_WINDOW_MIN,
            "vol_window_max": config.VOL_WINDOW_MAX,
            "vol_eps": config.VOL_EPS,
            "direction_mode": config.DIRECTION_MODE,
            "max_long_frac": config.MAX_LONG_FRAC,
            "max_short_frac": config.MAX_SHORT_FRAC,
            "starting_cash_usdc_per_symbol": config.STARTING_CASH_USDC_PER_SYMBOL,
            "taker_fee_bps": config.TAKER_FEE_BPS,
            "min_trade_notional_pct": config.MIN_TRADE_NOTIONAL_PCT,
        }

    @staticmethod
    def validate_and_materialize(input_params: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        defaults = BacktestParams.default_params_dict()

        def merge_with_schema(
            base: Dict[str, Any],
            override: Dict[str, Any],
            prefix: str = "",
        ) -> tuple[Dict[str, Any], List[str]]:
            merged: Dict[str, Any] = {}
            unapplied: List[str] = []

            for key, base_val in base.items():
                if key not in override:
                    merged[key] = base_val
                    continue

                override_val = override[key]
                path = f"{prefix}{key}"

                if isinstance(base_val, dict):
                    if isinstance(override_val, dict):
                        merged_val, sub_unapplied = merge_with_schema(
                            base_val,
                            override_val,
                            prefix=f"{path}.",
                        )
                        merged[key] = merged_val
                        unapplied.extend(sub_unapplied)
                    else:
                        unapplied.append(path)
                        merged[key] = base_val
                    continue

                if isinstance(override_val, dict) or isinstance(override_val, list):
                    unapplied.append(path)
                    merged[key] = base_val
                    continue

                merged[key] = override_val

            for key in override:
                if key not in base:
                    unapplied.append(f"{prefix}{key}")

            return merged, unapplied

        if input_params is None:
            input_params = {}
        if not isinstance(input_params, dict):
            raise TypeError("input_params must be a dict")

        effective, unapplied = merge_with_schema(defaults, input_params)
        return effective, sorted(set(unapplied))
