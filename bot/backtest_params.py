from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Dict


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
    direction_mode: str = "long_only"
    starting_cash_usdc_per_symbol: float = 0.0
    taker_fee_bps: float = 0.0

    def to_hashable_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timeframes": self.timeframes,
            "trend_existence": self.trend_existence,
            "trend_quality": self.trend_quality,
            "execution": self.execution,
            "direction_mode": self.direction_mode,
            "starting_cash_usdc_per_symbol": self.starting_cash_usdc_per_symbol,
            "taker_fee_bps": self.taker_fee_bps,
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
            direction_mode=str(payload.get("direction_mode", "long_only")),
            starting_cash_usdc_per_symbol=float(payload.get("starting_cash_usdc_per_symbol", 0.0)),
            taker_fee_bps=float(payload.get("taker_fee_bps", 0.0)),
        )
