"""
Execution policy for translating position deltas into trade intents.
"""
from __future__ import annotations

from typing import Dict


def compute_trade_intent(
    *,
    equity: float,
    current_notional: float,
    target_notional: float,
    min_trade_notional_pct: float,
    must_trade: bool,
    eps: float = 1e-12,
) -> Dict[str, float | str]:
    delta_notional = float(target_notional) - float(current_notional)
    threshold_notional = float(min_trade_notional_pct) * float(equity)

    threshold_floor = max(threshold_notional, eps)

    if must_trade:
        trade_intent = "TRADE"
        reason = "trade"
    elif abs(delta_notional) < threshold_floor:
        trade_intent = "NOOP_SMALL_DELTA"
        reason = "noop_small_delta"
    else:
        trade_intent = "TRADE"
        reason = "trade"

    return {
        "trade_intent": trade_intent,
        "delta_notional": delta_notional,
        "threshold_notional": threshold_notional,
        "reason": reason,
    }
