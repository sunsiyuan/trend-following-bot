"""
Project config.

Single source of truth for:
- which symbols to trade (default quote is USDC)
- which timeframe is used for each decision layer
- which indicator each layer uses (MA vs Donchian, etc.)
- backtest and storage settings

Keep this file boring: constants + small helper validators only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, TypedDict

# -----------------------------
# Market / symbols
# -----------------------------

QUOTE_ASSET: str = "USDC"

# Symbols are "coin" names for perpetuals by default (BTC, ETH, SOL...).
# If you later add SPOT support, you may need mapping (see data_client.py).
SYMBOLS: List[str] = ["BTC", "ETH", "SOL"]

# "perp" or "spot" (v1 defaults to perp; spot naming is more complex).
MARKET_TYPE: Literal["perp", "spot"] = "perp"

# -----------------------------
# Timeframes
# -----------------------------

# Supported intervals include: "1m","3m","5m","15m","30m","1h","2h","4h","8h","12h","1d","3d","1w","1M"
TIMEFRAMES: Dict[str, str] = {
    "trend": "1d",
    "execution": "4h",
}

# -----------------------------
# Indicator layer configs
# -----------------------------

IndicatorType = Literal["ma", "donchian"]

class TrendExistenceCfg(TypedDict):
    indicator: IndicatorType
    timeframe: str
    window: int

class TrendQualityCfg(TypedDict):
    indicator: Literal["ma"]  # v1 uses MA for quality
    timeframe: str
    window: int
    neutral_band_pct: float  # e.g. 0.01 = +/-1% around MA treated as neutral

class ExecutionCfg(TypedDict):
    indicator: Literal["ma"]  # v1 execution uses MA
    timeframe: str
    window: int
    build_min_step_bars: int   # execution guard (cooldown) in bars of execution timeframe
    build_max_delta_frac: float  # max change in target position fraction per execution (0..1)
    reduce_min_step_bars: int  # execution guard for reductions
    reduce_max_delta_frac: float  # max change in target position fraction per execution (0..1)

class RangeCfg(TypedDict):
    enabled: bool
    price_band_pct: float
    ma_band_pct: float
    slope_band_pct: float

TREND_EXISTENCE: TrendExistenceCfg = {
    "indicator": "ma",  # "ma" or "donchian"
    "timeframe": TIMEFRAMES["trend"],
    "window": 15,
}

TREND_QUALITY: TrendQualityCfg = {
    "indicator": "ma",
    "timeframe": TIMEFRAMES["trend"],
    "window": 50,             # e.g. 50 or 90
    "neutral_band_pct": 0.01, # +/-1% band around MA
}

EXECUTION: ExecutionCfg = {
    "indicator": "ma",
    "timeframe": TIMEFRAMES["execution"],
    "window": 7,
    "build_min_step_bars": 2,       # e.g. 3 bars on 4h => 12 hours
    "build_max_delta_frac": 0.25,   # position changes are smoothed (<=25% of full target per exec)
    "reduce_min_step_bars": 1,
    "reduce_max_delta_frac": 0.5,
}

RANGE: RangeCfg = {
    "enabled": True,
    "price_band_pct": 0.01,
    "ma_band_pct": 0.005,
    "slope_band_pct": 0.001,
}

# -----------------------------
# Risk / sizing
# -----------------------------

RiskMode = Literal["RISK_ON", "RISK_NEUTRAL", "RISK_OFF"]
DirectionMode = Literal["long_only", "short_only", "both_side"]

MAX_POSITION_FRAC: Dict[RiskMode, float] = {
    "RISK_ON": 1.0,
    "RISK_NEUTRAL": 0.5,
    "RISK_OFF": 0.0,  # reduce-only in strategy; target=0 means exit
}

# Directional bias for the strategy.
# - "long_only": only take long signals; shorts are treated as exit/flat
# - "short_only": only take short signals; longs are treated as exit/flat
# - "both_side": take both long and short signals (default)
DIRECTION_MODE: DirectionMode = "both_side"

# Backtest settings
STARTING_CASH_USDC_PER_SYMBOL: float = 10_000.0
TAKER_FEE_BPS: float = 0.0  # set to realistic value if you want (e.g., 3.5 bps => 3.5)

# Data / storage
DATA_DIR: str = "data"
MARKET_DATA_DIR: str = f"{DATA_DIR}/market_data"
BACKTEST_RESULT_DIR: str = f"{DATA_DIR}/backtest_result"

# Hyperliquid public API
HL_INFO_URL: str = "https://api.hyperliquid.xyz/info"

def fee_rate_from_bps(bps: float) -> float:
    return bps / 10_000.0
