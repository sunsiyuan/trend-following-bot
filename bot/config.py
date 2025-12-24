"""
Project config.

这个文件的定位：单一真相源（single source of truth），用于“把策略可调的旋钮集中在一起”。
不要把逻辑写进来：只放常量 + 少量 helper（例如 bps 转 fee rate）。

它控制的东西主要有：
1) 交易标的：哪些币、默认 quote（USDC）
2) 分层决策：每一层用哪个 timeframe + 哪个指标（MA / Donchian）
3) 风控与仓位：风险档位 -> 最大仓位
4) 回测：起始资金、手续费、数据与结果落盘目录
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, NotRequired, Optional, TypedDict

# -----------------------------
# Market / symbols
# -----------------------------

# 默认计价资产（你回测和持仓价值都用 USDC 来算）
QUOTE_ASSET: str = "USDC"

# 交易标的列表（perp 默认 coin 名：BTC/ETH/SOL...）
# 调参思路：
# - 先只跑 1 个（BTC），保证策略语义/统计口径/数据对齐没问题
# - 再扩到 ETH、SOL；最后才考虑山寨（因为噪声与跳空更高，range 判定更难）
SYMBOLS: List[str] = ["BTC", "ETH", "SOL"]

# 市场类型：v1 默认 perp
# spot 在 Hyperliquid 上 coin 命名/映射更复杂（通常需要 spotMeta 映射），所以先别碰
MARKET_TYPE: Literal["perp", "spot"] = "perp"

# -----------------------------
# Timeframes
# -----------------------------

# 这里定义“每个决策层用哪个粒度”的默认值
# trend: 用来判断方向/趋势质量（更慢、更稳）
# execution: 用来控制调仓节奏（更快、更敏感）
#
# 调参思路：
# - trend 用 1d 基本合理（抗噪）；你之前担心 3D/7D 太慢是对的
# - execution 用 4h 是个折中：比 1h 更稳、比 1d 更及时
TIMEFRAMES: Dict[str, str] = {
    "trend": "1d",
    "execution": "4h",
}

# -----------------------------
# Indicator layer configs
# -----------------------------

# 指标类型：目前只支持 ma / donchian
# 调参思路：先把“语义收敛 + 统计口径对齐”做好，再扩指标集合
IndicatorType = Literal["ma", "donchian"]
MAType = Literal["sma", "ema"]

class TrendExistenceCfg(TypedDict):
    # 趋势存在层：回答“现在应该偏多/偏空/还是不参与？”
    # 注意：你后来已经把语义收敛成 3 类（TREND_UP / TREND_DOWN / RANGE）
    # 那么这里的 indicator/window 影响的是“趋势信号出现的频率与稳定性”
    indicator: IndicatorType
    timeframe: str
    window: int
    ma_type: NotRequired[MAType]
    slope_k: NotRequired[int]

class TrendQualityCfg(TypedDict):
    # 趋势质量层：回答“最多可以给多大仓位？”
    # v1 固定用 MA（更容易解释与调参）
    # neutral_band_pct 是一个非常关键的“去抖动”参数：
    # - 越大：越容易判定为中性/不加仓（降低暴露、减少反复进出）
    # - 越小：越激进（暴露更高，但震荡期回撤/换手可能更差）
    indicator: Literal["ma"]
    timeframe: str
    window: int
    neutral_band_pct: float
    ma_type: NotRequired[MAType]

class ExecutionCfg(TypedDict):
    # 执行层：回答“今天/这一根执行K线，是否要调整仓位？调整多少？”
    # 这层的目标不是预测，而是“控制交易频率 + 平滑仓位变化”
    #
    # build_*：加仓相关
    # reduce_*：减仓相关（你已经把减仓允许更快更猛，这是合理的：止损/退出应该更果断）
    indicator: Literal["ma"]
    timeframe: str
    window: int
    ma_type: NotRequired[MAType]
    slope_k: NotRequired[int]

    # 关键：min_step_bars 相当于“动作冷却”，避免每根都动
    # - build_min_step_bars=2 且 execution=4h => 最快 8 小时才允许再加仓一次
    build_min_step_bars: int

    # 关键：max_delta_frac 限制单次目标仓位变化幅度（0..1）
    # - 0.25 表示：一次最多改变 25% 的“满仓目标”
    # - 用它来抑制过度敏感（尤其是你说“策略会不会太敏感”的核心解法之一）
    build_max_delta_frac: float

    reduce_min_step_bars: int
    reduce_max_delta_frac: float

class RangeCfg(TypedDict):
    # RANGE 判定层（是否把当前市场视为“震荡/无趋势”）
    # 注意：range 的存在非常容易导致“暴露不足”：
    # - 过于严格 => 大量时间被判为 range，长期持有拿不到大波段
    # - 过于宽松 => 趋势/震荡都在交易，回撤和换手会增大
    #
    # enabled：允许你快速做 A/B（range 开/关）
    enabled: bool

    # ma_fast/ma_slow：用于判断“是否缠绕/收敛/无趋势”的快慢均线
    # 调参建议：
    # - fast 通常和 TREND_EXISTENCE window 同级别（你现在是 15）
    # - slow 通常和 TREND_QUALITY window 同级别（你现在是 50）
    ma_fast_window: int
    ma_slow_window: int

    # price_band_pct：价格围绕某个基准（通常是 slow MA 或中轴）在多窄的区间内算震荡
    # - 越小：越容易判 trend（更少 range）
    # - 越大：越容易判 range（更少暴露）
    price_band_pct: float

    # ma_band_pct：快慢 MA 之间的距离小于多少算“均线缠绕”
    # - 这是 range 判定里最直接的“趋势强度” proxy
    ma_band_pct: float

    # slope_band_pct：均线斜率绝对值低于多少算“走平”（无趋势）
    # - 越小：需要更“平”才算 range（更激进）
    # - 越大：更容易判 range（更保守）
    slope_band_pct: float

# -----------------------------
# Layer defaults
# -----------------------------

# 趋势存在层默认：MA(1d, 15)
# 调参思路（很重要）：
# - window 小（比如 10-15）：更敏感，容易早进早出，震荡期会吃亏
# - window 大（比如 25-30）：更稳，更慢，可能错过早期反转，但通常更“趋势交易像样”
TREND_EXISTENCE: TrendExistenceCfg = {
    "indicator": "ma",  # "ma" or "donchian"
    "timeframe": TIMEFRAMES["trend"],
    "window": 15,
    "ma_type": "ema",
    "slope_k": 2,
}

# 趋势质量层默认：MA(1d, 50) + neutral band
# neutral_band_pct 是你现在“暴露 vs 回撤”的大旋钮之一：
# - 如果你觉得“暴露不够”：可以先把 band 降低一点（例如 1% -> 0.5%）
# - 如果你觉得“震荡期亏太多/反复进出”：提高 band（例如 1% -> 1.5%/2%）
TREND_QUALITY: TrendQualityCfg = {
    "indicator": "ma",
    "timeframe": TIMEFRAMES["trend"],
    "window": 50,             # e.g. 50 or 90（50 更灵敏；90 更慢更稳）
    "neutral_band_pct": 0.01, # +/-1% band around MA
}

# 执行层默认：MA(4h, 7) + 平滑仓位变化
# 调参思路：
# - window=7（4h）大概是 28 小时的平滑；你要更慢，就 9/12；更快就 5
# - build_max_delta_frac 决定“加仓速度”；reduce_max_delta_frac 决定“退出速度”
EXECUTION: ExecutionCfg = {
    "indicator": "ma",
    "timeframe": TIMEFRAMES["execution"],
    "window": 7,
    "ma_type": "ema",
    "slope_k": 1,

    # 加仓：更谨慎（减少追高/噪声），所以冷却更长、单次幅度更小
    "build_min_step_bars": 2,       # e.g. 3 bars on 4h => 12 hours（注：这里是 2 => 8h）
    "build_max_delta_frac": 0.25,   # <=25% of full target per exec

    # 减仓：更果断（避免回撤扩大），所以允许更快更大
    "reduce_min_step_bars": 1,
    "reduce_max_delta_frac": 0.5,
}

# RANGE：默认开启
# 你现在的主要任务之一其实就是“让 range 判定合理”，否则长期暴露不足
# 建议的调参顺序（非常实用）：
# 1) 先关掉 slope_band_pct，只用 ma_band_pct + price_band_pct 观察 range 天数/收益变化
# 2) 再逐步引入 slope_band_pct 做去噪
RANGE: RangeCfg = {
    "enabled": True,
    "ma_fast_window": 15,
    "ma_slow_window": 50,
    "price_band_pct": 0.02,
    "ma_band_pct": 0.012,
    "slope_band_pct": 0.0025,
}

# -----------------------------
# Risk / sizing
# -----------------------------

RiskMode = Literal["RISK_ON", "RISK_NEUTRAL", "RISK_OFF"]
DirectionMode = Literal["long_only", "short_only", "both_side"]

# 风险档位 -> 最大仓位比例
# 这是“趋势质量层”的输出映射表
# 调参思路：
# - 如果你觉得策略“仓位太满导致回撤大”：把 RISK_ON 从 1.0 降到 0.7/0.8
# - 如果你觉得策略“暴露不够”：先别动这个，优先排查 range/neutral 判定是否过严
MAX_POSITION_FRAC: Dict[RiskMode, float] = {
    "RISK_ON": 1.0,
    "RISK_NEUTRAL": 0.5,
    "RISK_OFF": 0.0,  # target=0 means exit（策略里一般应当 reduce-only 到 0）
}

# 方向模式：你当前为了贴近 BTC 大周期而用 long_only
# 调参思路（关键认知）：
# - long_only 会天然提升表现（因为长期上行偏置），但这不是“纯趋势策略的普适性证明”
# - both_side 才能检验“熊市/反转期是否能保护回撤”
# - 所以：你可以先 long_only 把 range/暴露问题调顺，再切 both_side 验证稳健性
DIRECTION_MODE: DirectionMode = "long_only"

# -----------------------------
# Backtest settings
# -----------------------------

# 单标的起始资金（如果多 symbol，你的回测会按 symbol 分开跑或合并跑，取决于 backtest 实现）
STARTING_CASH_USDC_PER_SYMBOL: float = 10_000.0

# 手续费（bps）
# 强烈建议尽早填真实值，不然 trade_count 很高时，回测会严重乐观
# 调参思路：先用 taker 费率做 worst-case，再考虑 maker/滑点模型
TAKER_FEE_BPS: float = 0.0  # e.g., 3.5 bps => 3.5

# -----------------------------
# Data / storage
# -----------------------------

# 数据落盘目录
DATA_DIR: str = "data"
MARKET_DATA_DIR: str = f"{DATA_DIR}/market_data"
BACKTEST_RESULT_DIR: str = f"{DATA_DIR}/backtest_result"

# Hyperliquid public API
HL_INFO_URL: str = "https://api.hyperliquid.xyz/info"

def fee_rate_from_bps(bps: float) -> float:
    # bps -> fraction
    # 10 bps = 0.1% = 0.001
    return bps / 10_000.0
