# Overview / 总览

**中文**

本仓库核心代码集中在 `bot/` 目录，包含回测入口、策略逻辑、数据获取与指标计算等模块。本文仅基于仓库内真实代码梳理“项目结构与策略文件关系”，并标注每个关键结论的证据位置（文件路径 + 行号范围）。策略/回测均通过同一套 `strategy.decide` 逻辑执行，回测入口位于 `bot/backtest.py`，并输出结果到 `data/backtest_result/...`；新增排名脚本 `bot/rank_runs.py` 读取 `runs.jsonl` 与 run_dir 产出 `data/backtest_rank/{YYYYMMDDTHHMMSSZ}__{rank_id}/...`，评分始终按 `docs/KEY_METRICS.md` 公式计算，`mdd_pass` 为独立护栏字段、`mdd_score` 在 `-0.30` 到 `-0.60` 线性衰减，并额外输出 `base/mdd_score` 以便诊断。证据：`bot/backtest.py:L1-L13, L345-L723` 与 `bot/rank_runs.py:L1-L718`。

**English**

The core code lives in `bot/` and includes the backtest entry, strategy logic, data access, and metrics. This document maps structure and strategy file relationships based strictly on the repository’s actual code, with evidence for each key statement (path + line range). Both backtest and live flow call the shared `strategy.decide` logic; the backtest entry is `bot/backtest.py` and outputs results under `data/backtest_result/...`, while the ranking script `bot/rank_runs.py` reads `runs.jsonl` + run_dir outputs and emits `data/backtest_rank/{YYYYMMDDTHHMMSSZ}__{rank_id}/...`, always computing the score per `docs/KEY_METRICS.md` with `mdd_pass` as a separate guardrail field, `mdd_score` linearly decaying between `-0.30` and `-0.60`, and extra `base/mdd_score` diagnostics. Evidence: `bot/backtest.py:L1-L13, L345-L723` and `bot/rank_runs.py:L1-L718`.

# Repo Tree / 仓库树

**中文（仅展示与 backtest/signal/strategy/data/metrics/config/cli 相关的部分）**

```
/ (repo root)
├─ bot/                         # 策略、回测、数据与指标的核心实现
│  ├─ backtest.py                # 回测 CLI 入口 + 回测主流程 + 结果落盘
│  ├─ backtest_params.py         # 回测参数容器 + stable_json + param_hash
│  ├─ backtest_store.py          # 数据切片/manifest/fingerprint + runs.jsonl 写入
│  ├─ config.py                  # 参数/常量单一真相源（symbols、timeframes、费用等）
│  ├─ execution_policy.py        # 执行意图判定（trade vs NOOP_SMALL_DELTA）单一真源
│  ├─ data_client.py             # 数据下载/缓存/读取（Hyperliquid candleSnapshot）
│  ├─ indicators.py              # 指标计算（MA、Donchian、log_slope、hlc3 等）
│  ├─ metrics.py                 # 回测指标计算（收益、回撤、Sharpe、Ulcer 等）
│  ├─ quarterly_stats.py         # 按季度统计（基于 equity/trades 输出）
│  ├─ rank_runs.py               # 回测结果排名/对比（runs.jsonl -> backtest_rank）
│  ├─ strategy.py                # 策略逻辑（特征计算 + 决策）
│  └─ main.py                    # 最小 live runner（CLI 入口，输出最新决策）
├─ docs/                         # 文档（非回测流程本体）
│  └─ KEY_METRICS.md             # 排名/对比指标契约与公式
└─ data/                         # 数据与回测结果落盘目录（运行时生成）
   └─ backtest_rank/             # 排名结果输出目录（run 级比较）
```

证据：文件职责见各文件头部注释与导入关系，例如 `backtest.py` 的职责描述（`bot/backtest.py:L1-L13`）、`data_client.py` 的职责说明（`bot/data_client.py:L1-L19`）、`strategy.py` 的策略说明（`bot/strategy.py:L1-L16`）、`metrics.py` 的指标说明（`bot/metrics.py:L1-L5`）、`main.py` 的 live runner 说明（`bot/main.py:L1-L10`）。

**English (only backtest/signal/strategy/data/metrics/config/cli-relevant paths)**

```
/ (repo root)
├─ bot/                         # Core strategy/backtest/data/metrics modules
│  ├─ backtest.py                # Backtest CLI entry + main loop + outputs
│  ├─ backtest_params.py         # Backtest params container + stable_json + param_hash
│  ├─ backtest_store.py          # Data slicing/manifest/fingerprint + runs.jsonl append
│  ├─ config.py                  # Single source of truth for parameters
│  ├─ execution_policy.py        # Trade intent policy (trade vs NOOP_SMALL_DELTA)
│  ├─ data_client.py             # Data download/cache/read (Hyperliquid)
│  ├─ indicators.py              # Indicator math helpers
│  ├─ metrics.py                 # Backtest performance metrics
│  ├─ quarterly_stats.py         # Quarterly stats export
│  ├─ rank_runs.py               # Backtest run ranking (runs.jsonl -> backtest_rank)
│  ├─ strategy.py                # Strategy features + decision logic
│  └─ main.py                    # Minimal live runner (latest decision)
├─ docs/                         # Documentation (non-execution path)
│  └─ KEY_METRICS.md             # Ranking/compare metrics contract
└─ data/                         # Runtime data/result storage
   └─ backtest_rank/             # Ranking result outputs
```

Evidence: module-level docstrings and imports describe responsibilities (e.g., `bot/backtest.py:L1-L13`, `bot/data_client.py:L1-L19`, `bot/strategy.py:L1-L16`, `bot/metrics.py:L1-L5`, `bot/main.py:L1-L10`).

# Backtest Entry & Call Graph / 回测入口与调用链

**中文**

- **实际 CLI 入口**：`bot/backtest.py` 内的 `main()` 使用 `argparse` 定义 `--start/--end/--symbols/--run_id` 并在 `__main__` 下执行，因此实际运行入口为模块 `bot.backtest`（如 `python -m bot.backtest ...`）。证据：`main()` 与 `if __name__ == "__main__":`（`bot/backtest.py:L918-L967`）。
- **可调用核心入口**：`run_backtest(symbols, start, end, params, run_id=None)` 是可 import 的回测入口，返回 run 级别结果并追加 runs.jsonl。证据：`bot/backtest.py:L388-L536`。
- **主流程**：`main()` 解析参数 → 组装 `BacktestParams` → 调用 `run_backtest`；当传入多个 symbols 时，每个 symbol 会生成独立的 run_id/run_dir 并各自写入 `config_snapshot.json` 与 `summary_all.json`。证据：`bot/backtest.py:L918-L980`。
- **数据读取**：`run_backtest_for_symbol` 调用 `data_client.ensure_market_data` 拉取/加载趋势与执行时间框架数据（含缓存下载）。证据：`bot/backtest.py:L345-L365` 与 `bot/data_client.py:L308-L394`。
- **策略/信号计算**：特征计算通过 `strategy.prepare_features_1d` 与 `strategy.prepare_features_exec`，决策通过 `strategy.decide`。证据：`bot/backtest.py:L369-L507`、`bot/strategy.py:L102-L228, L353-L582`。
- **模拟执行/撮合**：回测内用 cash + position 模型更新仓位与费用，按执行 bar 迭代。证据：仓位/费用更新与 trade 记录写入（`bot/backtest.py:L488-L605`）。
- **指标计算**：调用 `metrics.compute_equity_metrics` 和 `metrics.build_buy_hold_curve` 生成策略与买入持有指标。证据：`bot/backtest.py:L617-L637`、`bot/metrics.py:L158-L217`。
- **结果落盘**：输出 `equity_by_day.csv`、`equity_by_day_bh.csv`、`trades.jsonl`、`summary.json`、`summary_all.json`、`config_snapshot.json`、`quarterly_stats.csv`。证据：`bot/backtest.py:L639-L770`、`bot/quarterly_stats.py:L253-L325`。

**English**

- **CLI entry**: `bot/backtest.py` defines `main()` with `argparse` arguments (`--start/--end/--symbols/--run_id`) and runs it under `__main__`, so the runnable entry is module `bot.backtest` (e.g., `python -m bot.backtest ...`). Evidence: `main()` + `if __name__ == "__main__"` (`bot/backtest.py:L918-L967`).
- **Callable entry**: `run_backtest(symbols, start, end, params, run_id=None)` is the importable backtest entrypoint and appends runs.jsonl. Evidence: `bot/backtest.py:L388-L536`.
- **Main flow**: `main()` parses args → builds `BacktestParams` → calls `run_backtest`; when multiple symbols are provided, each symbol produces its own run_id/run_dir and its own `config_snapshot.json` and `summary_all.json`. Evidence: `bot/backtest.py:L918-L980`.
- **Data read**: `run_backtest_for_symbol` calls `data_client.ensure_market_data` for trend/execution timeframes (cache + download). Evidence: `bot/backtest.py:L345-L365` and `bot/data_client.py:L308-L394`.
- **Strategy/signal**: features via `strategy.prepare_features_1d` & `strategy.prepare_features_exec`, decisions via `strategy.decide`. Evidence: `bot/backtest.py:L369-L507`, `bot/strategy.py:L102-L228, L353-L582`.
- **Execution simulation**: cash + position model updated per execution bar, including fees and trade records. Evidence: `bot/backtest.py:L488-L605`.
- **Metrics**: `metrics.compute_equity_metrics` and `metrics.build_buy_hold_curve`. Evidence: `bot/backtest.py:L617-L637`, `bot/metrics.py:L158-L217`.
- **Outputs**: `equity_by_day.csv`, `equity_by_day_bh.csv`, `trades.jsonl`, `summary.json`, `summary_all.json`, `config_snapshot.json`, `quarterly_stats.csv`. Evidence: `bot/backtest.py:L639-L770`, `bot/quarterly_stats.py:L253-L325`.

**调用链图 / Call Graph (mermaid)**

```mermaid
flowchart TD
    CLI[bot.backtest: main()] -->|parse args| RUN[run_backtest]
    RUN -->|slice/manifest| STORE[backtest_store.slice_bars + manifest]
    RUN -->|single-symbol loop| RUNSYM[run_backtest_for_symbol]
    RUNSYM -->|ensure_market_data| DATA[data_client.ensure_market_data]
    RUN -->|features| FEAT[strategy.prepare_features_1d / prepare_features_exec]
    RUN -->|decision| DEC[strategy.decide]
    DEC -->|target_pos_frac| POL[execution_policy.compute_trade_intent]
    POL -->|trade_intent| SIM[backtest cash+position simulation]
    SIM -->|equity series| MET[metrics.compute_equity_metrics + build_buy_hold_curve]
    MET --> OUT[summary.json / equity_by_day.csv / trades.jsonl]
    OUT --> Q[quarterly_stats.csv]
    RUN --> RUNS[runs.jsonl append]
```

证据：回测主流程、数据读取、特征计算、决策、回测执行、指标、输出对应函数（`bot/backtest.py:L345-L770`, `bot/data_client.py:L308-L394`, `bot/strategy.py:L102-L582`, `bot/metrics.py:L158-L217`, `bot/quarterly_stats.py:L253-L325`）。

# Strategy Map / 策略文件关系图

**中文**

## 策略相关模块与职责

- `bot/strategy.py`：策略核心逻辑（特征构造 + 目标仓位决策），输出 `target_pos_frac` 等字段；共享给 backtest 与 live runner。证据：模块说明与 `decide()` 输出定义（`bot/strategy.py:L1-L16, L353-L582`）。
- `bot/indicators.py`：纯数学指标（MA、Donchian、log_slope、HLC3、quantize）。证据：模块说明与函数定义（`bot/indicators.py:L1-L99`）。
- `bot/config.py`：层级配置（trend/execution 时间框架与窗口、方向模式、角度衰减等）。证据：配置定义（`bot/config.py:L40-L177`）。

## 结构件（层）对应实现位置

| 结构件 | 代码位置 | 输入 | 输出/作用 | 证据 |
| --- | --- | --- | --- | --- |
| 趋势存在层 (trend existence) | `strategy.prepare_features_1d` + `decide_trend_existence` | 1D 数据 (HLC3 / MA / Donchian) | `raw_dir` 方向（LONG/SHORT/None） | `bot/strategy.py:L102-L259` + 指标 `bot/indicators.py:L18-L74` |
| 趋势质量/对齐层 (trend quality / alignment) | `strategy.prepare_features_1d` 中的 `quality_*`、`align` 计算 | 1D 数据 + MA/log_slope/波动估计 | `align` (0..1) 调整目标仓位幅度 | `bot/strategy.py:L129-L200` + `bot/config.py:L160-L170` |
| 执行层 (execution layer) | `strategy.prepare_features_exec` + `execution_gate_mode` | 执行周期数据（close + exec_ma） | 是否允许执行（gate） | `bot/strategy.py:L219-L308` |
| 目标仓位 & 平滑 | `compute_desired_target_frac` + `smooth_target` | `fast_sign_eff`, `align`, `direction_mode` | `desired_pos_frac` 与 `target_pos_frac` | `bot/strategy.py:L310-L337, L538-L582` |
| 状态/冷却 | `StrategyState` + flip cooldown 逻辑 | 上次执行 bar + 目标侧 | 冷却期阻止反向直接翻转 | `bot/strategy.py:L81-L87, L412-L435` |

## 输入输出说明（面向交易员）

- **输入**：
  - 1D 特征表：`strategy.prepare_features_1d` 需要包含 `open/high/low/close` 等 OHLC，输出包含 `hlc3`、`trend_ma`、`quality_ma`、`fast_sign_raw/fast_sign_eff/slow_sign`、`align` 等用于决策的列。证据：`bot/strategy.py:L102-L216`。
  - 执行周期特征表：`strategy.prepare_features_exec` 生成 `exec_ma` 用于执行过滤。证据：`bot/strategy.py:L219-L228`。
- **输出**：
  - `strategy.decide` 产出包含 `desired_pos_frac`、`target_pos_frac`、`action`、`reason` 等字段，被回测与 live runner 直接消费。证据：`bot/strategy.py:L353-L582`，回测消费位置 `bot/backtest.py:L501-L605`。

**English**

## Strategy-related modules and responsibilities

- `bot/strategy.py`: core strategy logic (features + target position decision), outputs `target_pos_frac`, shared by backtest and live runner. Evidence: module description and `decide()` output (`bot/strategy.py:L1-L16, L353-L582`).
- `bot/indicators.py`: pure math indicators (MA, Donchian, log_slope, HLC3, quantize). Evidence: module description and function definitions (`bot/indicators.py:L1-L99`).
- `bot/config.py`: layer configs (timeframes/windows, direction mode, angle sizing, etc.). Evidence: config definitions (`bot/config.py:L40-L177`).

## Layer mapping to code

| Layer | Location | Input | Output/Effect | Evidence |
| --- | --- | --- | --- | --- |
| Trend existence | `strategy.prepare_features_1d` + `decide_trend_existence` | 1D data (HLC3/MA/Donchian) | `raw_dir` LONG/SHORT/None | `bot/strategy.py:L102-L259` + `bot/indicators.py:L18-L74` |
| Trend quality / alignment | `strategy.prepare_features_1d` (`quality_*`, `align`) | 1D data + MA/log_slope/vol | `align` (0..1) attenuates target | `bot/strategy.py:L129-L200` + `bot/config.py:L160-L170` |
| Execution layer | `strategy.prepare_features_exec` + `execution_gate_mode` | Execution timeframe data (close + exec_ma) | execution gate allow/deny | `bot/strategy.py:L219-L308` |
| Target sizing & smoothing | `compute_desired_target_frac` + `smooth_target` | `fast_sign_eff`, `align`, `direction_mode` | `desired_pos_frac` & `target_pos_frac` | `bot/strategy.py:L310-L337, L538-L582` |
| State / cooldown | `StrategyState` + flip cooldown logic | last exec bar + side | prevents immediate flip re-entry | `bot/strategy.py:L81-L87, L412-L435` |

## Inputs/Outputs (trader-facing)

- **Inputs**:
  - 1D features: `strategy.prepare_features_1d` expects OHLC and emits `hlc3`, `trend_ma`, `quality_ma`, `fast_sign_raw/fast_sign_eff/slow_sign`, `align`, etc. Evidence: `bot/strategy.py:L102-L216`.
  - Execution features: `strategy.prepare_features_exec` emits `exec_ma` for gating. Evidence: `bot/strategy.py:L219-L228`.
- **Outputs**:
  - `strategy.decide` returns `desired_pos_frac`, `target_pos_frac`, `action`, `reason`, etc., consumed by backtest/live. Evidence: `bot/strategy.py:L353-L582`, consumer in `bot/backtest.py:L501-L605`.

# Params Audit / 参数来源审计

**中文**

## 参数来源总览（来源类型）

- **配置常量**：集中在 `bot/config.py`（symbols/timeframes/层参数/费用/目录等）。证据：`bot/config.py:L23-L219`。
- **命令行参数**：回测 CLI 通过 `argparse` 提供 `--start/--end/--symbols/--run_id`。证据：`bot/backtest.py:L725-L754`。
- **环境变量**：仓库内未发现 `os.environ`/`getenv` 使用（已在 `bot/` 内搜索无结果）。证据：`rg -n "os\.environ|getenv" bot` 无匹配。
- **默认值/硬编码**：策略/回测中存在阈值与 eps（如 `1e-9`, `1e-12`）与窗口偏移，用于判断收敛/平滑/空仓。证据：`bot/strategy.py:L339-L350, L377-L440`；`bot/backtest.py:L320-L341, L389-L392`。

## 影响回测结果的参数清单（按类别）

### Strategy Params（策略参数）

- **趋势存在层**：`TREND_EXISTENCE`（indicator, window, ma_type, slope_k, timeframe, fast_state_deadband_pct）。证据：`bot/config.py:L110-L124` 与 `bot/strategy.py:L113-L127`。
- **趋势质量层/对齐**：`TREND_QUALITY` + `ANGLE_SIZING_*` + `VOL_*`。证据：`bot/config.py:L122-L170` 与 `bot/strategy.py:L129-L200`。
- **方向模式**：`DIRECTION_MODE` 控制 long/short/both。证据：`bot/config.py:L158-L177` 与 `bot/strategy.py:L310-L325`。

### Execution Params（执行参数）

- **执行层配置**：`EXECUTION` 中 `window/ma_type/build_min_step_bars/build_max_delta_frac/reduce_*`。证据：`bot/config.py:L133-L151` 与 `bot/strategy.py:L219-L228, L474-L503`。
- **执行过滤**：`execution_gate_mode` 使用 `exec_ma` 与 `min_step_bars` 实现节奏与趋势过滤。证据：`bot/strategy.py:L283-L308`。

### Cost Params（成本参数）

- **手续费**：`TAKER_FEE_BPS` 由 `fee_rate_from_bps` 转换后用于每次交易扣费。证据：`bot/config.py:L186-L219` 与 `bot/backtest.py:L379-L564`。
- **滑点/撮合模型**：回测没有滑点与限价模型（仅按执行 bar close 价格换仓）。证据：回测执行逻辑只用 `close` 价格与费率（`bot/backtest.py:L488-L605`）。

### Risk Params（风险参数）

- **已有**：方向模式 `DIRECTION_MODE`、角度衰减/对齐（`ANGLE_SIZING_*`）；风险/对齐参数已进入 `BacktestParams` 并参与 `param_hash`。证据：`bot/config.py:L158-L170`、`bot/backtest_params.py:L39-L82`、`bot/strategy.py:L102-L420`。
- **多空缩放**：`MAX_LONG_FRAC` / `MAX_SHORT_FRAC` 按符号缩放目标仓位（空头更保守）。证据：`bot/config.py:L179-L186` 与 `bot/strategy.py:L314-L343`。
- **未确认/未找到**：止损/止盈、最大回撤限制、风险状态机等在策略/配置/回测中没有实现（`risk_mode` 始终为 `None`）。证据：`bot/strategy.py:L33-L64, L353-L582`（`risk_mode=None`）。

### Data Params（数据参数）

- **数据源/接口**：`HL_INFO_URL`、`MARKET_TYPE`，并通过 `data_client` 调用 Hyperliquid `candleSnapshot`。证据：`bot/config.py:L200-L201` 与 `bot/data_client.py:L156-L215`。
- **时间粒度**：`TIMEFRAMES` 影响 `trend` 与 `execution` 数据读取。证据：`bot/config.py:L47-L50` 与 `bot/backtest.py:L355-L364`。
- **缓存路径**：`MARKET_DATA_DIR`、`BACKTEST_RESULT_DIR`。证据：`bot/config.py:L195-L198` 与 `bot/data_client.py:L106-L107`、`bot/backtest.py:L737-L739`。
- **数据窗口限制**：`HYPERLIQUID_KLINE_MAX_LIMIT` 与 earliest timestamp 限制。证据：`bot/config.py:L203-L214` 与 `bot/data_client.py:L76-L84, L320-L376`。

**English**

## Parameter sources (types)

- **Config constants**: centralized in `bot/config.py` (symbols/timeframes/layer params/fees/paths). Evidence: `bot/config.py:L23-L219`.
- **CLI args**: backtest `argparse` provides `--start/--end/--symbols/--run_id`. Evidence: `bot/backtest.py:L725-L754`.
- **Environment variables**: none found under `bot/` (searched `os.environ/getenv`, no hits). Evidence: `rg -n "os\.environ|getenv" bot` returned no matches.
- **Defaults / hardcoded thresholds**: epsilon thresholds and numeric cutoffs inside strategy/backtest (e.g., `1e-9`, `1e-12`) affect decisions and position logic. Evidence: `bot/strategy.py:L339-L350, L377-L440`; `bot/backtest.py:L320-L341, L389-L392`.

## Parameters affecting backtest results (by category)

### Strategy params

- **Trend existence**: `TREND_EXISTENCE` (indicator, window, ma_type, slope_k, timeframe). Evidence: `bot/config.py:L110-L120` and `bot/strategy.py:L113-L127`.
- **Trend quality / alignment**: `TREND_QUALITY` + `ANGLE_SIZING_*` + `VOL_*`. Evidence: `bot/config.py:L122-L170` and `bot/strategy.py:L129-L200`.
- **Direction mode**: `DIRECTION_MODE` controls long/short/both. Evidence: `bot/config.py:L158-L177` and `bot/strategy.py:L310-L325`.

### Execution params

- **Execution layer**: `EXECUTION` (`window/ma_type/build_min_step_bars/build_max_delta_frac/reduce_*`). Evidence: `bot/config.py:L133-L151` and `bot/strategy.py:L219-L228, L474-L503`.
- **Execution gating**: `execution_gate_mode` uses `exec_ma` and `min_step_bars`. Evidence: `bot/strategy.py:L283-L308`.

### Cost params

- **Fees**: `TAKER_FEE_BPS` converted via `fee_rate_from_bps` and applied per trade. Evidence: `bot/config.py:L186-L219` and `bot/backtest.py:L379-L564`.
- **Slippage/limit model**: none; trades use execution bar close price only. Evidence: execution logic uses `close` + fee (no slippage modeling) (`bot/backtest.py:L488-L605`).

### Risk params

- **Present**: `DIRECTION_MODE` and angle sizing (`ANGLE_SIZING_*`); the risk/alignment params now live in `BacktestParams` and are part of `param_hash`. Evidence: `bot/config.py:L158-L170`, `bot/backtest_params.py:L39-L82`, usage in strategy (`bot/strategy.py:L102-L420`).
- **Sign-based scaling**: `MAX_LONG_FRAC` / `MAX_SHORT_FRAC` scale the final target by direction (shorts more conservative). Evidence: `bot/config.py:L179-L186` and `bot/strategy.py:L314-L343`.
- **Not found / unconfirmed**: stop-loss/take-profit/max drawdown/risk state machine; `risk_mode` remains `None`. Evidence: `bot/strategy.py:L33-L64, L353-L582`.

### Data params

- **Source/API**: `HL_INFO_URL`, `MARKET_TYPE`, Hyperliquid `candleSnapshot` calls in `data_client`. Evidence: `bot/config.py:L200-L201`, `bot/data_client.py:L156-L215`.
- **Timeframes**: `TIMEFRAMES` for trend/execution. Evidence: `bot/config.py:L47-L50`, `bot/backtest.py:L355-L364`.
- **Paths**: `MARKET_DATA_DIR`, `BACKTEST_RESULT_DIR`. Evidence: `bot/config.py:L195-L198`, `bot/data_client.py:L106-L107`, `bot/backtest.py:L737-L739`.
- **Data window limits**: `HYPERLIQUID_KLINE_MAX_LIMIT` + earliest timestamp constraints. Evidence: `bot/config.py:L203-L214`, `bot/data_client.py:L76-L84, L320-L376`.

# Output & Reproducibility / 结果输出与可复现性

**中文**

## 结果输出位置

- `data/backtest_result/{run_id}/equity_by_day.csv`、`equity_by_day_bh.csv`、`trades.jsonl`、`summary.json`、`equity_by_day_with_benchmark.csv`（若含 `net_exposure`）、`quarterly_stats.csv`。证据：`bot/backtest.py:L706-L866` 与 `bot/quarterly_stats.py:L253-L325`。
- 运行级别输出：`config_snapshot.json`、`run_record.json`、`runs.jsonl`。证据：`bot/backtest.py:L918-L980` 与 `bot/backtest_store.py:L97-L115`。
- `quarterly_stats.csv` 主要字段包含 `pnl`、`return_annualized`、`sharpe_annualized` 等（`return_annualized` 为季度内首尾 equity 复利年化）。证据：`bot/quarterly_stats.py:L151-L223, L279-L325`。
- 排名输入：`runs.jsonl` + run_dir 下的 `summary.json`、`equity_by_day.csv`、`equity_by_day_bh.csv`、`trades.jsonl`（rank_runs 不再读取 `quarterly_stats.csv`）。证据：`bot/rank_runs.py:L118-L718`。
- 排名输出：`data/backtest_rank/{rank_id}/rank_results.json` 与 `rank_results.csv`（可选 `rank_spec.json` 复制）；`filters.param_hash_prefix` 允许逗号分隔或列表输入并归一为 list，用于稳定 hash。证据：`bot/rank_runs.py:L118-L718`。

## 可复现性检查

- **run_id 默认确定性**：`run_backtest` 使用 `param_hash` 与 `data_fingerprint` 组装默认 `run_id`（`{symbol}__{start}__{end}__{param_hash[:8]}__{data_fingerprint[:8]}`）。证据：`bot/backtest.py:L402-L477`。
- **参数哈希与数据指纹**：`BacktestParams.to_hashable_dict` + `calc_param_hash` 生成 `param_hash`，`backtest_store.build_data_manifest`/`calc_data_fingerprint` 生成 `data_fingerprint` 并写入 runs.jsonl；风险/对齐参数（`angle_sizing_*`, `vol_*`）已进入 `BacktestParams` 并参与 hash。证据：`bot/backtest_params.py:L1-L80`、`bot/backtest_store.py:L70-L115`、`bot/backtest.py:L402-L533`、`bot/strategy.py:L102-L420`。
- **strategy_version 来源**：`bot/strategy.py::STRATEGY_VERSION`，进入 `BacktestParams.to_hashable_dict` 从而影响 `param_hash` 与默认 `run_id`，并落盘到 `run_record.json`/`runs.jsonl`。证据：`bot/strategy.py:L27-L30`、`bot/backtest_params.py:L29-L56`、`bot/backtest.py:L502-L560`。
- **数据读取仍依赖实时网络**：`data_client` 通过 `requests.post` 调用 Hyperliquid API；若缓存不足将下载。证据：`bot/data_client.py:L156-L215, L380-L392`。
- **当前时间仍用于数据抓取的闭合K线过滤**：`fetch_candle_snapshot` 使用 `now_ms()` 过滤未收盘K线，但回测切片不再使用 `now()`。证据：`bot/data_client.py:L211-L229` 与 `bot/backtest_store.py:L28-L64`。
- **确定性写入**：交易文件输出前会先删除旧文件（保证文件内容与顺序由当前回测结果决定）。证据：`bot/backtest.py:L733-L739`。

## 最小改造切入点（仅列位置，不写改法）

- `bot/backtest.py::run_backtest_for_symbol`：当前回测核心入口，适合作为可调用/可复现核心函数的落点。证据：`bot/backtest.py:L345-L723`。
- `bot/backtest.py::main`：CLI 汇总参数与写入 `config_snapshot.json` 的位置，可在此将参数汇总并传给新的核心函数。证据：`bot/backtest.py:L725-L770`。
- `bot/data_client.py::ensure_market_data` / `fetch_candle_snapshot`：数据读取/下载的边界，若要固定数据指纹或禁用网络需在此控制。证据：`bot/data_client.py:L187-L394`。
- `bot/backtest.py` 输出段：所有结果文件落盘集中在该段落，适合作为结果存储接口扩展入口。证据：`bot/backtest.py:L639-L721`。

**English**

## Output locations

- `data/backtest_result/{run_id}/equity_by_day.csv`, `equity_by_day_bh.csv`, `trades.jsonl`, `summary.json`, `equity_by_day_with_benchmark.csv` (if `net_exposure`), `quarterly_stats.csv`. Evidence: `bot/backtest.py:L639-L721`, `bot/quarterly_stats.py:L253-L325`.
- Run-level outputs: `config_snapshot.json`, `run_record.json`, `runs.jsonl`. Evidence: `bot/backtest.py:L918-L980`.
- `quarterly_stats.csv` key fields include `pnl`, `return_annualized`, `sharpe_annualized` (with `return_annualized` as the compound annualized return over the quarter span). Evidence: `bot/quarterly_stats.py:L151-L223, L279-L325`.
- Ranking inputs: `runs.jsonl` plus per-run `summary.json`, `equity_by_day.csv`, `equity_by_day_bh.csv`, `trades.jsonl` (`rank_runs` no longer reads `quarterly_stats.csv`). Evidence: `bot/rank_runs.py:L118-L718`.
- Ranking outputs: `data/backtest_rank/{rank_id}/rank_results.json` and `rank_results.csv` (optional `rank_spec.json` copy); `filters.param_hash_prefix` accepts CSV or list input and is normalized to a list for stable hashing. Evidence: `bot/rank_runs.py:L118-L718`.

## Reproducibility audit

- **Default `run_id` uses current time** via `utc_now_compact()` (`datetime.now(timezone.utc)`). Evidence: `bot/backtest.py:L52-L53, L737-L739`.
- **Network-dependent data**: `data_client` calls Hyperliquid API and downloads if cache is missing. Evidence: `bot/data_client.py:L156-L215, L380-L392`.
- **Current time affects data windows**: `fetch_latest` / `fetch_candle_snapshot` use `now_ms()` and filter out unclosed candles. Evidence: `bot/data_client.py:L66-L67, L211-L229`.
- **Deterministic file write within a run**: existing trades file is removed before write. Evidence: `bot/backtest.py:L659-L663`.
- **Missing reproducibility metadata**: no param hash or data fingerprint recorded (only `config_snapshot.json` with selected fields). Evidence: `bot/backtest.py:L741-L754`.

## Minimal retrofit touchpoints (locations only)

- `bot/backtest.py::run_backtest_for_symbol`: core engine for a callable backtest entry. Evidence: `bot/backtest.py:L345-L723`.
- `bot/backtest.py::main`: CLI parameter collection + config snapshot point. Evidence: `bot/backtest.py:L725-L770`.
- `bot/data_client.py::ensure_market_data` / `fetch_candle_snapshot`: data download boundary for deterministic data control. Evidence: `bot/data_client.py:L187-L394`.
- Output block in `bot/backtest.py`: centralized result storage. Evidence: `bot/backtest.py:L639-L721`.

# Minimal Retrofit Summary / 改造能力落地（当前实现）

**中文**

- **可调用入口**：`run_backtest(symbols, start, end, params, run_id=None)` 已落地，并由 CLI 调用；参数集中于 `BacktestParams`。证据：`bot/backtest.py:L388-L536`、`bot/backtest_params.py:L1-L63`。
- **数据可复现元信息**：`backtest_store.build_data_manifest` 生成 manifest、`calc_data_fingerprint` 生成指纹并写入 `runs.jsonl`。证据：`bot/backtest_store.py:L70-L115`、`bot/backtest.py:L402-L533`。
- **输出路径保持**：仍使用 `run_dir = Path(config.BACKTEST_RESULT_DIR) / run_id` 且 per-symbol 输出路径不变。证据：`bot/backtest.py:L481-L866`。

**English**

- **Callable entry**: `run_backtest(symbols, start, end, params, run_id=None)` is now implemented and invoked by CLI; params are centralized in `BacktestParams`. Evidence: `bot/backtest.py:L388-L536`, `bot/backtest_params.py:L1-L63`.
- **Repro metadata**: `backtest_store.build_data_manifest` builds manifests, `calc_data_fingerprint` fingerprints them, and `runs.jsonl` is appended. Evidence: `bot/backtest_store.py:L70-L115`, `bot/backtest.py:L402-L533`.
- **strategy_version source**: `bot/strategy.py::STRATEGY_VERSION` is included in `BacktestParams.to_hashable_dict`, affecting `param_hash` and the default `run_id`, and is persisted to `run_record.json`/`runs.jsonl`. Evidence: `bot/strategy.py:L27-L30`, `bot/backtest_params.py:L29-L56`, `bot/backtest.py:L502-L560`.
- **Output placement preserved**: `run_dir = Path(config.BACKTEST_RESULT_DIR) / run_id` with existing per-symbol output paths unchanged. Evidence: `bot/backtest.py:L481-L866`.
