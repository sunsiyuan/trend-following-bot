# 策略上下文 / Strategy Context

## 概览 / Overview
策略意图未在代码中明确声明；从命名与流程可推断为趋势跟随型框架，但此处不做额外假设，仅记录代码实现。  
Strategy intent is not explicitly declared in code; names and flow suggest a trend-following framework, but no further assumptions are made here beyond the implementation.

- 代码位置：bot/strategy.py::decide（关键变量：fast_sign, align, desired, target）  
- Code location: bot/strategy.py::decide (key vars: fast_sign, align, desired, target)
- 代码位置：bot/backtest.py::run_backtest_for_symbol（关键变量：target_frac, delta_notional, fee）  
- Code location: bot/backtest.py::run_backtest_for_symbol (key vars: target_frac, delta_notional, fee)
- 代码位置：bot/main.py::latest_decision_for_symbol（关键变量：df_1d_feat, df_ex_feat, decision）  
- Code location: bot/main.py::latest_decision_for_symbol (key vars: df_1d_feat, df_ex_feat, decision)

输入与输出的“契约”如下（仅基于代码）：输入为多时间尺度K线数据（含close/高低价等字段），输出为目标仓位比例（target_pos_frac，范围逻辑为-1..+1）与动作意图（HOLD/REBALANCE）。  
The input/output “contract” is as follows (code-only): inputs are multi-timeframe candles (including close/high/low fields), outputs are a target position fraction (target_pos_frac, logically -1..+1) and an action intent (HOLD/REBALANCE).

## Backtest Experiment Contract / 回测实验契约

**中文**

- **时间边界语义**：回测裁剪固定为 start inclusive、end exclusive（使用 `backtest_store.slice_bars` 基于 `open_ts`/`close_ts` 裁剪）。回测侧不允许 `now()` 裁剪。  
- **data_fingerprint（低配 manifest）**：每个 timeframe 生成 manifest，字段固定为 `tf`, `requested_start_ts`, `requested_end_ts`, `actual_first_ts`, `actual_last_ts`, `row_count`, `expected_row_count`，并通过 `stable_json` + `sha256` 生成 `data_fingerprint`。  
- **param_hash**：`BacktestParams.to_hashable_dict` 只包含影响结果的参数；`stable_json` 后 sha256 得到 `param_hash`。  
- **run_id 格式**：默认 `{symbol}__{start}__{end}__{param_hash[:8]}__{data_fingerprint[:8]}`，CLI 可用 `--run_id` 覆盖。  

**English**

- **Time boundary semantics**: slicing is start-inclusive, end-exclusive via `backtest_store.slice_bars` using `open_ts`/`close_ts`. No `now()` trimming in backtest slicing.  
- **data_fingerprint (minimal manifest)**: each timeframe emits a manifest with fields `tf`, `requested_start_ts`, `requested_end_ts`, `actual_first_ts`, `actual_last_ts`, `row_count`, `expected_row_count`, hashed via `stable_json` + `sha256`.  
- **param_hash**: `BacktestParams.to_hashable_dict` includes only result-affecting params; `stable_json` + sha256 yields `param_hash`.  
- **run_id format**: default is `{symbol}__{start}__{end}__{param_hash[:8]}__{data_fingerprint[:8]}`, overridable via CLI `--run_id`.  

## 数据与时间粒度 / Data & Timeframe
策略计算使用的字段包括：close（所有路径），high/low（HLC3与Donchian路径），其它字段如open/volume/trades未在信号计算中使用。  
Fields used in strategy calculations include: close (all paths), high/low (HLC3 and Donchian paths); other fields like open/volume/trades are not used in signal calculations.

- 代码位置：bot/strategy.py::prepare_features_1d（关键变量：hlc3, trend_ma, quality_ma）  
- Code location: bot/strategy.py::prepare_features_1d (key vars: hlc3, trend_ma, quality_ma)
- 代码位置：bot/indicators.py::hlc3（关键变量：high, low, close）  
- Code location: bot/indicators.py::hlc3 (key vars: high, low, close)

时间粒度在配置中固定为trend=1d、execution=4h；代码中未出现重采样逻辑，直接拉取各粒度数据并在各自粒度上计算特征。  
Timeframes are fixed in config as trend=1d and execution=4h; there is no resampling in code, and features are computed directly on each timeframe’s data.

- 代码位置：bot/config.py::TIMEFRAMES（关键变量：trend, execution）  
- Code location: bot/config.py::TIMEFRAMES (key vars: trend, execution)
- 代码位置：bot/data_client.py::ensure_market_data（关键变量：interval, start_ms, end_ms）  
- Code location: bot/data_client.py::ensure_market_data (key vars: interval, start_ms, end_ms)

预热与缺失历史处理：MA与Donchian使用min_periods=window产生前置NaN；log_slope使用shift(k)产生k根NaN；对齐度（align）若检测到任一关键字段NaN则强制为1.0；若缺少对应时间点数据则决策返回HOLD且理由为insufficient_data。  
Warmup/insufficient history handling: MA/Donchian use min_periods=window yielding leading NaNs; log_slope uses shift(k) producing k NaNs; alignment (align) is forced to 1.0 if any key fields are NaN; if required rows are missing, decision returns HOLD with reason=insufficient_data.

- 代码位置：bot/indicators.py::moving_average（关键变量：min_periods, window）  
- Code location: bot/indicators.py::moving_average (key vars: min_periods, window)
- 代码位置：bot/indicators.py::log_slope（关键变量：shift, k）  
- Code location: bot/indicators.py::log_slope (key vars: shift, k)
- 代码位置：bot/strategy.py::prepare_features_1d（关键变量：align, nan_mask）  
- Code location: bot/strategy.py::prepare_features_1d (key vars: align, nan_mask)
- 代码位置：bot/strategy.py::decide（关键变量：reason, insufficient_data）  
- Code location: bot/strategy.py::decide (key vars: reason, insufficient_data)

## 指标层 / Indicators
价格代理与趋势均在HLC3上计算（high/low缺失时回退到close）。  
Price proxy and trends are computed on HLC3 (fallback to close when high/low are missing).

- 公式：hlc3 = (high + low + close) / 3；若high或low缺失则hlc3 = close  
- Formula: hlc3 = (high + low + close) / 3; if high or low missing then hlc3 = close
- 代码位置：bot/indicators.py::hlc3（关键变量：high, low, close）  
- Code location: bot/indicators.py::hlc3 (key vars: high, low, close)

趋势存在（Trend Existence）为MA或Donchian之一，MA路径计算log_slope；Donchian路径仅生成上下轨用于方向判断。  
Trend existence is either MA or Donchian; MA path computes log_slope, Donchian path only builds upper/lower bands for direction checks.

- 公式：trend_ma = MA(hlc3, window, ma_type)  
- Formula: trend_ma = MA(hlc3, window, ma_type)
- 公式：trend_log_slope = (ln(trend_ma_t) - ln(trend_ma_{t-slope_k})) / slope_k  
- Formula: trend_log_slope = (ln(trend_ma_t) - ln(trend_ma_{t-slope_k})) / slope_k
- 代码位置：bot/strategy.py::prepare_features_1d（关键变量：trend_ma, trend_log_slope）  
- Code location: bot/strategy.py::prepare_features_1d (key vars: trend_ma, trend_log_slope)

趋势质量（Trend Quality）固定为MA，并使用相同的log_slope定义。  
Trend quality is fixed to MA and uses the same log_slope definition.

- 公式：quality_ma = MA(hlc3, window, ma_type)  
- Formula: quality_ma = MA(hlc3, window, ma_type)
- 公式：quality_log_slope = (ln(quality_ma_t) - ln(quality_ma_{t-slope_k})) / slope_k  
- Formula: quality_log_slope = (ln(quality_ma_t) - ln(quality_ma_{t-slope_k})) / slope_k
- 代码位置：bot/strategy.py::prepare_features_1d（关键变量：quality_ma, quality_log_slope）  
- Code location: bot/strategy.py::prepare_features_1d (key vars: quality_ma, quality_log_slope)

收益率与波动率代理（仅当趋势存在与质量均为MA时计算）：  
Return/volatility proxies (computed only when both trend existence and quality are MA-based):

- 公式：logret_t = ln(hlc3_t) - ln(hlc3_{t-1})  
- Formula: logret_t = ln(hlc3_t) - ln(hlc3_{t-1})
- 公式：sigma_price = rolling_std(logret, n_vol, min_periods=n_vol)  
- Formula: sigma_price = rolling_std(logret, n_vol, min_periods=n_vol)
- 公式：n_vol = vol_window_from_fast_window(w_fast)  
- Formula: n_vol = vol_window_from_fast_window(w_fast)
- 代码位置：bot/strategy.py::prepare_features_1d（关键变量：logret, sigma_price, n_vol）  
- Code location: bot/strategy.py::prepare_features_1d (key vars: logret, sigma_price, n_vol)
- 代码位置：bot/config.py::vol_window_from_fast_window（关键变量：VOL_WINDOW_DIV, VOL_WINDOW_MIN, VOL_WINDOW_MAX）  
- Code location: bot/config.py::vol_window_from_fast_window (key vars: VOL_WINDOW_DIV, VOL_WINDOW_MIN, VOL_WINDOW_MAX)

量化（quantization）函数用于z的向零量化：  
Quantization function used for z is toward-zero: 

- 公式：quantize_toward_zero(x, q) = sign(x) * floor(abs(x)/q) * q  
- Formula: quantize_toward_zero(x, q) = sign(x) * floor(abs(x)/q) * q
- 代码位置：bot/indicators.py::quantize_toward_zero（关键变量：x, q）  
- Code location: bot/indicators.py::quantize_toward_zero (key vars: x, q)

## 对齐度 align / Alignment
对齐度在MA/MA路径中定义为“斜率不匹配惩罚”的tanh衰减；若任何关键输入为NaN则align强制为1.0。  
Alignment is defined on the MA/MA path as a tanh attenuation of slope mismatch penalty; if any key inputs are NaN, align is forced to 1.0.

- 公式：delta = trend_log_slope - quality_log_slope  
- Formula: delta = trend_log_slope - quality_log_slope
- 公式：alpha_f = 2/(w_fast+1), alpha_s = 2/(w_slow+1)  
- Formula: alpha_f = 2/(w_fast+1), alpha_s = 2/(w_slow+1)
- 公式：vf = alpha_f/(2-alpha_f) + alpha_s/(2-alpha_s)  
- Formula: vf = alpha_f/(2-alpha_f) + alpha_s/(2-alpha_s)
- 公式：sigma_mismatch_mean = sigma_price * sqrt(vf) / sqrt(slope_k)  
- Formula: sigma_mismatch_mean = sigma_price * sqrt(vf) / sqrt(slope_k)
- 公式：z = delta / max(sigma_mismatch_mean, VOL_EPS)  
- Formula: z = delta / max(sigma_mismatch_mean, VOL_EPS)
- 公式：zq = quantize_toward_zero(z, ANGLE_SIZING_Q)  
- Formula: zq = quantize_toward_zero(z, ANGLE_SIZING_Q)
- 公式：fast_state = ln(hlc3) - ln(trend_ma)  
- Formula: fast_state = ln(hlc3) - ln(trend_ma)
- 公式：slow_state = ln(hlc3) - ln(quality_ma)  
- Formula: slow_state = ln(hlc3) - ln(quality_ma)
- 公式：fast_sign = 1 if fast_state>0; -1 if fast_state<0; 0 if fast_state==0; NaN if fast_state is NaN  
- Formula: fast_sign = 1 if fast_state>0; -1 if fast_state<0; 0 if fast_state==0; NaN if fast_state is NaN
- 公式：slow_sign = 1 if slow_state>0; -1 if slow_state<0; 0 if slow_state==0; NaN if slow_state is NaN  
- Formula: slow_sign = 1 if slow_state>0; -1 if slow_state<0; 0 if slow_state==0; NaN if slow_state is NaN
- 公式：z_dir = slow_sign * z  
- Formula: z_dir = slow_sign * z
- 公式：penalty = max(0, -z_dir)  
- Formula: penalty = max(0, -z_dir)
- 公式：penalty_q = floor(penalty/ANGLE_SIZING_Q) * ANGLE_SIZING_Q  
- Formula: penalty_q = floor(penalty/ANGLE_SIZING_Q) * ANGLE_SIZING_Q
- 公式：align = 1 - tanh(penalty_q/ANGLE_SIZING_A); align = clip(align, 0, 1)  
- Formula: align = 1 - tanh(penalty_q/ANGLE_SIZING_A); align = clip(align, 0, 1)
- 代码位置：bot/strategy.py::prepare_features_1d（关键变量：delta, sigma_mismatch_mean, z, zq, fast_state, slow_state, fast_sign, slow_sign, align）  
- Code location: bot/strategy.py::prepare_features_1d (key vars: delta, sigma_mismatch_mean, z, zq, fast_state, slow_state, fast_sign, slow_sign, align)

NaN覆盖条件（满足任一则align=1.0）：hlc3, trend_ma, quality_ma, trend_log_slope, quality_log_slope, sigma_price, sigma_mismatch_mean, z, fast_state, slow_state, fast_sign, slow_sign。  
NaN override conditions (any triggers align=1.0): hlc3, trend_ma, quality_ma, trend_log_slope, quality_log_slope, sigma_price, sigma_mismatch_mean, z, fast_state, slow_state, fast_sign, slow_sign.

- 代码位置：bot/strategy.py::prepare_features_1d（关键变量：nan_mask, align）  
- Code location: bot/strategy.py::prepare_features_1d (key vars: nan_mask, align)

若趋势存在或质量不为MA，则上述align链条不计算，相关字段设为NaN且align强制为1.0。  
If trend existence or quality is not MA-based, the align chain is skipped, related fields are NaN, and align is forced to 1.0.

- 代码位置：bot/strategy.py::prepare_features_1d（关键变量：align, z, fast_sign）  
- Code location: bot/strategy.py::prepare_features_1d (key vars: align, z, fast_sign)

## 趋势方向与状态 / Trend Direction & State
趋势方向（实际用于仓位方向）由fast_sign决定；fast_sign来源于fast_state的符号，market_state在决策中等于fast_dir。  
Trend direction (used for position direction) is decided by fast_sign; fast_sign comes from the sign of fast_state, and market_state is set to fast_dir in decision logic.

- 代码位置：bot/strategy.py::prepare_features_1d（关键变量：fast_state, fast_sign）  
- Code location: bot/strategy.py::prepare_features_1d (key vars: fast_state, fast_sign)
- 代码位置：bot/strategy.py::decide（关键变量：fast_sign, fast_dir, market_state）  
- Code location: bot/strategy.py::decide (key vars: fast_sign, fast_dir, market_state)

趋势存在方向（raw_dir）由trend_log_slope或Donchian上下轨决定，作为诊断记录；slow_sign/slow_dir来自quality_ma的log_slope。  
Trend-existence direction (raw_dir) is determined by trend_log_slope or Donchian bands and used for diagnostics; slow_sign/slow_dir come from quality_ma log_slope.

- 代码位置：bot/strategy.py::decide_trend_existence（关键变量：trend_log_slope, trend_upper, trend_lower）  
- Code location: bot/strategy.py::decide_trend_existence (key vars: trend_log_slope, trend_upper, trend_lower)
- 代码位置：bot/strategy.py::decide_slow_dir（关键变量：quality_log_slope）  
- Code location: bot/strategy.py::decide_slow_dir (key vars: quality_log_slope)

“fast/slow”的含义在代码中仅体现为两组不同窗口MA与其log_slope/状态；未发现进一步语义定义。  
The meaning of “fast/slow” in code is only reflected as two MA windows and their slopes/states; no further semantic definition is present.

- 代码位置：bot/strategy.py::prepare_features_1d（关键变量：trend_ma, quality_ma）  
- Code location: bot/strategy.py::prepare_features_1d (key vars: trend_ma, quality_ma)

## 仓位生成 / Position Targeting
目标仓位分数由fast_sign与align决定，并受方向模式限制。  
Target position fraction is derived from fast_sign and align, with direction mode constraints.

- 公式：desired = 0 if fast_sign is NaN or 0  
- Formula: desired = 0 if fast_sign is NaN or 0
- 公式（both_side）：desired = fast_sign * align  
- Formula (both_side): desired = fast_sign * align
- 公式（long_only）：desired = align if fast_sign>0 else 0  
- Formula (long_only): desired = align if fast_sign>0 else 0
- 公式（short_only）：desired = -align if fast_sign<0 else 0  
- Formula (short_only): desired = -align if fast_sign<0 else 0
- 代码位置：bot/strategy.py::compute_desired_target_frac（关键变量：fast_sign, align, direction_mode）  
- Code location: bot/strategy.py::compute_desired_target_frac (key vars: fast_sign, align, direction_mode)

目标值在执行层被平滑：每次最多改变max_delta，且构建/减仓使用不同max_delta。  
Targets are smoothed at execution: each step changes by at most max_delta, with different max_delta for build vs reduction.

- 公式：target = current + clamp(desired-current, -max_delta, +max_delta)  
- Formula: target = current + clamp(desired-current, -max_delta, +max_delta)
- 代码位置：bot/strategy.py::smooth_target（关键变量：current, desired, max_delta）  
- Code location: bot/strategy.py::smooth_target (key vars: current, desired, max_delta)

## 执行与调仓 / Execution & Rebalance
执行门控包括最小间隔（min_step_bars）与执行MA过滤（仅在加仓时启用）；减仓不要求趋势过滤。  
Execution gating includes a minimum spacing (min_step_bars) and an execution MA filter (enabled only for build); reductions skip the trend filter.

- 公式：allowed = (exec_bar_idx - last_exec_bar_idx) >= min_step_bars  
- Formula: allowed = (exec_bar_idx - last_exec_bar_idx) >= min_step_bars
- 公式（趋势过滤）：LONG需close > exec_ma；SHORT需close < exec_ma  
- Formula (trend filter): LONG requires close > exec_ma; SHORT requires close < exec_ma
- 代码位置：bot/strategy.py::execution_gate_mode（关键变量：exec_bar_idx, last_exec_bar_idx, exec_ma）  
- Code location: bot/strategy.py::execution_gate_mode (key vars: exec_bar_idx, last_exec_bar_idx, exec_ma)

翻转冷却（hysteresis）：若当前持仓方向与目标方向相反，则先强制目标=0，随后在build_min_step_bars内禁止同向重新进入。  
Flip cooldown (hysteresis): if current side and desired side oppose, target is forced to 0, and re-entry on that side is blocked for build_min_step_bars.

- 代码位置：bot/strategy.py::decide（关键变量：flip_block_until_exec_bar_idx, flip_blocked_side）  
- Code location: bot/strategy.py::decide (key vars: flip_block_until_exec_bar_idx, flip_blocked_side)

回测执行模型为“按执行K线收盘价立即成交”；未见限价/滑点模型。手续费按成交名义金额的固定比例扣除。  
Backtest execution is “fill at execution bar close price”; no limit/slippage model is present. Fees are applied as a fixed fraction of traded notional.

- 公式：target_notional = target_frac * equity_before  
- Formula: target_notional = target_frac * equity_before
- 公式：delta_notional = target_notional - current_notional  
- Formula: delta_notional = target_notional - current_notional
- 公式：fee = abs(delta_notional) * fee_rate  
- Formula: fee = abs(delta_notional) * fee_rate
- 代码位置：bot/backtest.py::run_backtest_for_symbol（关键变量：target_notional, delta_notional, fee）  
- Code location: bot/backtest.py::run_backtest_for_symbol (key vars: target_notional, delta_notional, fee)

## 风险控制 / Risk Controls
代码中明确的风险控制仅体现为方向模式限制、对齐度衰减、以及执行层节奏/步长限制；未发现独立的止损、回撤制动或波动率目标控制。  
Explicit risk controls in code are limited to direction mode constraints, alignment attenuation, and execution pacing/step limits; no standalone stop-loss, drawdown brake, or volatility targeting is found.

- 代码位置：bot/config.py::DIRECTION_MODE（关键变量：direction_mode）  
- Code location: bot/config.py::DIRECTION_MODE (key vars: direction_mode)
- 代码位置：bot/strategy.py::compute_desired_target_frac（关键变量：direction_mode）  
- Code location: bot/strategy.py::compute_desired_target_frac (key vars: direction_mode)
- 代码位置：bot/strategy.py::decide（关键变量：build_min_step_bars, reduce_min_step_bars, max_delta）  
- Code location: bot/strategy.py::decide (key vars: build_min_step_bars, reduce_min_step_bars, max_delta)

未找到的内容（已查找文件/符号）：风控状态机、止损/止盈、最大回撤限制、波动率目标；已查找 bot/strategy.py（risk_mode字段为None）、bot/config.py（无stop/drawdown参数）、bot/backtest.py（无额外风险规则）。  
Not found in code (files/symbols searched): risk state machine, stop-loss/take-profit, max drawdown limits, volatility targeting; searched bot/strategy.py (risk_mode field is None), bot/config.py (no stop/drawdown params), bot/backtest.py (no extra risk rules).

## 回测与统计 / Backtest & Metrics
回测账户模型为现金+持仓数量，盈亏通过平均成本与持仓数量计算；多空翻转时先平仓再按新价格开仓。  
The backtest account model is cash + position quantity; PnL uses average cost and position size; on flips it closes then opens at the new price.

- 公式：equity = cash + qty * price  
- Formula: equity = cash + qty * price
- 公式：realized_pnl = closed_qty * (price - avg_entry) * sign(q0)  
- Formula: realized_pnl = closed_qty * (price - avg_entry) * sign(q0)
- 代码位置：bot/backtest.py::update_avg_and_realized（关键变量：q0, avg0, dq, price）  
- Code location: bot/backtest.py::update_avg_and_realized (key vars: q0, avg0, dq, price)

日频指标：total_return、max_drawdown、sharpe_ratio、ulcer_index、ulcer_performance_index。  
Daily metrics: total_return, max_drawdown, sharpe_ratio, ulcer_index, ulcer_performance_index.

- 公式：total_return = ending_equity / starting_cash - 1  
- Formula: total_return = ending_equity / starting_cash - 1
- 公式：max_drawdown = min(equity / cummax(equity) - 1)  
- Formula: max_drawdown = min(equity / cummax(equity) - 1)
- 公式：daily_return = pct_change(equity)  
- Formula: daily_return = pct_change(equity)
- 公式：sharpe = mean(daily_return - rf_daily) / std(daily_return - rf_daily) * sqrt(365)  
- Formula: sharpe = mean(daily_return - rf_daily) / std(daily_return - rf_daily) * sqrt(365)
- 公式：ulcer_index = sqrt(mean(drawdown^2)); drawdown = equity/cummax(equity) - 1  
- Formula: ulcer_index = sqrt(mean(drawdown^2)); drawdown = equity/cummax(equity) - 1
- 公式：ulcer_performance_index = total_return / ulcer_index（ui为0时按代码分支处理）  
- Formula: ulcer_performance_index = total_return / ulcer_index (ui==0 handled by code branches)
- 代码位置：bot/metrics.py::compute_equity_metrics（关键变量：total_return, max_drawdown, sharpe_ratio, ulcer_index, ulcer_performance_index）  
- Code location: bot/metrics.py::compute_equity_metrics (key vars: total_return, max_drawdown, sharpe_ratio, ulcer_index, ulcer_performance_index)

季度统计指标使用equity_returns_with_first_zero与sharpe_annualized_from_returns，并计算净敞口相关指标与turnover_proxy。  
Quarterly stats use equity_returns_with_first_zero and sharpe_annualized_from_returns, and compute net exposure diagnostics plus turnover_proxy.

- 公式：returns[0]=0；returns[t]=equity[t]/equity[t-1]-1  
- Formula: returns[0]=0; returns[t]=equity[t]/equity[t-1]-1
- 公式：turnover_proxy = sum_{t=1..T} abs(net_exposure_t - net_exposure_{t-1})  
- Formula: turnover_proxy = sum_{t=1..T} abs(net_exposure_t - net_exposure_{t-1})
- 代码位置：bot/quarterly_stats.py::_build_row（关键变量：returns, turnover_proxy, net_exposure）  
- Code location: bot/quarterly_stats.py::_build_row (key vars: returns, turnover_proxy, net_exposure)

## 已知限制与未决问题 / Known Limitations & Open Questions
未在策略逻辑中使用的配置项：neutral_band_pct在TREND_QUALITY中定义，但代码未读取或应用。  
Unused config in strategy logic: neutral_band_pct is defined in TREND_QUALITY but is not read or applied in code.

- 代码位置：bot/config.py::TREND_QUALITY（关键变量：neutral_band_pct）  
- Code location: bot/config.py::TREND_QUALITY (key vars: neutral_band_pct)
- 代码位置：bot/strategy.py::prepare_features_1d（关键变量：TREND_QUALITY读取路径）  
- Code location: bot/strategy.py::prepare_features_1d (key vars: TREND_QUALITY access)

风险状态字段risk_mode始终为None；代码未说明其用途或未来计划。  
The risk_mode field is always None; code does not define its purpose or future plan.

- 代码位置：bot/strategy.py::decide（关键变量：risk_mode）  
- Code location: bot/strategy.py::decide (key vars: risk_mode)

log_slope要求输入为正数；代码未在调用前检查hlc3或MA为正，依赖价格数据为正数这一外部假设。  
log_slope requires positive inputs; code does not check hlc3 or MA positivity before calling, relying on the external assumption that prices are positive.

- 代码位置：bot/indicators.py::log_slope（关键变量：arr<=0检查）  
- Code location: bot/indicators.py::log_slope (key vars: arr<=0 check)
- 代码位置：bot/strategy.py::prepare_features_1d（关键变量：trend_ma, quality_ma）  
- Code location: bot/strategy.py::prepare_features_1d (key vars: trend_ma, quality_ma)
