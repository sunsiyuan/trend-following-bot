# Key Metrics / 关键指标

本文档定义回测对比/排名层使用的指标口径与公式，仅描述现有代码实现的契约。  
This document defines the metric contract for ranking/compare layer, based strictly on the current code semantics.

## 公式与定义 / Formulas & Definitions

**Scoring**

- `ui_floor = 0.05`
- `mdd_guard = -0.30`
- `gamma = 0.5`
- `UI_eff = max(UI, ui_floor)`
- `E = R_total - R_bh_total`
- `mdd_pass = (MDD > mdd_guard)` （注意：`MDD <= -0.30` 为 False）
- `mdd_score = clip(1 - abs(MDD) / abs(mdd_guard), 0, 1)`
- `final = (E / UI_eff) * (mdd_score ** gamma)`

**Rolling**

- `window_days = 180`
- `step_days = 60`
- 对齐方式：从 `equity_by_day.csv` 的 `actual_first_day` 开始滚动  
  Alignment: roll from the first day in `equity_by_day.csv`.
- 时间边界语义：`start inclusive / end exclusive`（rolling 切片同样遵循）  
  Boundary semantics: start inclusive, end exclusive (rolling windows follow the same rule).

**E (Excess Return)**

- `E = R_total - R_bh_total`

**Style profile (from trades.jsonl)**

- `in_position(t) = abs(position_frac_after) > 0`
- 使用 `equity_by_day.csv` 的日期作为日历轴，将 trades 按时间排序并将最新
  `position_frac_after` 应用到其后的日期（直到下一笔 trade 改变）。
- 计算：
  - `max_consecutive_days_in_position`：连续 True 的最长长度（天数）
  - `max_flat_streak_days`：连续 False 的最长长度（天数）
  - `avg_holding_period_days`：所有 True-run 的平均长度（天数）

## 指标字段清单 / Schema

### A) 全周期主指标与护栏
- `final`
- `E`
- `mdd_pass`

### B) Rolling（180d / 60d）
- `worst_final` (p0)
- `p25_final`
- `median_final` (p50)
- `p75_final`
- `best_final` (p100)
- `p0_E`
- `p25_E`
- `p50_E`
- `p75_E`
- `p100_E`
- `hit_rate`
- `window_count`

### C) 全周期明细画像（沿用 summary.json / quarterly_stats.csv 口径）
- `pnl`
- `return_annualized`
- `sharpe_annualized`
- `upi_annualized`
- `mdd`
- `ui`
- `avg_net_exposure`
- `pct_days_in_position`
- `pct_days_long`
- `pct_days_short`
- `trade_count`
- `start_equity`
- `end_equity`
- `fees_paid`
- `turnover_proxy`

### D) 交易风格画像（来自 trades.jsonl）
- `max_consecutive_days_in_position`
- `max_flat_streak_days`
- `avg_holding_period_days`
