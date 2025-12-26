## 任务名称
{一句话：动词 + 对象 + 目的}

### 背景
- 现状：
- 痛点：
- 触发原因/上下文（关联 PR/issue/讨论链接）：

---

### 目标（P0）
- [ ] {必须达成的可验证目标 1}
- [ ] {必须达成的可验证目标 2}
- [ ] {必须达成的可验证目标 3}

---

### 关键决策
> 写“已拍板、不再争论”的点（每条要能转成工程约束）

- 决策 1：
- 决策 2：
- 决策 3：

---

### 实现方式
> 写“怎么做”，给到模块落点/接口契约/数据结构/文件路径

- 模块/文件落点：
- 新增/修改的核心函数与 I/O：
- 关键数据结构（JSON schema / dataclass）：
- 兼容性与迁移策略：
- 回滚方式（如有）：

---

### 非目标（不要做）
- 不做：
- 明确拒绝的增强：
- 不解决的边界问题：

---

### 交付物
- （每次必有）更新：`docs/PROJECT_MAP.md`、`docs/STRATEGY_CONTEXT.md`、`CHANGELOG.md`
- （每次必有）如涉及任何策略/函数结构性变动：`bot/strategy.py::STRATEGY_VERSION` 必须 bump
- （按需）新增/修改代码文件清单：
  - `...`

---

### 验收标准
- 命令/脚本可复现：
  - `python -m bot.backtest --start {YYYY-MM-DD} --end {YYYY-MM-DD} --symbols {SYMBOLS}`
- 输出检查点（必须列出“文件 + 字段 + 预期”）：
  - `data/backtest_result/{run_id}/summary_all.json`：包含 `{...}`
  - `data/backtest_result/runs.jsonl`：新增 1 条记录，包含 `{...}`
- 行为语义检查点：
  - {例如：start inclusive / end exclusive；run_id 确定性；覆盖策略}

---

## Codex 可执行 Prompt

```text
你在一个 Python repo 里工作。目标是实现下面“P0目标”，并满足“关键决策/约束”。

# 约束
- 不要做额外假设；一切以当前代码为准
- 不要引入重型框架/平台；保持最小侵入
- 不要添加不必要的 CLI 增强
- 除非本任务明确要求，否则不要新增 docs 文件（只更新指定 3 个）
- 如涉及策略结构性变更，必须 bump bot/strategy.py::STRATEGY_VERSION

# 已确定的工程决策（必须严格遵守）
- {决策 1}
- {决策 2}
- {决策 3}

# 需要你完成的工作（按顺序）
1) 代码改动：{……}
2) 本地自测：{……}
3) 结果与日志核对：{……}
4) 更新文档（必须做，且基于代码现状）：
   - 更新 docs/PROJECT_MAP.md（结构/调用链/输出）
   - 更新 docs/STRATEGY_CONTEXT.md（策略实现“契约”、公式、关键字段）
   - 更新 CHANGELOG.md（新增一条版本记录：变更点 + 影响范围）

# 自检/验收（你需要在本地跑通并确保输出）
- 运行一次（必须离线，使用 repo 内现成测试数据）：
  python -m bot.backtest --start 2024-12-01 --end 2025-12-20 --symbols BTCTEST

- 运行行为要求（必须满足）：
  1) BTCTEST 必须只读本地测试数据；不得发起任何远端数据下载/HTTP 请求
  2) 若测试数据缺失或时间覆盖不足：必须 fail-fast，并打印清晰报错（缺失文件/缺失时间段）

- 产物要求（把结果片段贴在 Codex 回复里，截断即可）：
  1) 输出目录：data/backtest_result/{run_id}/...
  2) summary_all.json：必须包含（字段名以代码现状为准）
     - run_id
     - param_hash（且 param_hash 的输入必须包含 strategy_version）
     - data_fingerprint（基于数据 manifest 生成）
     - data_coverage / manifest（至少能看到实际数据边界：actual_first_ts / actual_last_ts 或等价字段）
  3) data/backtest_result/runs.jsonl：必须 append 一条记录（允许重复 run_id），包含 run_id/param_hash/data_fingerprint/strategy_version（字段名以代码现状为准）

