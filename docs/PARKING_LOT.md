1. rank_run 补充 纯解释字段，加速排查
- base = E / UI_eff（现在没在 schema 里，但 debug 很有用）
- mdd_score（0~1）因为它能让你一眼看出：到底是 “E/UI 不行” 还是 “mdd 归零”。

2. 另外你提到的 eps_in = k_in * ATRrel：我检查了你上传的 STRATEGY_CONTEXT.md，确实没有 ATR/ATRrel 的定义；所以我把它降级为 P1 可选增强，P0 用 “pct → log1p” 的方式定义 eps，完全基于你现有上下文，不做额外假设。