# NFJD Phase 2: 消融实验计划书

## 1. 阶段目标

量化NFJD各组件（AdaptiveRescaling、双动量、StochasticGramianSolver、本地轮次E）的独立贡献。

## 2. 消融配置

| 消融组 | 配置 | 变量 |
|--------|------|------|
| A: AdaptiveRescaling | NFJD vs NFJD(no AR) | use_adaptive_rescaling |
| B: 全局动量 | NFJD(β=0.9) vs NFJD(β=0.0) | global_momentum_beta |
| C: 本地动量 | NFJD(β=0.9) vs NFJD(β=0.0) | local_momentum_beta |
| D: StochasticGramian | NFJD vs NFJD(no SG) | use_stochastic_gramian |
| E: 本地轮次 | E=1,3,5 | local_epochs |

## 3. 实验矩阵

- 消融A: 2配置 × m∈{2,5} × 3种子 = 18次
- 消融B: 2配置 × m∈{2,5} × 3种子 = 18次
- 消融C: 2配置 × m∈{2,5} × 3种子 = 18次
- 消融D: 2配置 × m∈{2,5} × 3种子 = 18次
- 消融E: 3配置 × m∈{2,5} × 3种子 = 18次

**总计: 90次实验**

## 4. 评估指标

| 指标 | 含义 |
|------|------|
| ΔRI | 消融前后Avg RI的变化 |
| ΔNHV | 消融前后NHV的变化 |
| Rescale Factor | 仅AdaptiveRescaling消融 |

## 5. 通过标准

1. AdaptiveRescaling贡献 > 3% RI提升
2. 全局动量贡献 > 2% RI提升
3. 本地轮次E=3为最优（E=1过少，E=5过多）
4. StochasticGramian在m=5时质量损失 < 5%
