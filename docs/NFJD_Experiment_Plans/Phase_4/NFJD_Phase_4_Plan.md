# NFJD Phase 4: 完整基准测试计划书

## 1. 阶段目标

在多种任务、m值、模型大小下全面对比NFJD与所有基线方法，生成最终性能报告。

## 2. 实验配置

| 维度 | 取值 |
|------|------|
| 方法 | NFJD, FedJD, FMGDA, WeightedSum, DirectionAvg |
| 数据集 | Synthetic Regression, HighConflict Regression, Classification |
| m值 | 2, 3, 5, 10 |
| 模型 | Small, Medium |
| Non-IID | 0.0, 0.3, 0.6, 0.9 (仅分类) |
| 种子 | 7, 42, 123 |

## 3. 实验矩阵

- 回归: 5方法 × 4个m × 2模型 × 3种子 = 120次
- 高冲突回归: 5方法 × 3个m × 3种子 = 45次
- 分类Non-IID: 5方法 × 2个m × 4 Non-IID × 3种子 = 120次

**总计: 285次实验**

## 4. 评估指标

| 指标 | 定义 | 用途 |
|------|------|------|
| NHV | 归一化超体积 | Pareto前沿质量 |
| NPG | 归一化Pareto Gap | 距理想点距离 |
| Avg RI | 平均相对改善率 | 收敛速度 |
| Upload/Client | 每客户端上传量 | 通信效率 |
| Round Time | 每轮时间 | 计算效率 |
| Total Time | 总训练时间 | 端到端效率 |

## 5. 通过标准

1. NFJD在 ≥ 80% 的配置中Avg RI优于FedJD
2. NFJD通信量在所有m值下均为d×4 B
3. NFJD在m=10时仍能正常工作（vs FedJD NHV=0.68）
4. 生成完整的对比表格和可视化图表
