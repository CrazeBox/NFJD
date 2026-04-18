# NFJD Phase 1: 基线验证实验计划书

## 1. 阶段目标

验证NFJD核心链路（客户端本地多目标优化→Δθ上传→服务器FedAvg聚合→全局动量更新）在合成数据上可正确运行，且性能优于原FedJD。

## 2. 实验配置

| 参数 | 值 |
|------|-----|
| 数据集 | Synthetic Regression (m=2,3,5) |
| 客户端数 | 10 |
| 每客户端样本 | 100 |
| 参与率 | 0.5 |
| 学习率 | 0.01 |
| 本地轮次(E) | 3 |
| 全局轮次 | 50 |
| 随机种子 | 7, 42, 123 |
| 模型 | SmallRegressor |
| 全局动量β | 0.9 |
| 本地动量β | 0.9 |
| AdaptiveRescaling | 开启 |
| StochasticGramianSolver | 开启(subset_size=4) |

## 3. 对比方法

| 方法 | 描述 | 上传内容 |
|------|------|----------|
| NFJD | 新架构（完整功能） | Δθ (d维) |
| FedJD | 原架构（MinNorm方向） | Jacobian (m×d维) |
| FMGDA | 固定等权重Jacobian | Jacobian (m×d维) |
| WeightedSum | 本地加权梯度 | 梯度 (d维) |
| DirectionAvg | 本地Jacobian行均值 | 方向 (d维) |

## 4. 评估指标

| 指标 | 定义 | 期望 |
|------|------|------|
| NHV | 归一化超体积 | NFJD ≥ 基线 |
| Avg RI | 平均相对改善率 | NFJD > FedJD |
| Upload/Client | 每客户端上传量 | NFJD = d×4 B（与m无关） |
| All Decreased | 所有目标是否下降 | 100% |
| Rescale Factor | AdaptiveRescaling缩放因子 | 记录变化趋势 |

## 5. 实验矩阵

5方法 × 3个m值 × 3种子 = **45次实验**

## 6. 通过标准

1. NFJD在所有m值下Avg RI > FedJD
2. NFJD上传量与m无关（恒为d×4 B）
3. 所有实验无NaN/Inf
4. 同种子结果可复现
