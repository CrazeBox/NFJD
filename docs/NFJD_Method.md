# NFJD: Normalized Federated Jacobian Descent

## 1. 概述

NFJD（Normalized Federated Jacobian Descent）是对FedJD的全面架构改进，解决三个核心缺陷：

| 缺陷 | 改进方案 | 章节 |
|------|----------|------|
| 收敛速度慢 | AdaptiveRescaling + 双动量 | §2.5, §2.6 |
| 通信开销大 | Δθ上传 + 对齐分数 | §2.4, §3 |
| 计算效率低 | 单次backward + λ*复用 | §2.7, §2.2 |
| 高m可扩展性 | StochasticGramianSolver | §2.3 |

---

## 2. 核心架构

### 2.1 整体流程

```
NFJDServer:
  1. 客户端采样
  2. 模型分发 → 客户端本地训练
  3. 收集 Δθ + align_scores
  4. 对齐感知权重调整
  5. 加权聚合 Δθ
  6. 全局动量更新
  7. θ_global += momentum_delta

NFJDClient:
  对每个本地 epoch:
    对每个 mini-batch:
      重计算步: 计算Jacobian → MGDA方向 → 自适应缩放 → 动量
      非重计算步: 复用λ权重 → 加权梯度 → 动量
      参数更新: θ = θ - lr * momentum_direction
  上传: Δθ + align_scores
```

### 2.2 客户端侧：JD + 自适应缩放

每个客户端在本地执行E轮多目标优化训练：
- 使用上一步λ*构建联合损失实现单次backward
- 提取独立梯度用于MinNorm求解
- AdaptiveRescaling调整方向范数
- 本地动量更新参数
- 最终仅上传Δθ

### 2.3 StochasticGramianSolver

当m > threshold时，随机采样子集构建Gram矩阵：
- 计算复杂度：O(m²) → O(k²)
- 内存占用：O(m×d) → O(k×d)
- 子集λ扩展到全目标空间

### 2.4 通信效率

| 方法 | 上传内容 | 通信量/客户端 |
|------|----------|--------------|
| FedJD | Jacobian (m×d) | m×d×4 B |
| NFJD | Δθ (d) + align_scores (m) | (d+m)×4 B |

通信量与m完全解耦（m个对齐分数可忽略不计）。

### 2.5 AdaptiveRescaling

参考范数为各目标梯度均值的L2范数：
- 缩放因子 = min(N_raw / N_d, max_scale)
- 确保Gram矩阵病态时步长仍合理

### 2.6 双动量机制

- **本地动量**（beta=0.9）：平滑客户端更新方向
- **全局动量**（beta=0.9）：平滑服务器聚合方向
- **冲突感知动量**：β = base_β × (1 - avg_cosine_sim)，下限min_β

### 2.7 单次联合反向传播

将m次独立backward优化为1次联合backward：
- 用上一步λ*构建联合损失
- 非重计算步仅需1次backward
- 重计算间隔控制Jacobian计算频率

---

## 3. 目标对齐分数（v2更新）

### 3.1 定义

`align_scores = G @ Δθ`，衡量更新对各目标的"友好程度"：
- align_scores[j] > 0：更新对目标j友好（损失下降）
- align_scores[j] < 0：更新对目标j有害（损失上升）

### 3.2 动态权重调整

```
1. global_align = Σ(weight_i × align_scores_i)
2. 对每个客户端:
   若 global_align[j] < 0 (目标j被普遍伤害):
     若 client_align[j] > 0 (当前客户端对目标j友好):
       adjustment *= 1.5
3. 归一化调整后权重
4. 加权聚合 Δθ
```

### 3.3 多目标健康监控

通过`global_align`实时监控各目标优化状况，提供早期预警。

---

## 4. 与FedJD的对比

| 特性 | FedJD (旧) | NFJD (新) |
|------|-----------|-----------|
| 上传内容 | Jacobian (m×d) | Δθ (d) + align_scores (m) |
| 通信量/客户端 | m×d×4 B | (d+m)×4 B |
| 服务器聚合 | MinNorm方向寻找 | 对齐感知加权平均 |
| 方向范数 | 保守（范数小） | AdaptiveRescaling恢复 |
| 动量 | 无 | 双动量（本地+全局） |
| Backward次数 | m次/轮 | 1次/轮（非重计算步） |
| 高m处理 | O(m²) | StochasticGramian O(k²) |
| 隐私保护 | 仅数据不出客户端 | Δθ天然遮蔽梯度信息 |

---

## 5. 性能优化

| 优化项 | 方法 | 效果 |
|--------|------|------|
| GPU加速 | 纯GPU tensor操作 | 避免GPU-CPU同步 |
| 自适应缩放 | torch.minimum | 保持GPU在设备上 |
| 梯度缓存 | recompute_interval | 非重算步90%+计算节省 |
| 随机Gramian | StochasticGramianSolver | O(m²)→O(k²) |
| 动态迭代 | 根据m调整MinNorm迭代次数 | m≤2:50, m≤5:100, m≤8:200 |
| 混合精度 | torch.amp.autocast | GPU上自动FP16加速 |

---

## 6. 关键参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| local_epochs | 本地训练轮数 | 3 |
| local_momentum_beta | 本地动量系数 | 0.9 |
| global_momentum_beta | 全局动量系数 | 0.9 |
| use_adaptive_rescaling | 是否使用自适应缩放 | True |
| use_stochastic_gramian | 是否使用随机Gramian | True |
| stochastic_subset_size | 随机子集大小 | 4 |
| recompute_interval | Jacobian重算间隔 | 4 |
| conflict_aware_momentum | 是否启用冲突感知动量 | False |
| momentum_min_beta | 动量下限 | 0.1 |
| rescaling_max_scale | 缩放上限 | 5.0 |
| participation_rate | 客户端参与率 | 0.5 |
| learning_rate | 学习率 | 0.01 |

---

## 7. 实验阶段

| Phase | 脚本 | 焦点 |
|-------|------|------|
| Phase 1 | run_phase1_baseline.py | 基线验证（5方法 × 3 m × 3种子） |
| Phase 2 | run_phase2_ablation.py | 消融实验（AR, 动量, SG, 本地轮次） |
| Phase 3 | run_phase3_highconflict.py | 高冲突场景验证 |
| Phase 4 | run_phase4_benchmark.py | 完整基准（回归+分类+Non-IID） |
| Phase 5 | run_phase5_multimnist.py / run_phase5_celeba.py / run_phase5_riverflow.py | 真实数据（按数据集分开） |
| Phase 6 | run_phase6_recompute_ablation.py | 重计算间隔消融 |

---

## 8. FedJD遗留问题与NFJD解决方案

| FedJD问题 | 严重度 | NFJD解决方案 |
|-----------|--------|-------------|
| 收敛速度慢于基线 | 高 | AdaptiveRescaling + 双动量 |
| 通信开销随m线性增长 | 高 | Δθ上传，通信量与m解耦 |
| 计算复杂度高于基线 | 中 | 单次backward + λ*复用 |
| m=10时性能严重退化 | 高 | StochasticGramianSolver |
| 高冲突场景无优势 | 中 | 冲突感知动量 |
| 缺乏差分隐私保护 | 低 | Δθ天然遮蔽梯度信息 |
| 缺乏真实数据集验证 | 高 | Phase 5: MultiMNIST + CelebA + RiverFlow |
