# NFJD: Normalized Federated Jacobian Descent

## 0. Current Status

This document describes the current intended mainline definition of NFJD.

Current canonical choices:

- client local multi-objective core: `UPGrad`
- local solver mode: exact UPGrad recomputation on every local batch
- objective handling: per-objective gradient normalization before the UPGrad solve
- client update transport: upload `delta_theta` only
- server update: plain sample-weighted FedAvg on uploaded `delta_theta`

Adaptive rescaling, local/global momentum, alignment-aware reweighting, stochastic Gramian sampling, and weight reuse are now treated as optional extensions rather than part of the mainline definition.

For a compact implementation-state summary and future work plan, see `docs/NFJD_Status_and_Roadmap.md`.

## 1. 概述

NFJD（Normalized Federated Jacobian Descent）是对FedJD的全面架构改进，解决三个核心缺陷：

| 缺陷 | 改进方案 | 章节 |
|------|----------|------|
| 服务端启发式过多 | 回退到 plain FedAvg 聚合 | §2.1 |
| 多目标量纲不一致 | per-objective gradient normalization | §2.3 |
| 通信开销大 | 仅上传 Δθ | §2.2 |
| Jacobian 路线过慢 | exact local UPGrad 作为最强主干 | §2.2 |

---

## 2. 核心架构

### 2.1 整体流程

```
NFJDServer:
  1. 客户端采样
  2. 模型分发 → 客户端本地训练
  3. 收集 Δθ
  4. 按样本数加权聚合 Δθ
  5. θ_global += aggregated_delta

NFJDClient:
  对每个本地 epoch:
    对每个 mini-batch:
      拆分参数为 shared params 与 task-head params
      只在 shared params 上计算各目标梯度
      对 shared Jacobian 做 norm normalization
      用 UPGrad 求 shared 公共方向
      shared params: θ_s = θ_s - lr * direction
      head params: θ_h = θ_h - lr * grad(sum(loss_i))
  上传: Δθ
```

### 2.2 客户端侧：Exact Local UPGrad

每个客户端在本地执行E轮多目标优化训练：
- 每个 mini-batch 都显式计算 shared trunk 上的 task Jacobian
- 使用高精度 Gramian-space UPGrad 对偶QP求 shared 公共下降方向
- task-specific heads 不参加共同方向求解，而是按总损失做普通梯度更新
- 最终仅上传 Δθ

当前主线在小规模任务数（Phase 5 的 `m<=8`）下优先使用精确 active-set 枚举来求解 UPGrad 子问题，避免因为廉价近似求解器导致本地多目标方向偏弱。

### 2.2.1 Shared Params 与 Head Params

对于当前 Phase 5 的多头模型，参数分为两类：

- `shared params`：所有任务共享的表示层参数
  - MultiMNIST: `shared + fc_shared`
  - RiverFlow: `shared`
  - CelebA: `features + shared_fc`
- `head params`：每个任务私有的输出头参数
  - `heads[i]`

NFJD 现在只在 `shared params` 上执行 UPGrad，因为跨任务冲突主要发生在共享表示层；私有 head 本来就只服务各自任务，不应被强行耦合到共同方向里。

### 2.3 Objective Normalization

在把 Jacobian 送入 UPGrad 前，对每个 objective 的梯度向量按其运行范数做归一化：

- 目的：降低不同任务量纲、不同 loss 尺度对 UPGrad 权重求解的偏置
- 当前实现：客户端内的 EMA gradient-norm normalization
- 作用点：只影响 UPGrad 求方向，不改变最终上传格式

### 2.4 通信效率

| 方法 | 上传内容 | 通信量/客户端 |
|------|----------|--------------|
| FedJD | Jacobian (m×d) | m×d×4 B |
| NFJD | Δθ (d) | d×4 B |

主线 NFJD 不再依赖 `align_scores` 才能完成训练，因此通信量与 `m` 解耦。

### 2.5 Optional Extensions (Deferred)

以下机制保留在代码里，但不属于当前最强主干：

- `AdaptiveRescaling`
- local momentum / global momentum
- alignment-aware weighted averaging
- `recompute_interval` + weight reuse
- `StochasticGramianSolver`

---

## 3. 当前不纳入主线的机制

### 3.1 Alignment Scores

`align_scores = -(G @ Δθ)`，衡量更新对各目标的"友好程度"：
- align_scores[j] > 0：更新对目标j友好（损失下降）
- align_scores[j] < 0：更新对目标j有害（损失上升）

当前该机制不作为主线默认项，仅作为未来 server 侧增强的候选模块保留。

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
| 上传内容 | Jacobian (m×d) | Δθ (d) |
| 通信量/客户端 | m×d×4 B | d×4 B |
| 客户端多目标核心 | 服务端MinNorm方向寻找 | 客户端本地UPGrad-JD |
| 服务器聚合 | 加权平均 | 样本加权 FedAvg |
| 方向尺度处理 | 无 | objective normalization |
| 动量 | 无 | 默认关闭 |
| Backward次数 | m次/轮 | m次/本地 batch（exact mainline） |
| 高m处理 | O(m²) | 默认全量 exact，sampling 为可选项 |
| 隐私保护 | 仅数据不出客户端 | Δθ天然遮蔽梯度信息 |

---

## 5. 性能优化

| 优化项 | 方法 | 效果 |
|--------|------|------|
| GPU加速 | 纯GPU tensor操作 | 避免GPU-CPU同步 |
| Objective normalization | EMA grad-norm normalization | 降低任务尺度偏置 |
| 动态迭代 | 根据m调整UPGrad对偶QP迭代次数 | m≤2:50, m≤5:100, m≤8:200 |
| 混合精度 | torch.amp.autocast | GPU上自动FP16加速 |
| 可选近似 | recompute / stochastic Gramian | 后续确认有效后再启用 |

---

## 6. 关键参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| local_epochs | 本地训练轮数 | 3 |
| exact_upgrad | 是否每个 batch 都重算 UPGrad | True |
| use_objective_normalization | 是否使用目标梯度归一化 | True |
| objective_norm_momentum | EMA 动量系数 | 0.9 |
| local_momentum_beta | 本地动量系数 | 0.0 |
| global_momentum_beta | 全局动量系数 | 0.0 |
| use_adaptive_rescaling | 是否使用自适应缩放 | False |
| use_stochastic_gramian | 是否使用随机Gramian | False |
| stochastic_subset_size | 随机子集大小 | 4 |
| recompute_interval | Jacobian重算间隔 | 1 |
| conflict_aware_momentum | 是否启用冲突感知动量 | False |
| momentum_min_beta | 动量下限 | 0.1 |
| rescaling_max_scale | 缩放上限 | 5.0 |
| participation_rate | 客户端参与率 | 0.5 |
| learning_rate | 学习率 | 0.01 |

---

## 7. 评估指标

### 7.1 分类任务指标（MultiMNIST, CelebA）

| 指标 | 英文名 | 公式 | 含义 |
|------|--------|------|------|
| 准确率 | Accuracy | 正确数/总数 | 分类正确率 |
| F1分数 | F1 Score | 2×P×R/(P+R) | 精确率与召回率的调和平均 |
| Jain公平指数 | JFI | (Σx_i)²/(n×Σx_i²) | 任务间性能公平性，[0,1]，1=完全公平 |
| 最小-最大差距 | MMAG | max(acc)-min(acc) | 任务间极端性能差距 |

### 7.2 回归任务指标（RiverFlow）

| 指标 | 英文名 | 公式 | 含义 |
|------|--------|------|------|
| 均方误差 | MSE | Σ(pred-target)²/n | 回归精度 |
| Jain公平指数 | JFI | (Σ1/(mse_i+ε))²/(n×Σ(1/(mse_i+ε))²) | 任务间MSE公平性 |
| 最小-最大差距 | MMAG | max(mse)-min(mse) | 任务间极端MSE差距 |

### 7.3 通用指标

| 指标 | 英文名 | 含义 |
|------|--------|------|
| 平均相对改进 | Avg RI | (initial-final)/|initial| 的任务平均 |
| 所有目标改善 | All Decreased | 是否所有目标都改善 |
| 通信开销 | Upload/Client | 每客户端每轮上传字节数 |
| 训练时间 | Elapsed Time | 总训练时间 |

### 7.4 已废弃指标

| 旧指标 | 废弃原因 |
|--------|----------|
| NHV（归一化超体积） | 从整个训练历史提取Pareto前沿，几乎总=1.0，无法区分方法 |
| Pareto Gap | 用历史min做理想点，无法区分方法 |

---

## 8. 实验阶段

| Phase | 脚本 | 焦点 |
|-------|------|------|
| Phase 1 | run_phase1_baseline.py | 基线验证（5方法 × 3 m × 3种子） |
| Phase 2 | run_phase2_ablation.py | 消融实验（AR, 动量, SG, 本地轮次） |
| Phase 3 | run_phase3_highconflict.py | 高冲突场景验证 |
| Phase 4 | run_phase4_benchmark.py | 完整基准（回归+分类+Non-IID） |
| Phase 5 | run_phase5_suite.py / run_phase5_multimnist.py / run_phase5_celeba.py / run_phase5_riverflow.py | 真实数据 + 正式来源 baseline |
| Phase 6 | run_phase6_recompute_ablation.py | 重计算间隔消融 |

Phase 5 正式 baseline 来源见：`docs/NFJD_Experiment_Plans/Phase_5/Phase5_Official_Baselines.md`

---

## 9. FedJD遗留问题与NFJD解决方案

| FedJD问题 | 严重度 | NFJD解决方案 |
|-----------|--------|-------------|
| 收敛速度慢于基线 | 高 | AdaptiveRescaling + 双动量 |
| 通信开销随m线性增长 | 高 | Δθ上传，通信量与m解耦 |
| 计算复杂度高于基线 | 中 | 单次backward + UPGrad权重复用 |
| m=10时性能严重退化 | 高 | StochasticGramianSolver |
| 高冲突场景无优势 | 中 | 冲突感知动量 |
| 缺乏差分隐私保护 | 低 | Δθ天然遮蔽梯度信息 |
| 缺乏真实数据集验证 | 高 | Phase 5: MultiMNIST + CelebA + RiverFlow |

---

## 10. Future Work

The most relevant next steps are:

1. Validate the new exact local UPGrad + objective normalization backbone on Phase 5
2. Compare exact vs approximate local UPGrad only after the new backbone result is clear
3. Re-introduce optional modules one at a time:
   `adaptive rescaling`, `local momentum`, `global momentum`, `alignment-aware weighting`, `stochastic Gramian`
4. Add sensitivity analysis for objective normalization and optional server heuristics
5. Decide whether older benchmark scripts should remain active or be archived as legacy

The current practical recommendation is to keep the existing server-side global momentum in the mainline implementation unless ablations show that it is unnecessary.
