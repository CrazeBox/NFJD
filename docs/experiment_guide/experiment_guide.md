# FedJD Stage 1 实验指南

## 1. 概述

本指南详细说明如何运行 FedJD Stage 1（基线跑通与玩具验证）的全部实验。Stage 1 的核心目标是验证：**在完整上传局部 Jacobian 的设置下，FedJD 能否作为一个 server-centric 联邦多目标优化方法稳定跑通。**

本指南覆盖：
- 环境准备
- 单次实验运行
- 多配置多种子扫描
- 结果分析与可视化
- 验证清单

## 2. 环境准备

### 2.1 依赖安装

```bash
pip install torch numpy pyyaml matplotlib
```

可选依赖（用于内存监控）：
```bash
pip install psutil
```

### 2.2 项目结构

```
fedjd/
├── __init__.py                      # 包入口
├── config.py                        # ExperimentConfig 配置类
├── visualization.py                 # 可视化工具
├── aggregators/
│   └── __init__.py                  # Jacobian 聚合器（MinNorm/Mean/Random）
├── core/
│   ├── client.py                    # FedJDClient - 客户端 Jacobian 计算
│   ├── server.py                    # FedJDServer - 服务器端聚合与更新
│   └── trainer.py                   # FedJDTrainer - 训练循环与指标保存
├── data/
│   └── synthetic.py                 # 合成联邦回归数据生成
├── models/
│   └── small_regressor.py           # SmallRegressor MLP 模型
├── problems/
│   └── regression.py                # 二目标回归损失函数
├── toy_example/                     # ★ 玩具实验入口
│   ├── __init__.py
│   ├── run_experiment.py            # 单次实验运行
│   ├── run_sweep.py                 # 多配置扫描
│   └── analyze_results.py           # 结果分析
├── experiment_guide/                # ★ 实验指南
│   └── experiment_guide.md          # 本文档
│   └── stage1_checklist.md          # 验证清单
└── fedjd_experiment_plan/           # 实验计划文档
    ├── 00_unified_protocol.md
    ├── 01_stage_baseline_smoke.md
    └── ...
```

## 3. 快速开始

### 3.1 运行单次实验（最简方式）

```bash
cd /path/to/project
python -m fedjd.toy_example.run_experiment
```

这将使用默认配置运行一次实验：
- K=8 客户端，C=0.5 参与率
- 30 轮训练，MinNorm 聚合器
- seed=7

结果保存在 `results/S1-synth_m2-mlp-m2-K8-C0.5-E1-fulljac-v1/` 目录下。

### 3.2 自定义参数运行

```bash
python -m fedjd.toy_example.run_experiment \
    --seed 42 \
    --num-clients 16 \
    --participation-rate 0.25 \
    --aggregator mean \
    --num-rounds 50 \
    --lr 0.01 \
    --device cpu
```

### 3.3 完整参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seed` | 7 | 随机种子 |
| `--num-clients` | 8 | 客户端数量 (K) |
| `--samples-per-client` | 64 | 每客户端样本数 |
| `--input-dim` | 8 | 输入维度 |
| `--hidden-dim` | 16 | 隐藏层维度 |
| `--num-rounds` | 30 | 通信轮数 |
| `--batch-size` | 32 | 批大小 |
| `--lr` | 0.05 | 学习率 |
| `--participation-rate` | 0.5 | 客户端参与率 (C) |
| `--aggregator` | minnorm | 聚合策略 (minnorm/mean/random) |
| `--output-dir` | 自动生成 | 输出目录 |
| `--save-checkpoints` | False | 是否保存检查点 |
| `--device` | cpu | 设备 (cpu/cuda) |

## 4. 多配置扫描实验

### 4.1 运行完整扫描

按照 00 统一协议要求，至少使用 3 个随机种子，并覆盖不同参与率和客户端数。

```bash
python -m fedjd.toy_example.run_sweep \
    --seeds 7 42 2024 \
    --participation-rates 0.25 0.5 1.0 \
    --client-counts 8 16 \
    --aggregators minnorm mean random \
    --num-rounds 30 \
    --sweep-dir results/s1_sweep
```

这将运行 3 × 3 × 2 × 3 = 54 次实验。

### 4.2 快速验证（减少配置）

如果只想快速验证链路是否正确，可以减少配置：

```bash
python -m fedjd.toy_example.run_sweep \
    --seeds 7 42 \
    --participation-rates 0.5 \
    --client-counts 8 \
    --aggregators minnorm random \
    --num-rounds 10 \
    --sweep-dir results/s1_quick
```

这将运行 2 × 1 × 1 × 2 = 4 次实验。

### 4.3 扫描参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sweep-dir` | results/s1_sweep | 扫描结果根目录 |
| `--seeds` | 7 42 2024 | 随机种子列表 |
| `--participation-rates` | 0.25 0.5 1.0 | 参与率列表 |
| `--client-counts` | 8 16 | 客户端数列表 |
| `--aggregators` | minnorm mean random | 聚合器列表 |
| `--num-rounds` | 30 | 每次实验轮数 |
| `--lr` | 0.05 | 学习率 |
| `--batch-size` | 32 | 批大小 |
| `--device` | cpu | 设备 |

## 5. 结果分析

### 5.1 运行分析

```bash
python -m fedjd.toy_example.analyze_results \
    --sweep-dir results/s1_sweep
```

分析脚本将：
1. 读取所有实验的 sweep_results.csv 或 sweep_results.json
2. 输出聚合器对比、参与率影响、客户端数影响、种子稳定性分析
3. 生成对比图表
4. 生成 analysis_report.md

### 5.2 输出图表

| 文件名 | 内容 |
|--------|------|
| `aggregator_comparison.png` | 三种聚合器的目标值变化对比 |
| `participation_rate_impact.png` | 参与率对优化效果的影响 |
| `seed_stability.png` | 不同种子下结果的稳定性 |
| `objective_trajectories.png` | 典型配置的目标值训练曲线 |

## 6. 单次实验结果目录结构

每次实验运行后，结果目录包含：

```
results/<experiment_id>/
├── config.yaml          # 实验配置（可复现）
├── metrics.csv          # 每轮详细指标
├── summary.md           # 实验总结
├── stdout.log           # 完整运行日志
├── plots/
│   ├── objectives.png   # 目标值曲线
│   ├── norms.png        # Jacobian范数与方向范数
│   ├── communication.png # 通信量
│   └── time_breakdown.png # 时间分解
└── checkpoints/         # 模型检查点（如启用）
```

### 6.1 metrics.csv 字段说明

| 字段 | 说明 |
|------|------|
| `round` | 轮次索引 |
| `sampled_clients` | 本轮采样的客户端ID列表 |
| `num_sampled` | 采样客户端数 |
| `objective_0` | 目标0的全局值 |
| `objective_1` | 目标1的全局值 |
| `direction_norm` | 下降方向的L2范数 |
| `jacobian_norm` | 聚合Jacobian的Frobenius范数 |
| `round_time` | 本轮总时间 |
| `upload_bytes` | 本轮上行字节数 |
| `download_bytes` | 本轮下行字节数 |
| `nan_inf_count` | NaN/Inf出现次数 |
| `client_compute_time` | 客户端计算时间 |
| `aggregation_time` | 聚合时间 |
| `direction_time` | 方向计算时间 |
| `update_time` | 参数更新时间 |

## 7. 核心算法流程

FedJD 的单轮训练流程如下：

```
1. 服务器采样参与客户端（按参与率C）
2. 每个参与客户端：
   a. 接收全局模型
   b. 在本地数据上计算 m 个目标的梯度
   c. 组装局部 Jacobian 矩阵 [m × d]
   d. 上传 Jacobian 到服务器
3. 服务器聚合：
   a. 按样本数加权平均所有客户端的 Jacobian
   b. 得到全局 Jacobian J ∈ R^{m×d}
4. 服务器方向决策：
   a. MinNorm: 在单纯形上求解最小范数方向（MGDA风格）
   b. Mean: 简单平均所有目标梯度
   c. Random: 随机权重加权
5. 服务器更新全局参数：θ ← θ - lr × d
6. 评估全局目标值，记录指标
```

## 8. 关键验证点

### 8.1 Jacobian 维度正确性

- Jacobian 形状应为 `[m, d]`，其中 m=2（目标数），d 为模型参数量
- SmallRegressor(input_dim=8, hidden_dim=16, output_dim=2) 的参数量 d = 8×16+16 + 16×2+2 = 162
- 因此 Jacobian 形状应为 `[2, 162]`

### 8.2 聚合一致性

- 加权聚合公式：J_global = Σ_k (n_k / N) × J_k
- 其中 n_k 是客户端 k 的样本数，N 是所有参与客户端的总样本数

### 8.3 方向有效性

- MinNorm 方向应满足：d = J^T × λ，其中 λ 在单纯形上
- 方向范数 ||d|| 应为有限正值
- 方向应使所有目标有下降趋势（Pareto 下降方向）

### 8.4 数值稳定性

- 不应出现 NaN 或 Inf
- Jacobian Frobenius 范数应在合理范围内（不爆炸也不消失）

## 9. 基线对照说明

Stage 1 的三种聚合器对比目的：

| 聚合器 | 作用 | 预期表现 |
|--------|------|----------|
| **MinNorm** | FedJD 核心方法，求最小范数下降方向 | 应给出最平衡的多目标优化 |
| **Mean** | 简单平均梯度，作为朴素基线 | 可能偏向某个目标 |
| **Random** | 随机权重，作为下界参考 | 性能应最差，用于确认 FedJD 不是空跑 |

**关键判定**：MinNorm 应优于 Random，否则说明方向选择机制存在问题。

## 10. 常见问题排查

### Q1: 目标值不下降
- 检查学习率是否过大（导致震荡）或过小（收敛太慢）
- 检查 Jacobian 是否全为零（可能模型未正确计算梯度）
- 检查方向范数是否合理

### Q2: 出现 NaN/Inf
- 降低学习率
- 检查数据中是否有异常值
- 检查 MinNorm 聚合器的迭代是否收敛

### Q3: MinNorm 不优于 Random
- 确认 MinNorm 聚合器的 max_iters 和 lr 设置合理
- 检查 gramian 矩阵是否正确计算
- 尝试增加聚合器迭代次数

### Q4: 不同种子差异过大
- 增加训练轮数
- 增加每客户端样本数
- 检查数据生成是否对种子过于敏感

## 11. 与统一协议的对应

本实验指南遵循 `00_unified_protocol.md` 的以下要求：

| 协议要求 | 本指南实现 |
|----------|-----------|
| 至少3个随机种子 | 默认使用 seeds=[7, 42, 2024] |
| 报告 mean/std/最优最差 | sweep_summary.md 中自动计算 |
| 相同数据划分 | 同一种子生成相同数据 |
| 相同模型结构 | 统一使用 SmallRegressor |
| 相同参与率 | 扫描中保持一致 |
| 相同训练预算 | 所有配置使用相同轮数 |
| 完整指标采集 | metrics.csv 包含所有必需字段 |
| 结果保存格式 | config.yaml + metrics.csv + summary.md + stdout.log + plots/ |

## 12. 下一步

Stage 1 通过后，进入 Stage 2（通信代价画像），详见 `02_stage_comm_profile.md`。
