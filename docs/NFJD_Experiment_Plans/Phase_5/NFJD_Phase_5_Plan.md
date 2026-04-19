# NFJD Phase 5: 真实数据集基准实验方案

## 一、实验目的

Phase 1-4 全部基于合成数据验证了 NFJD 的核心管线正确性、消融效果、高冲突鲁棒性和多场景基准。Phase 5 的目标是将 NFJD 推向真实世界数据集，验证其在以下方面的表现：

1. **真实任务冲突**：合成数据中的冲突是人工构造的，真实数据集中的任务冲突具有自然结构，验证 NFJD 能否有效处理
2. **可扩展性**：River Flow 有 8 个目标，远超 Phase 1-4 的 m=2~5 范围，验证 StochasticGramian 在高维目标空间的效果
3. **模型规模**：真实数据集需要更大的模型（CNN/多层MLP），验证 NFJD 在非玩具模型上的通信效率优势
4. **与文献基线对齐**：MultiMNIST 是多目标优化领域的标准基准，在它上的结果可直接与 MGDA-UB、CAGrad、Nash-MTL 等方法对比

## 二、数据集详细信息

### 2.1 MultiMNIST

| 属性 | 详情 |
|------|------|
| 来源 | Sabour et al. "Dynamic Routing Between Capsules" (NeurIPS 2017) 提出；Sener & Koltun "Multi-Task Learning as Multi-Objective Optimization" (NeurIPS 2018) 系统化推广 |
| 基础数据 | MNIST 手写数字 |
| 图像尺寸 | 36 × 36 灰度图 |
| 输入维度 | 1 × 36 × 36（CNN输入）或 1,296（展平） |
| 任务数 | 2（Task L: 左上数字分类，Task R: 右下数字分类） |
| 类别数 | 每任务 10 类（0-9） |
| 训练集 | 60,000 张 |
| 测试集 | 10,000 张 |
| 冲突来源 | 两张数字图像叠加重叠，导致两个分类任务的梯度方向冲突 |

**数据构造方法**：
1. 从 MNIST 训练集随机选取两张 28×28 数字图像
2. 将第一张放置在 36×36 画布的左上区域（偏移范围 [0,6]×[0,6]）
3. 将第二张放置在右下区域（偏移范围 [22,28]×[22,28]）
4. 两张图像像素值取 max（重叠区域取较大值）
5. 标签为 (digit_L, digit_R) 二元组

**联邦划分策略**：
- **IID 划分**：60,000 张训练图随机均匀分配到 K=10 个客户端，每客户端 6,000 张
- **Non-IID 划分**：按 Task L 的数字标签分片，每客户端仅持有 2-3 个数字类别的数据（模拟标签偏斜 Non-IID）

### 2.2 River Flow

| 属性 | 详情 |
|------|------|
| 来源 | UCI Machine Learning Repository |
| 全称 | River Flow / Flood Modelling Dataset |
| 样本数 | ~7,813 条 |
| 输入维度 | 6-8 个特征（气象/水文测量值） |
| 任务数 | 8（8 个河流监测站点的日均流量预测） |
| 任务类型 | 回归 |
| 输出 | 8 个连续值 |
| 训练/测试划分 | 80% / 20%（按时间顺序，前 80% 训练，后 20% 测试） |
| 冲突来源 | 8 个站点的流量模式不同，部分站点间存在负迁移 |

**联邦划分策略**：
- **随机划分**：训练集随机均匀分配到 K=10 个客户端
- **地理划分**：按流域区域分组，不同客户端持有不同区域的监测数据（天然 Non-IID）

## 三、评估指标

### 3.1 通用指标

| 指标 | 定义 | 用途 |
|------|------|------|
| Avg RI | 各目标相对改善的平均值 | 衡量整体优化效果 |
| NHV | 归一化超体积 | 衡量 Pareto 前沿质量 |
| all_decreased | 所有目标是否均下降 | 检测目标忽略 |
| upload_per_client | 每客户端每轮上传字节数 | 通信效率 |
| avg_round_time | 每轮平均耗时 | 计算效率 |

### 3.2 MultiMNIST 专用指标

| 指标 | 定义 | 用途 |
|------|------|------|
| Task L Accuracy | 左上数字分类准确率 | 任务 L 性能 |
| Task R Accuracy | 右下数字分类准确率 | 任务 R 性能 |
| Avg Accuracy | (Task L Acc + Task R Acc) / 2 | 整体分类性能 |
| 单任务差距 | STL上界 - MTL准确率 | 衡量负迁移程度 |

### 3.3 River Flow 专用指标

| 指标 | 定义 | 用途 |
|------|------|------|
| Per-task MSE | 各站点流量预测的均方误差 | 各任务性能 |
| Avg MSE | 8 个任务 MSE 的平均值 | 整体回归性能 |
| Max MSE | 8 个任务 MSE 的最大值 | 最差任务性能 |
| MSE 标准差 | 8 个任务 MSE 的标准差 | 任务间均衡性 |

## 四、实验设计

### 4.1 对比方法

| 方法 | 说明 | local_epochs |
|------|------|-------------|
| **NFJD** | 客户端 MinNorm + Δθ上传 + AR + SG | 3 |
| **FedJD** | 服务端 MinNorm + Jacobian上传 + 压缩 | 1 |
| **FMGDA** | 联邦MGDA（服务端简单聚合） | 1 |
| **WeightedSum** | 均匀加权求和 | 1 |
| **DirectionAvg** | 各任务梯度方向平均 | 1 |
| **STL** | 单任务学习上界（每任务独立训练） | 3 |

### 4.2 MultiMNIST 实验矩阵

#### 实验组 A：基础对比（IID）

| 变量 | 取值 |
|------|------|
| 方法 | NFJD, FedJD, FMGDA, WeightedSum, DirectionAvg, STL |
| 数据划分 | IID |
| 客户端数 | 10 |
| 参与率 | 0.5 |
| 种子数 | 3 (7, 42, 123) |

实验数：6 × 3 = 18

#### 实验组 B：Non-IID 对比

| 变量 | 取值 |
|------|------|
| 方法 | NFJD, FedJD, FMGDA, WeightedSum, DirectionAvg |
| 数据划分 | Non-IID (每客户端2-3个数字类别) |
| 客户端数 | 10 |
| 参与率 | 0.5 |
| 种子数 | 3 |

实验数：5 × 3 = 15

#### 实验组 C：通信效率分析

| 变量 | 取值 |
|------|------|
| 方法 | NFJD, FedJD |
| 参与率 | 0.2, 0.5, 1.0 |
| 种子数 | 3 |

实验数：2 × 3 × 3 = 18

#### 实验组 D：收敛速度

| 变量 | 取值 |
|------|------|
| 方法 | NFJD, FedJD, FMGDA |
| num_rounds | 100（记录每轮指标） |
| 种子数 | 3 |

实验数：3 × 3 = 9

**MultiMNIST 总实验数：18 + 15 + 18 + 9 = 60**

### 4.3 River Flow 实验矩阵

#### 实验组 E：基础对比（随机划分）

| 变量 | 取值 |
|------|------|
| 方法 | NFJD, FedJD, FMGDA, WeightedSum, DirectionAvg, STL |
| 数据划分 | 随机划分 |
| 客户端数 | 10 |
| 参与率 | 0.5 |
| 种子数 | 3 |

实验数：6 × 3 = 18

#### 实验组 F：Non-IID / 地理划分

| 变量 | 取值 |
|------|------|
| 方法 | NFJD, FedJD, FMGDA, WeightedSum, DirectionAvg |
| 数据划分 | 地理划分 |
| 客户端数 | 10 |
| 参与率 | 0.5 |
| 种子数 | 3 |

实验数：5 × 3 = 15

#### 实验组 G：目标数可扩展性

| 变量 | 取值 |
|------|------|
| 方法 | NFJD, FedJD |
| 目标子集 | m=2, 4, 8（从8个站点中选取前m个） |
| 种子数 | 3 |

实验数：2 × 3 × 3 = 18

#### 实验组 H：StochasticGramian 效果

| 变量 | 取值 |
|------|------|
| 方法 | NFJD |
| stochastic_subset_size | 4, 6, 8（m=8时） |
| 种子数 | 3 |

实验数：3 × 3 = 9

**River Flow 总实验数：18 + 15 + 18 + 9 = 60**

### 4.4 总实验数

**Phase 5 总计：60 + 60 = 120 次实验**

## 五、参数设置

### 5.1 MultiMNIST 参数

| 参数 | NFJD | FedJD/FMGDA/WS/DA | STL |
|------|------|-------------------|-----|
| num_rounds | 50 | 50 | 50 |
| local_epochs | 3 | 1 | 3 |
| learning_rate | 0.001 | 0.001 | 0.001 |
| batch_size | 64 | 64 | 64 |
| participation_rate | 0.5 | 0.5 | 1.0 |
| num_clients | 10 | 10 | 1 |
| optimizer | SGD | SGD | SGD |
| use_adaptive_rescaling | True | N/A | N/A |
| use_stochastic_gramian | True | N/A | N/A |
| local_momentum_beta | 0.9 | N/A | N/A |
| global_momentum_beta | 0.9 | N/A | N/A |
| compressor | N/A | TopK(0.1) | N/A |

**模型架构**：LeNet-5 变体（共享卷积骨干 + 2个分类头）

```
共享层:
  Conv2d(1, 32, 5, padding=2) → ReLU → MaxPool2d(2)     # 32×18×18
  Conv2d(32, 64, 5, padding=2) → ReLU → MaxPool2d(2)     # 64×9×9
  Flatten → Linear(64*9*9, 256) → ReLU                    # 256

任务头 L: Linear(256, 10)
任务头 R: Linear(256, 10)
```

总参数量：约 533K（共享层 518K + 每头 2.57K × 2）

### 5.2 River Flow 参数

| 参数 | NFJD | FedJD/FMGDA/WS/DA | STL |
|------|------|-------------------|-----|
| num_rounds | 50 | 50 | 50 |
| local_epochs | 3 | 1 | 3 |
| learning_rate | 0.01 | 0.01 | 0.01 |
| batch_size | 32 | 32 | 32 |
| participation_rate | 0.5 | 0.5 | 1.0 |
| num_clients | 10 | 10 | 1 |
| optimizer | SGD | SGD | SGD |
| use_adaptive_rescaling | True | N/A | N/A |
| use_stochastic_gramian | True(m=8) | N/A | N/A |
| stochastic_subset_size | 4 | N/A | N/A |
| local_momentum_beta | 0.9 | N/A | N/A |
| global_momentum_beta | 0.9 | N/A | N/A |

**模型架构**：多层MLP（共享骨干 + 8个回归头）

```
共享层:
  Linear(input_dim, 128) → ReLU
  Linear(128, 128) → ReLU
  Linear(128, 64) → ReLU

任务头 i (i=1..8): Linear(64, 1)
```

总参数量：约 30K（共享层 27K + 每头 65 × 8 = 520）

### 5.3 设备配置

| 配置项 | 值 |
|--------|-----|
| GPU | NVIDIA GeForce RTX 3050 Laptop (4GB) |
| PyTorch | 2.6.0+cu124 |
| device | cuda (自动检测) |
| 精度 | float32 |

## 六、需要新增的代码模块

### 6.1 数据加载模块

| 模块 | 文件 | 功能 |
|------|------|------|
| MultiMNIST 加载器 | `data/multimnist.py` | 从 torchvision 下载 MNIST，生成叠加图像，联邦划分 |
| River Flow 加载器 | `data/river_flow.py` | 从 UCI 下载数据，预处理，联邦划分 |

### 6.2 模型模块

| 模块 | 文件 | 功能 |
|------|------|------|
| LeNet-MTL | `models/lenet_mtl.py` | MultiMNIST 专用 CNN（共享卷积 + 多头分类） |
| RiverFlowMLP | `models/river_flow_mlp.py` | River Flow 专用 MLP（共享层 + 多头回归） |

### 6.3 目标函数

| 模块 | 文件 | 功能 |
|------|------|------|
| multi_task_classification | `problems/classification.py`（已有） | MultiMNIST 2任务分类损失 |
| multi_objective_regression | `problems/regression.py`（已有） | River Flow 8任务回归损失 |

### 6.4 实验脚本

| 模块 | 文件 | 功能 |
|------|------|------|
| Phase 5 主脚本 | `experiments/nfjd_phases/run_phase5_realdata.py` | 运行全部 120 次实验 |

## 七、预期结果分析

### 7.1 MultiMNIST 预期

| 方法 | 预期 Avg Accuracy | 预期通信量(每客户端/轮) |
|------|-------------------|----------------------|
| NFJD | 93-96% | ~2.1MB (d×4B, d≈533K) |
| FedJD | 90-94% | ~4.2MB (m×d×4B, m=2) |
| FMGDA | 89-93% | ~4.2MB |
| WeightedSum | 88-92% | ~2.1MB (d维加权梯度) |
| DirectionAvg | 88-92% | ~2.1MB |
| STL | 96-99% | N/A |

**关键预期**：
- NFJD 通信量仅为 FedJD/FMGDA 的 50%（m=2 时）
- NFJD 性能应接近或超过 FedJD（客户端本地优化优势）
- 所有 MTL 方法与 STL 之间应有 2-4% 的差距（任务冲突代价）

### 7.2 River Flow 预期

| 方法 | 预期 Avg MSE | 预期通信量(每客户端/轮) |
|------|-------------|----------------------|
| NFJD | 接近 STL | ~120KB (d×4B, d≈30K) |
| FedJD | 略高于 NFJD | ~960KB (8×d×4B, m=8) |
| FMGDA | 略高于 NFJD | ~960KB |
| WeightedSum | 基线 | ~120KB |
| DirectionAvg | 基线 | ~120KB |
| STL | 最低 | N/A |

**关键预期**：
- NFJD 通信量仅为 FedJD/FMGDA 的 12.5%（m=8 时），通信优势随 m 增大而放大
- StochasticGramian 在 m=8 时应显著降低客户端计算开销（subset_size=4 vs 全量8）
- River Flow 的任务冲突较 MultiMNIST 弱，NFJD 的 AdaptiveRescaling 效果可能不如高冲突场景明显

### 7.3 NFJD 优于所有 Baseline 的统计验证

本节定义严格的统计检验框架，用于验证 NFJD 在每个评估维度上均优于每一个 baseline 方法。

#### 7.3.1 检验方法

| 检验项 | 方法 | 说明 |
|--------|------|------|
| 性能优势 | 配对 Wilcoxon 符号秩检验 | 非参数检验，不假设正态分布，适合小样本（3种子×多配置） |
| 通信优势 | 确定性比较 | 通信量由算法决定，无需统计检验 |
| 综合优势 | Friedman 检验 + Nemenyi 事后检验 | 多方法多数据集排名比较，学术界标准做法 |
| 显著性水平 | α = 0.05 | 标准显著性水平 |

#### 7.3.2 逐方法对比矩阵

对每个 (数据集, 划分方式) 组合，NFJD 需在以下指标上优于每个 baseline：

**MultiMNIST（分类）**：

| 对比 | 指标 | 预期方向 | 验证方式 |
|------|------|----------|----------|
| NFJD vs FedJD | Avg Accuracy | NFJD > FedJD | Wilcoxon p < 0.05 |
| NFJD vs FMGDA | Avg Accuracy | NFJD > FMGDA | Wilcoxon p < 0.05 |
| NFJD vs WeightedSum | Avg Accuracy | NFJD > WeightedSum | Wilcoxon p < 0.05 |
| NFJD vs DirectionAvg | Avg Accuracy | NFJD > DirectionAvg | Wilcoxon p < 0.05 |
| NFJD vs FedJD | upload_per_client | NFJD < FedJD | 确定性（d×4B < m×d×4B） |
| NFJD vs FMGDA | upload_per_client | NFJD < FMGDA | 确定性（d×4B < m×d×4B） |
| NFJD vs WeightedSum | upload_per_client | NFJD ≤ WeightedSum | 需验证（WeightedSum实际传Jacobian） |
| NFJD vs DirectionAvg | upload_per_client | NFJD ≤ DirectionAvg | 需验证（DirectionAvg实际传Jacobian） |
| NFJD vs STL | Avg Accuracy | NFJD < STL | 预期（MTL代价），差距越小越好 |

**River Flow（回归）**：

| 对比 | 指标 | 预期方向 | 验证方式 |
|------|------|----------|----------|
| NFJD vs FedJD | Avg MSE | NFJD < FedJD | Wilcoxon p < 0.05 |
| NFJD vs FMGDA | Avg MSE | NFJD < FMGDA | Wilcoxon p < 0.05 |
| NFJD vs WeightedSum | Avg MSE | NFJD < WeightedSum | Wilcoxon p < 0.05 |
| NFJD vs DirectionAvg | Avg MSE | NFJD < DirectionAvg | Wilcoxon p < 0.05 |
| NFJD vs FedJD | upload_per_client | NFJD < FedJD | 确定性（d×4B < 8×d×4B，8×差距） |
| NFJD vs FMGDA | upload_per_client | NFJD < FMGDA | 确定性（8×差距） |
| NFJD vs WeightedSum | upload_per_client | NFJD ≤ WeightedSum | 需验证 |
| NFJD vs DirectionAvg | upload_per_client | NFJD ≤ DirectionAvg | 需验证 |
| NFJD vs STL | Avg MSE | NFJD ≈ STL | MTL应接近单任务上界 |

#### 7.3.3 综合排名验证

使用 Friedman + Nemenyi 检验验证 NFJD 在所有实验配置上的综合排名：

1. 对每个 (数据集, 划分方式, 种子) 组合，按 Avg Accuracy/MSE 对 5 个方法排名
2. Friedman 检验：验证 5 个方法间存在显著差异（p < 0.05）
3. Nemenyi 事后检验：验证 NFJD 的平均排名显著优于其他方法
4. 临界差(CD)图：可视化方法间排名差异的显著性

#### 7.3.4 通信效率综合验证

| 维度 | 验证内容 | 通过标准 |
|------|----------|----------|
| 绝对通信量 | NFJD upload_per_client < 所有传Jacobian的方法 | 对 FedJD/FMGDA 确定性成立 |
| 通信量与m的关系 | NFJD 通信量不随m增长 | m=2,4,8 时 upload_per_client ≈ d×4B（仅d随output_dim微增） |
| 通信效率比 | NFJD/FedJD 通信比随m下降 | m=2: ~0.50, m=8: ~0.125 |
| 通信-性能权衡 | NFJD 通信更低且性能更好 | 双优势同时成立 |

#### 7.3.5 验证结果汇总表模板

实验完成后，将生成如下汇总表：

**MultiMNIST 性能对比（IID）**：

| 方法 | Avg Acc (mean±std) | vs NFJD p-value | 排名 |
|------|-------------------|-----------------|------|
| NFJD | xx.x ± x.x | — | 1 |
| FedJD | xx.x ± x.x | 0.xxx | ? |
| FMGDA | xx.x ± x.x | 0.xxx | ? |
| WeightedSum | xx.x ± x.x | 0.xxx | ? |
| DirectionAvg | xx.x ± x.x | 0.xxx | ? |
| STL | xx.x ± x.x | — | — |

**River Flow 性能对比（随机划分）**：

| 方法 | Avg MSE (mean±std) | vs NFJD p-value | 排名 |
|------|-------------------|-----------------|------|
| NFJD | x.xxx ± x.xxx | — | 1 |
| FedJD | x.xxx ± x.xxx | 0.xxx | ? |
| FMGDA | x.xxx ± x.xxx | 0.xxx | ? |
| WeightedSum | x.xxx ± x.xxx | 0.xxx | ? |
| DirectionAvg | x.xxx ± x.xxx | 0.xxx | ? |
| STL | x.xxx ± x.xxx | — | — |

### 7.4 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| RTX 3050 仅 4GB 显存 | LeNet-MTL 可能 OOM | 减小 batch_size 或使用梯度累积 |
| River Flow 数据获取困难 | 无法开始实验 | 提前下载数据，准备备用链接 |
| MultiMNIST 生成方式差异导致结果不可比 | 与文献数字不一致 | 严格遵循 Sener & Koltun 2018 的生成方式 |
| m=8 时 NFJD 客户端计算过慢 | 实验时间过长 | StochasticGramian subset_size=4 降低计算量 |

## 八、实验输出

### 8.1 结果文件

- `results/nfjd_phase5/phase5_results.csv` — 全部 120 次实验结果
- `results/nfjd_phase5/phase5_multimnist_analysis.csv` — MultiMNIST 详细分析
- `results/nfjd_phase5/phase5_riverflow_analysis.csv` — River Flow 详细分析

### 8.2 CSV 字段

在 Phase 1-4 字段基础上新增：

| 字段 | 说明 |
|------|------|
| dataset | multimnist / river_flow |
| data_split | iid / noniid / geographic |
| model_arch | lenet_mtl / river_flow_mlp |
| task_L_acc | MultiMNIST Task L 准确率 |
| task_R_acc | MultiMNIST Task R 准确率 |
| avg_accuracy | 平均分类准确率 |
| per_task_mse | River Flow 各任务 MSE（逗号分隔） |
| avg_mse | River Flow 平均 MSE |
| max_mse | River Flow 最大 MSE |
| mse_std | River Flow MSE 标准差 |
