# NFJD Phase 5: 真实数据集基准实验方案

> 注：Phase 5 的跨版本方法状态、已完成修改和后续路线以 `docs/NFJD_Status_and_Roadmap.md` 为总览记录；本文件只记录当前阶段的正式实验方案。

## 一、实验目的

Phase 1-4 全部基于合成数据验证了 NFJD 的核心管线正确性、消融效果、高冲突鲁棒性和多场景基准。Phase 5 的目标是将 NFJD 推向真实世界数据集，验证其在以下方面的表现：

1. **真实任务冲突**：合成数据中的冲突是人工构造的，真实数据集中的任务冲突具有自然结构，验证 NFJD 能否有效处理
2. **可扩展性**：River Flow 有 8 个目标，远超 Phase 1-4 的 m=2~5 范围，验证 StochasticGramian 在高维目标空间的效果
3. **模型规模**：真实数据集需要更大的模型（CNN/多层MLP），验证 NFJD 在非玩具模型上的通信效率优势
4. **与文献基线对齐**：Phase 5 主表 baseline 将绑定到 FMGDA、PCGrad、CAGrad、线性标量化等有明确论文/官方代码来源的方法

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
- **IID 划分**：训练集随机均匀分配到 K=10 个客户端
- **Non-IID 划分**：按第 1 个目标值排序后切块，再打乱块顺序分配到客户端，形成目标分布偏斜

## 三、评估指标

### 3.1 通用指标

| 指标 | 定义 | 用途 |
|------|------|------|
| Avg RI | 各目标相对改善的平均值 | 衡量整体优化效果 |
| all_decreased | 所有目标是否均下降 | 检测目标忽略 |
| upload_per_client | 每客户端每轮上传字节数 | 通信效率 |
| avg_round_time | 每轮平均耗时 | 计算效率 |
| source_paper / source_official_repo | baseline 来源元数据 | 结果可追溯性 |

### 3.2 MultiMNIST 专用指标

| 指标 | 定义 | 用途 |
|------|------|------|
| Task L Accuracy | 左上数字分类准确率 | 任务 L 性能 |
| Task R Accuracy | 右下数字分类准确率 | 任务 R 性能 |
| Avg Accuracy | (Task L Acc + Task R Acc) / 2 | 整体分类性能 |
| task_jfi / task_mmag | 任务公平性与任务差距 | 衡量多任务均衡性 |

### 3.3 CelebA / River Flow 专用指标

| 指标 | 定义 | 用途 |
|------|------|------|
| Avg F1 | CelebA 多标签属性预测平均 F1 | 多标签分类性能 |
| Avg MSE | RiverFlow 任务 MSE 平均值 | 整体回归性能 |
| Max MSE | 8 个任务 MSE 的最大值 | 最差任务性能 |
| MSE 标准差 | 8 个任务 MSE 的标准差 | 任务间均衡性 |

## 四、实验设计

### 4.0 当前方法状态说明

Phase 5 当前默认使用的 NFJD 主线版本为：

- 客户端本地多目标核心：`UPGrad-JD`
- 局部近似：`recompute_interval` + 本地 UPGrad 权重复用
- 高 `m` 处理：`StochasticGramianSolver`
- 局部稳定化：`AdaptiveRescaling`
- 服务器更新：`alignment-aware weighted average + global momentum`

若后续需要评估 `NFJD-exact-UPGrad`、去掉服务器全局动量、或去掉对齐感知聚合，应作为单独 ablation 或版本分支记录，不应混入当前主表结果。

### 4.1 对比方法

| 方法 | 说明 | 来源绑定 | local_epochs |
|------|------|---------|-------------|
| **NFJD** | 客户端 UPGrad-JD + Δθ上传 + AR + SG | 本仓库方法 | 3 |
| **FedAvg+LS** | 本地等权线性标量化，多任务标准基线 | Nash-MTL 官方代码库 baseline API | 3 |
| **FMGDA** | 论文版联邦多目标优化基线；客户端按 objective 分别做本地轨迹更新，服务器端聚合后解 MGDA 方向 | Yang et al. 2023, *Federated Multi-Objective Learning* | 3 |
| **FedAvg+PCGrad** | 本地 gradient surgery | Yu et al. 2020 + 官方仓库 | 3 |
| **FedAvg+CAGrad** | 本地 conflict-averse 梯度组合 | Liu et al. 2021 + 官方仓库 | 3 |

说明：

1. Phase 5 的主表方法现为 `NFJD / FMGDA / FedAvg+LS / FedAvg+PCGrad / FedAvg+CAGrad`。
2. 其中 `FMGDA` 是原生联邦多目标 baseline；`FedAvg+LS / PCGrad / CAGrad` 是中心化多任务优化器在统一联邦外壳下的适配版。
3. 详细来源见 `Phase5_Official_Baselines.md`。
4. 当前实现要求 baseline 尽量贴近官方论文/官方代码：`LS` 使用原始 sum-loss 标量化；`PCGrad/CAGrad` 只在共享 trunk 上做梯度组合，任务 heads 按 summed loss 正常更新；`FMGDA` 使用按客户端样本权重的 objective 聚合。

### 4.2 MultiMNIST 实验矩阵

#### 实验组 A：正式基线对比（IID）

| 变量 | 取值 |
|------|------|
| 方法 | NFJD, FMGDA, FedAvg+LS, FedAvg+PCGrad, FedAvg+CAGrad |
| 数据划分 | IID |
| 客户端数 | 10 |
| 参与率 | 0.5 |
| 种子数 | 3 (7, 42, 123) |

实验数：5 × 3 = 15

#### 实验组 B：Non-IID 对比

| 变量 | 取值 |
|------|------|
| 方法 | NFJD, FMGDA, FedAvg+LS, FedAvg+PCGrad, FedAvg+CAGrad |
| 数据划分 | Non-IID (每客户端2-3个数字类别) |
| 客户端数 | 10 |
| 参与率 | 0.5 |
| 种子数 | 3 |

实验数：5 × 3 = 15

**MultiMNIST 总实验数：15 + 15 = 30**

### 4.3 River Flow 实验矩阵

#### 实验组 C：正式基线对比（随机划分）

| 变量 | 取值 |
|------|------|
| 方法 | NFJD, FMGDA, FedAvg+LS, FedAvg+PCGrad, FedAvg+CAGrad |
| 数据划分 | 随机划分 |
| 客户端数 | 10 |
| 参与率 | 0.5 |
| 种子数 | 3 |

实验数：5 × 3 = 15

#### 实验组 D：Non-IID / 地理划分

| 变量 | 取值 |
|------|------|
| 方法 | NFJD, FMGDA, FedAvg+LS, FedAvg+PCGrad, FedAvg+CAGrad |
| 数据划分 | 地理划分 |
| 客户端数 | 10 |
| 参与率 | 0.5 |
| 种子数 | 3 |

实验数：5 × 3 = 15

**River Flow 总实验数：15 + 15 = 30**

### 4.4 CelebA 实验矩阵

#### 实验组 E：正式基线对比（IID / 4属性）

| 变量 | 取值 |
|------|------|
| 方法 | NFJD, FMGDA, FedAvg+LS, FedAvg+PCGrad, FedAvg+CAGrad |
| 数据划分 | IID |
| 任务数 | 4 |
| 客户端数 | 10 |
| 种子数 | 3 |

实验数：5 × 3 = 15

#### 实验组 F：正式基线对比（Non-IID / 4属性）

| 变量 | 取值 |
|------|------|
| 方法 | NFJD, FMGDA, FedAvg+LS, FedAvg+PCGrad, FedAvg+CAGrad |
| 数据划分 | Non-IID |
| 任务数 | 4 |
| 客户端数 | 10 |
| 种子数 | 3 |

实验数：5 × 3 = 15

#### 实验组 G：属性数扩展性

| 变量 | 取值 |
|------|------|
| 方法 | NFJD, FMGDA, FedAvg+CAGrad |
| 数据划分 | IID |
| 任务数 | 2, 4, 6 |
| 种子数 | 3 |

实验数：3 × 3 × 3 = 27

**CelebA 总实验数：15 + 15 + 27 = 57**

### 4.4 总实验数

**Phase 5 总计：30 + 30 + 57 = 117 次实验**

### 4.5 当前程序跑完后的待办

以下事项不影响当前程序先跑通，但应在拿到第一轮 Phase 5 结果后尽快补做：

1. 对 `FMGDA / FedAvg+LS / FedAvg+PCGrad / FedAvg+CAGrad` 做小规模调参，避免主结论过度依赖统一默认学习率。
2. 对 `FMGDA` 额外测试本地步长 `eta_L` 与服务器步长 `eta_t` 分离的设定，确认当前单学习率配置不会误伤 baseline。
3. Phase 5 主表统一输出 `mean ± std`，不要只保留逐 seed 原始结果。
4. 在最终汇总表上补 Wilcoxon 与 Friedman/Nemenyi 检验，降低“只看均值”的争议。
5. 论文与结果说明中明确区分：`FMGDA` 是原生联邦多目标 baseline，`FedAvg+LS / PCGrad / CAGrad` 是统一联邦外壳下的适配 baseline。
6. 跑完后复核 `avg_round_time` 与 `avg_upload_bytes`，只在通信和时间指标真实站得住时再写效率结论。
7. 只有当 exact local UPGrad 主干验证有效后，才重新考虑加入 AdaptiveRescaling、local/global momentum、alignment-aware weighting、stochastic Gramian 或 weight reuse。

## 五、参数设置

### 5.1 MultiMNIST 参数

| 参数 | NFJD | FMGDA / FedAvg+LS / PCGrad / CAGrad |
|------|------|------------------------------------|
| num_rounds | 50 | 50 |
| local_epochs | 3 | 3 |
| learning_rate | 0.001 | 0.001 |
| batch_size | 256 | 256 |
| participation_rate | 0.5 | 0.5 |
| num_clients | 10 | 10 |
| 联邦外壳 | NFJD Δθ aggregation | FMGDA objective-wise aggregation / FedAvg local-delta wrapper |
| 本地多目标规则 | NFJD | FMGDA / LS / PCGrad / CAGrad |
| exact_upgrad | True | N/A |
| use_objective_normalization | True | N/A |
| use_adaptive_rescaling | False | N/A |
| use_stochastic_gramian | False | N/A |
| local_momentum_beta | 0.0 | N/A |
| global_momentum_beta | 0.0 | N/A |

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

### 5.2 River Flow / CelebA 参数

| 参数 | NFJD | FMGDA / FedAvg+LS / PCGrad / CAGrad |
|------|------|------------------------------------|
| num_rounds | 50 | 50 |
| local_epochs | 3 | 3 |
| participation_rate | 0.5 | 0.5 |
| num_clients | 10 | 10 |
| 联邦外壳 | NFJD Δθ aggregation | FMGDA objective-wise aggregation / FedAvg local-delta wrapper |
| exact_upgrad | True | N/A |
| use_objective_normalization | True | N/A |
| use_adaptive_rescaling | False | N/A |
| use_stochastic_gramian | False | N/A |
| stochastic_subset_size | 4 | N/A |
| local_momentum_beta | 0.0 | N/A |
| global_momentum_beta | 0.0 | N/A |

补充：

- River Flow: `learning_rate=0.001`, `batch_size=256`
- CelebA: `learning_rate=0.0001`, `batch_size=256`

**模型架构**：多层MLP（共享骨干 + 8个回归头）

```
共享层:
  Linear(input_dim, 128) → ReLU
  Linear(128, 128) → ReLU
  Linear(128, 64) → ReLU

任务头 i (i=1..8): Linear(64, 1)
```

总参数量：约 30K（共享层 27K + 每头 65 × 8 = 520）

**CelebA 模型架构**：共享 CNN trunk + 显式 attribute heads

```
共享层:
  Conv/Pool stack -> Flatten -> Linear(256*4*4, 512) -> ReLU -> Dropout

任务头 i:
  Linear(512, 1)
```

这样 `PCGrad/CAGrad` 可以像官方多任务代码那样，只对共享参数做梯度组合，而不把任务特定 head 混入 surgery。

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
| Phase 5 总入口 | `experiments/nfjd_phases/run_phase5_suite.py` | 顺序运行 MultiMNIST / CelebA / RiverFlow |
| Phase 5 MultiMNIST | `experiments/nfjd_phases/run_phase5_multimnist.py` | MultiMNIST 正式 baseline 套件 |
| Phase 5 CelebA | `experiments/nfjd_phases/run_phase5_celeba.py` | CelebA 正式 baseline 套件 |
| Phase 5 RiverFlow | `experiments/nfjd_phases/run_phase5_riverflow.py` | RiverFlow 正式 baseline 套件 |

## 七、预期结果分析

### 7.1 MultiMNIST 预期

| 方法 | 预期 Avg Accuracy | 预期通信量(每客户端/轮) |
|------|-------------------|----------------------|
| NFJD | 93-96% | ~2.1MB (d×4B, d≈533K) |
| FMGDA | 原生联邦 MOO 强基线 | ~2.1MB |
| FedAvg+PCGrad | 强基线 | ~2.1MB |
| FedAvg+CAGrad | 强基线 | ~2.1MB |
| FedAvg+LS | 标准标量化基线 | ~2.1MB |

**关键预期**：
- NFJD 主线先验证 exact local UPGrad + objective normalization 这条最强骨架本身是否有效
- Phase 5 重点验证“同级通信预算下的性能优势”，而不是依赖 Jacobian 上传差距
- 若该主干仍不占优，则后续不应继续盲目叠加 server 侧启发式

### 7.2 River Flow / CelebA 预期

| 方法 | 预期 Avg MSE | 预期通信量(每客户端/轮) |
|------|-------------|----------------------|
| NFJD | 最优或并列最优 | ~120KB (d×4B, d≈30K) |
| FMGDA | 原生联邦 MOO 强基线 | ~120KB |
| FedAvg+PCGrad | 强基线 | ~120KB |
| FedAvg+CAGrad | 强基线 | ~120KB |
| FedAvg+LS | 标准标量化基线 | ~120KB |

**关键预期**：
- NFJD 与正式 baseline 具有相近上传量，但应先验证 exact local UPGrad 主干是否能改善测试指标
- stochastic / momentum / alignment 等增强项暂不作为主表默认配置
- River Flow 的任务量纲不一致更强，因此 objective normalization 是否有效是当前最关键观察点之一

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
| NFJD vs FMGDA | Avg Accuracy | NFJD > baseline | Wilcoxon p < 0.05 |
| NFJD vs FedAvg+PCGrad | Avg Accuracy | NFJD > baseline | Wilcoxon p < 0.05 |
| NFJD vs FedAvg+CAGrad | Avg Accuracy | NFJD > baseline | Wilcoxon p < 0.05 |
| NFJD vs FedAvg+LS | Avg Accuracy | NFJD > baseline | Wilcoxon p < 0.05 |
| NFJD vs all baselines | upload_per_client | NFJD ≈ baseline | Δθ 上传同量级，比较性能与稳定性 |

**River Flow / CelebA（回归或多标签分类）**：

| 对比 | 指标 | 预期方向 | 验证方式 |
|------|------|----------|----------|
| NFJD vs FMGDA | Avg MSE / Avg F1 | NFJD 更优 | Wilcoxon p < 0.05 |
| NFJD vs FedAvg+PCGrad | Avg MSE / Avg F1 | NFJD 更优 | Wilcoxon p < 0.05 |
| NFJD vs FedAvg+CAGrad | Avg MSE / Avg F1 | NFJD 更优 | Wilcoxon p < 0.05 |
| NFJD vs FedAvg+LS | Avg MSE / Avg F1 | NFJD 更优 | Wilcoxon p < 0.05 |
| NFJD vs all baselines | upload_per_client | NFJD ≈ baseline | Δθ 上传同量级，比较性能与稳定性 |

#### 7.3.3 综合排名验证

使用 Friedman + Nemenyi 检验验证 NFJD 在所有实验配置上的综合排名：

1. 对每个 (数据集, 划分方式, 种子) 组合，按 Avg Accuracy/MSE/F1 对 5 个方法排名
2. Friedman 检验：验证 5 个方法间存在显著差异（p < 0.05）
3. Nemenyi 事后检验：验证 NFJD 的平均排名显著优于其他方法
4. 临界差(CD)图：可视化方法间排名差异的显著性

#### 7.3.4 通信效率综合验证

| 维度 | 验证内容 | 通过标准 |
|------|----------|----------|
| 绝对通信量 | Phase 5 所有方法均为 Δθ 上传 | 重点比较性能、稳定性与收敛质量 |
| 通信量与m的关系 | NFJD 通信量与官方本地优化 baseline 同量级 | upload_per_client 均约为 d×4B |
| 通信-性能权衡 | NFJD 在相近通信预算下性能更优 | 同级通信下取得更好任务指标 |

#### 7.3.5 验证结果汇总表模板

实验完成后，将生成如下汇总表：

**MultiMNIST 性能对比（IID）**：

| 方法 | Avg Acc (mean±std) | vs NFJD p-value | 排名 |
|------|-------------------|-----------------|------|
| NFJD | xx.x ± x.x | — | 1 |
| FedAvg+MGDA-UB | xx.x ± x.x | 0.xxx | ? |
| FedAvg+PCGrad | xx.x ± x.x | 0.xxx | ? |
| FedAvg+CAGrad | xx.x ± x.x | 0.xxx | ? |
| FedAvg+LS | xx.x ± x.x | 0.xxx | ? |

**River Flow 性能对比（随机划分）**：

| 方法 | Avg MSE (mean±std) | vs NFJD p-value | 排名 |
|------|-------------------|-----------------|------|
| NFJD | x.xxx ± x.xxx | — | 1 |
| FedAvg+MGDA-UB | x.xxx ± x.xxx | 0.xxx | ? |
| FedAvg+PCGrad | x.xxx ± x.xxx | 0.xxx | ? |
| FedAvg+CAGrad | x.xxx ± x.xxx | 0.xxx | ? |
| FedAvg+LS | x.xxx ± x.xxx | 0.xxx | ? |

### 7.4 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| RTX 3050 仅 4GB 显存 | LeNet-MTL 可能 OOM | 减小 batch_size 或使用梯度累积 |
| River Flow 数据获取困难 | 无法开始实验 | 提前下载数据，准备备用链接 |
| MultiMNIST 生成方式差异导致结果不可比 | 与文献数字不一致 | 严格遵循 Sener & Koltun 2018 的生成方式 |
| m=8 时 NFJD 客户端计算过慢 | 实验时间过长 | StochasticGramian subset_size=4 降低计算量 |

## 八、实验输出

### 8.1 结果文件

- `results/nfjd_phase5/multimnist/phase5_multimnist_results.csv` — MultiMNIST 结果
- `results/nfjd_phase5/celeba/phase5_celeba_results.csv` — CelebA 结果
- `results/nfjd_phase5/riverflow/phase5_riverflow_results.csv` — RiverFlow 结果
- `results/nfjd_phase5/phase5_suite.log` — Phase 5 总入口日志

### 8.2 CSV 字段

在 Phase 1-4 字段基础上新增：

| 字段 | 说明 |
|------|------|
| dataset | multimnist / celeba / riverflow |
| data_split | iid / noniid |
| model_arch | lenet_mtl / celeba_cnn / river_flow_mlp |
| method_display_name | 可展示的方法名 |
| method_family | proposed / official_baseline |
| source_paper | baseline 论文标题 |
| source_paper_url | baseline 论文链接 |
| source_official_repo | baseline 官方实现仓库 |
| avg_accuracy | 平均分类准确率 |
| avg_mse | River Flow 平均 MSE |
| max_mse | River Flow 最大 MSE |
| mse_std | River Flow MSE 标准差 |
