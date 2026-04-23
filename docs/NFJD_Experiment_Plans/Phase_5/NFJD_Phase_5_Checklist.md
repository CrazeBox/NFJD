# NFJD Phase 5: 真实数据集基准实验检查清单

## A. 实验前准备

| # | 检查项 | 通过标准 | 状态 |
|---|--------|----------|------|
| A1 | CUDA版PyTorch已安装且可用 | `torch.cuda.is_available() == True` | ⬜ |
| A2 | GPU显存足够运行实验 | RTX 3050 4GB，LeNet-MTL batch_size=64 可运行 | ⬜ |
| A3 | torchvision 已安装 | `import torchvision` 无报错 | ⬜ |
| A4 | 磁盘空间充足 | MultiMNIST + River Flow 数据 < 1GB，结果文件 < 100MB | ⬜ |
| A5 | 代码已同步到最新版本 | GPU device 自动检测已生效，ConflictAwareMomentum 已集成 | ⬜ |

## B. 数据预处理

### B1. MultiMNIST 数据

| # | 检查项 | 通过标准 | 状态 |
|---|--------|----------|------|
| B1.1 | MNIST 原始数据下载成功 | torchvision.datasets.MNIST 可加载 | ⬜ |
| B1.2 | MultiMNIST 生成逻辑正确 | 图像尺寸 36×36，两张数字叠加，标签为 (L, R) 二元组 | ⬜ |
| B1.3 | 训练集数量正确 | 60,000 张训练图像 | ⬜ |
| B1.4 | 测试集数量正确 | 10,000 张测试图像 | ⬜ |
| B1.5 | IID 联邦划分正确 | 10 客户端，每客户端 ~6,000 张，各客户端类别分布均匀 | ⬜ |
| B1.6 | Non-IID 联邦划分正确 | 每客户端仅持有 2-3 个数字类别，类别分布不均 | ⬜ |
| B1.7 | 数据格式与框架兼容 | `FederatedData` 格式，inputs 形状 (B,1,36,36)，targets 形状 (B,2) | ⬜ |
| B1.8 | 可视化验证 | 随机抽样 5 张图像，可肉眼辨认两个重叠数字 | ⬜ |

### B2. River Flow 数据

| # | 检查项 | 通过标准 | 状态 |
|---|--------|----------|------|
| B2.1 | 数据下载成功 | UCI River Flow 数据集可获取 | ⬜ |
| B2.2 | 数据维度正确 | ~7,813 样本 × (6~8 特征 + 8 目标) | ⬜ |
| B2.3 | 缺失值处理 | 检查并处理 NaN/Inf 值 | ⬜ |
| B2.4 | 特征标准化 | 对输入特征做 Z-score 标准化（训练集统计量） | ⬜ |
| B2.5 | 目标值处理 | 对 8 个目标值做标准化或对数变换（视分布而定） | ⬜ |
| B2.6 | 时间序列划分 | 按时间顺序前 80% 训练 / 后 20% 测试 | ⬜ |
| B2.7 | 随机联邦划分正确 | 10 客户端，每客户端 ~625 样本 | ⬜ |
| B2.8 | Non-IID 联邦划分正确 | 按目标值排序分块后分配，客户端目标分布明显偏斜 | ⬜ |
| B2.9 | 数据格式与框架兼容 | `FederatedRegressionData` 格式，inputs (B, features)，targets (B, 8) | ⬜ |

## C. 模型配置确认

### C1. LeNet-MTL（MultiMNIST）

| # | 检查项 | 通过标准 | 状态 |
|---|--------|----------|------|
| C1.1 | 模型结构正确 | 共享卷积骨干 + 2 个独立分类头 | ⬜ |
| C1.2 | 输入输出维度匹配 | 输入 (B,1,36,36)，输出 (B,2,10) | ⬜ |
| C1.3 | 参数量合理 | ~533K 参数 | ⬜ |
| C1.4 | GPU 上可运行 | 模型 .to(device) 无报错，前向传播无 OOM | ⬜ |
| C1.5 | 梯度计算正确 | 2 个目标分别 backward，Jacobian 形状 (2, 533K) | ⬜ |
| C1.6 | 与 multi_task_classification 兼容 | objective_fn 输出 2 个 loss | ⬜ |

### C2. RiverFlowMLP（River Flow）

| # | 检查项 | 通过标准 | 状态 |
|---|--------|----------|------|
| C2.1 | 模型结构正确 | 共享 MLP 骨干 + 8 个独立回归头 | ⬜ |
| C2.2 | 输入输出维度匹配 | 输入 (B, input_dim)，输出 (B, 8) | ⬜ |
| C2.3 | 参数量合理 | ~30K 参数 | ⬜ |
| C2.4 | GPU 上可运行 | 前向传播无 OOM | ⬜ |
| C2.5 | 梯度计算正确 | 8 个目标分别 backward，Jacobian 形状 (8, ~30K) | ⬜ |
| C2.6 | 与 multi_objective_regression 兼容 | objective_fn 输出 8 个 loss | ⬜ |

## D. 评估流程验证

| # | 检查项 | 通过标准 | 状态 |
|---|--------|----------|------|
| D1 | NFJD 在 MultiMNIST 上可运行 | 50 轮训练完成，无 NaN，两个任务准确率 > 85% | ⬜ |
| D2 | NFJD 在 River Flow / CelebA 上可运行 | 50 轮训练完成，无 NaN，指标持续改善 | ⬜ |
| D3 | FedAvg+MGDA-UB 可运行 | 所有数据集 50 轮训练可完成，无异常退出 | ⬜ |
| D4 | FedAvg+PCGrad 可运行 | 所有数据集 50 轮训练可完成，无异常退出 | ⬜ |
| D5 | FedAvg+CAGrad 可运行 | 所有数据集 50 轮训练可完成，无异常退出 | ⬜ |
| D6 | FedAvg+LS 可运行 | 所有数据集 50 轮训练可完成，无异常退出 | ⬜ |
| D7 | 分类准确率计算正确 | MultiMNIST / CelebA 指标与手动计算一致 | ⬜ |
| D8 | 回归 MSE 计算正确 | RiverFlow 各任务 MSE 与手动计算一致 | ⬜ |
| D9 | 来源字段记录正确 | 结果 CSV 中 baseline 来源字段非空 | ⬜ |
| D10 | GPU 训练速度提升 | GPU 上每轮耗时优于 CPU | ⬜ |

## E. 结果记录规范

| # | 检查项 | 通过标准 | 状态 |
|---|--------|----------|------|
| E1 | CSV 字段完整 | 包含 `method_display_name/source_paper/source_official_repo` 等来源字段 | ⬜ |
| E2 | 每次实验独立可复现 | 同种子同配置结果一致 | ⬜ |
| E3 | 异常实验标记 | 失败/发散实验记录原因，不丢弃 | ⬜ |
| E4 | 中间结果保存 | 每 10 轮记录一次目标值，用于收敛曲线绘制 | ⬜ |
| E5 | GPU 内存使用记录 | 记录峰值 GPU 内存使用量 | ⬜ |

## F. MultiMNIST 性能检查

| # | 检查项 | 通过标准 | 状态 |
|---|--------|----------|------|
| F1 | NFJD Avg Accuracy > FedAvg+MGDA-UB Avg Accuracy | IID 设置下，Wilcoxon p < 0.05 | ⬜ |
| F2 | NFJD Avg Accuracy > FedAvg+PCGrad Avg Accuracy | IID 设置下，Wilcoxon p < 0.05 | ⬜ |
| F3 | NFJD Avg Accuracy > FedAvg+CAGrad Avg Accuracy | IID 设置下，Wilcoxon p < 0.05 | ⬜ |
| F4 | NFJD Avg Accuracy > FedAvg+LS Avg Accuracy | IID 设置下，Wilcoxon p < 0.05 | ⬜ |
| F5 | NFJD 与正式 baseline 通信量同量级 | 全部均为 Δθ 上传，比较性能与稳定性 | ⬜ |
| F6 | Non-IID 下 NFJD 仍优于所有正式 baseline | Avg Accuracy 排名第1，Wilcoxon p < 0.05 | ⬜ |
| F7 | 所有方法 all_decreased=True | 两个任务准确率均高于初始值 | ⬜ |
| F8 | 收敛曲线平滑 | 无剧烈振荡或发散 | ⬜ |

## G. River Flow 性能检查

| # | 检查项 | 通过标准 | 状态 |
|---|--------|----------|------|
| G1 | NFJD Avg MSE < FedAvg+MGDA-UB Avg MSE | 随机划分，Wilcoxon p < 0.05 | ⬜ |
| G2 | NFJD Avg MSE < FedAvg+PCGrad Avg MSE | 随机划分，Wilcoxon p < 0.05 | ⬜ |
| G3 | NFJD Avg MSE < FedAvg+CAGrad Avg MSE | 随机划分，Wilcoxon p < 0.05 | ⬜ |
| G4 | NFJD Avg MSE < FedAvg+LS Avg MSE | 随机划分，Wilcoxon p < 0.05 | ⬜ |
| G5 | m=8 时 StochasticGramian 有效 | subset_size=4 vs 全量8，MSE 差距 < 5%，速度提升显著 | ⬜ |
| G6 | Non-IID 划分下 NFJD 仍优于所有正式 baseline | Avg MSE 排名第1，Wilcoxon p < 0.05 | ⬜ |
| G7 | 所有 8 个目标均下降 | per_task_mse 均低于初始值 | ⬜ |
| G8 | MSE 标准差合理 | 无单任务 MSE 异常高（> 平均值 3 倍） | ⬜ |

## H. 统计验证检查

| # | 检查项 | 通过标准 | 状态 |
|---|--------|----------|------|
| H1 | Friedman 检验：5个方法间存在显著差异 | p < 0.05 | ⬜ |
| H2 | Nemenyi 事后检验已完成 | 形成正式多方法排名比较 | ⬜ |
| H3 | NFJD vs FedAvg+MGDA-UB 配对 Wilcoxon 显著 | p < 0.05（性能指标） | ⬜ |
| H4 | NFJD vs FedAvg+PCGrad 配对 Wilcoxon 显著 | p < 0.05（性能指标） | ⬜ |
| H5 | NFJD vs FedAvg+CAGrad 配对 Wilcoxon 显著 | p < 0.05（性能指标） | ⬜ |
| H6 | NFJD vs FedAvg+LS 配对 Wilcoxon 显著 | p < 0.05（性能指标） | ⬜ |
| H7 | CD图（临界差图）已生成 | 可视化方法间排名差异 | ⬜ |
| H8 | 验证结果汇总表已填写 | 包含 mean±std, p-value, 排名 | ⬜ |

## I. 与文献对齐

| # | 检查项 | 通过标准 | 状态 |
|---|--------|----------|------|
| I1 | MultiMNIST 生成方式与 Sener & Koltun 2018 一致 | 偏移范围、叠加方式相同 | ⬜ |
| I2 | Phase 5 baseline 来源字段已写入 CSV | `source_paper/source_official_repo` 非空 | ⬜ |
| I3 | PCGrad/CAGrad/MGDA-UB/LS 的文档来源可追溯 | 与 `Phase5_Official_Baselines.md` 一致 | ⬜ |
| I4 | River Flow 预处理与 Phase 5 文档一致 | train/val/test 统一标准化策略 | ⬜ |

## 通过判定

| 条件 | 要求 | 实际 | 通过 |
|------|------|------|------|
| 实验前准备 | A1-A5 全部通过 | | ⬜ |
| 数据预处理 | B1.1-B2.9 全部通过 | | ⬜ |
| 模型配置 | C1.1-C2.6 全部通过 | | ⬜ |
| 评估流程 | D1-D10 全部通过 | | ⬜ |
| 结果记录 | E1-E5 全部通过 | | ⬜ |
| MultiMNIST 性能 | F1-F13 全部通过 | | ⬜ |
| River Flow / CelebA 性能 | G1-G8 全部通过 | | ⬜ |
| 统计验证 | H1-H8 全部通过 | | ⬜ |
| 文献对齐 | I1-I4 全部通过 | | ⬜ |

**Phase 5 总判定: ⬜ PASS / ⬜ FAIL**
