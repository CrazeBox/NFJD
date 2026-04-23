# Phase 5 正式 Baseline 绑定说明

## 1. 适用范围

本说明仅适用于 Phase 5 主实验：

- `experiments/nfjd_phases/run_phase5_multimnist.py`
- `experiments/nfjd_phases/run_phase5_celeba.py`
- `experiments/nfjd_phases/run_phase5_riverflow.py`
- `experiments/nfjd_phases/run_phase5_suite.py`

Phase 5 不再使用 `fedjd / weighted_sum / direction_avg / stl` 作为对外主表 baseline。

当前对外主表方法由一条原生联邦 MOO baseline 和三条统一联邦适配 baseline 组成。

## 2. Baseline 列表

| 代码名 | 展示名 | 论文来源 | 官方实现来源 | 在本仓库中的联邦化方式 |
|------|------|------|------|------|
| `fmgda` | FMGDA | Yang et al., NeurIPS 2023, *Federated Multi-Objective Learning* | 论文页: <https://arxiv.org/abs/2310.09866> | 每个客户端按 objective 分别执行本地轨迹更新，服务器按客户端样本权重聚合 objective 梯度，再做 MGDA min-norm 求解 |
| `fedavg_ls` | FedAvg+LS | 线性标量化基线 | Nash-MTL 官方代码库的 `ls` baseline API: <https://github.com/AvivNavon/nash-mtl> | 每个客户端本地直接优化 `sum(loss_i)`，不做额外 `1/m` 归一化，然后上传模型增量 |
| `fedavg_pcgrad` | FedAvg+PCGrad | Yu et al., ICLR 2020, *Gradient Surgery for Multi-Task Learning* | <https://github.com/tianheyu927/PCGrad> | 每个客户端仅在共享 trunk 参数上做 PCGrad surgery；任务 head 按 summed loss 的普通梯度更新，再上传模型增量 |
| `fedavg_cagrad` | FedAvg+CAGrad | Liu et al., NeurIPS 2021, *Conflict-Averse Gradient Descent for Multi-task Learning* | <https://github.com/Cranial-XIX/CAGrad> | 每个客户端仅在共享 trunk 参数上做 task-space CAGrad；使用官方常用 `c=0.4` 与 rescale 约定，任务 head 按 summed loss 更新 |

## 3. 重要说明

1. `FMGDA` 是原生联邦多目标 baseline；其当前实现已按论文中的 per-objective local trajectory + sample-weighted server aggregation 形式对齐。
2. `FedAvg+LS / PCGrad / CAGrad` 的“**多任务梯度组合规则**”绑定到了明确论文与官方代码来源。
3. `PCGrad / CAGrad` 当前实现遵循官方 multitask 用法：共享参数做 gradient surgery / CAGrad，任务特定 head 走普通 summed-loss 梯度。
4. 为了让 `CelebA` 也满足 shared-trunk + per-task-head 结构，Phase 5 的 `CelebaCNN` 已重构为显式多头输出。
5. `FedAvg+LS / PCGrad / CAGrad` 的“**联邦训练外壳**”不是外部官方仓库直接提供的，因为上述官方仓库本身是中心化 MTL 代码，而不是联邦学习代码。
6. 为保证公平性，Phase 5 对所有方法统一采用：
   - 相同模型架构
   - 相同数据划分
   - 相同 `num_rounds`
   - 相同 `local_epochs`
   - 相同 `learning_rate`
   - 相同 `participation_rate`
7. 因此，Phase 5 最准确的表述应是：

> NFJD is compared against one native federated multi-objective baseline (FMGDA) and federated adaptations of officially released multi-task optimization baselines.

而不是：

> NFJD is compared against end-to-end official federated implementations.

后者并不准确。

## 4. 代码绑定位置

| 功能 | 文件 |
|------|------|
| Baseline 元数据绑定 | `core/phase5_official_baselines.py` |
| Phase 5 训练器构建 | `experiments/nfjd_phases/phase5_utils.py` |
| Phase 5 多数据集总入口 | `experiments/nfjd_phases/run_phase5_suite.py` |

## 5. 跑完当前程序后的待办

1. 对 `FMGDA / FedAvg+LS / FedAvg+PCGrad / FedAvg+CAGrad` 做小规模调参。
2. 对 `FMGDA` 检查 `eta_L` 与 `eta_t` 分离后的稳定性。
3. 结果汇总表补 `mean ± std` 与统计显著性检验。
4. 论文表述中继续保持 `FMGDA` 与 FedAvg-style adaptations 的身份区分。

## 6. 结果 CSV 与来源追溯

当前 Phase 5 结果 CSV 主要保留训练配置、通信、时间和任务指标，不再把来源元数据单独展开成列。

当前来源追溯方式为：

1. 方法名与展示名绑定在 `core/phase5_official_baselines.py`
2. 训练入口绑定在 `experiments/nfjd_phases/phase5_utils.py`
3. 本文档作为 Phase 5 baseline 身份与来源的文字说明

如果后续论文整理需要，也可以在不影响当前跑数流程的前提下，把来源元数据列重新加入 CSV。
