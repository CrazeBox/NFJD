# Phase 5 正式 Baseline 绑定说明

## 1. 适用范围

本说明仅适用于 Phase 5 主实验：

- `experiments/nfjd_phases/run_phase5_multimnist.py`
- `experiments/nfjd_phases/run_phase5_celeba.py`
- `experiments/nfjd_phases/run_phase5_riverflow.py`
- `experiments/nfjd_phases/run_phase5_suite.py`

Phase 5 不再使用 `fedjd / fmgda / weighted_sum / direction_avg / stl` 作为对外主表 baseline。

当前对外主表 baseline 全部替换为“**官方多任务优化算法 + 统一联邦 FedAvg wrapper**”。

## 2. Baseline 列表

| 代码名 | 展示名 | 论文来源 | 官方实现来源 | 在本仓库中的联邦化方式 |
|------|------|------|------|------|
| `fedavg_ls` | FedAvg+LS | 线性标量化基线 | Nash-MTL 官方代码库的 `ls` baseline API: <https://github.com/AvivNavon/nash-mtl> | 每个客户端本地做等权损失求和更新，服务器做样本数加权 FedAvg |
| `fedavg_mgda` | FedAvg+MGDA-UB | Sener and Koltun, NeurIPS 2018, *Multi-Task Learning as Multi-Objective Optimization* | <https://github.com/isl-org/MultiObjectiveOptimization> | 每个客户端本地对任务梯度做 MGDA-UB min-norm 组合，再上传模型增量 |
| `fedavg_pcgrad` | FedAvg+PCGrad | Yu et al., ICLR 2020, *Gradient Surgery for Multi-Task Learning* | <https://github.com/tianheyu927/PCGrad> | 每个客户端本地做 PCGrad 投影修正，再上传模型增量 |
| `fedavg_cagrad` | FedAvg+CAGrad | Liu et al., NeurIPS 2021, *Conflict-Averse Gradient Descent for Multi-task Learning* | <https://github.com/Cranial-XIX/CAGrad> | 每个客户端本地做 CAGrad 方向求解，再上传模型增量 |

## 3. 重要说明

1. 这些 baseline 的“**多任务梯度组合规则**”绑定到了明确论文与官方代码来源。
2. 这些 baseline 的“**联邦训练外壳**”不是外部官方仓库直接提供的，因为上述官方仓库本身是中心化 MTL 代码，而不是联邦学习代码。
3. 为保证公平性，Phase 5 对所有方法统一采用：
   - 相同模型架构
   - 相同数据划分
   - 相同 `num_rounds`
   - 相同 `local_epochs`
   - 相同 `learning_rate`
   - 相同 `participation_rate`
4. 因此，Phase 5 最准确的表述应是：

> NFJD is compared against federated wrappers of officially released multi-task optimization baselines.

而不是：

> NFJD is compared against end-to-end official federated implementations.

后者并不准确。

## 4. 代码绑定位置

| 功能 | 文件 |
|------|------|
| Baseline 元数据绑定 | `core/phase5_official_baselines.py` |
| Phase 5 训练器构建 | `experiments/nfjd_phases/phase5_utils.py` |
| Phase 5 多数据集总入口 | `experiments/nfjd_phases/run_phase5_suite.py` |

## 5. 结果 CSV 中的来源字段

Phase 5 结果 CSV 现包含以下可追溯字段：

- `method_display_name`
- `method_family`
- `source_paper`
- `source_paper_url`
- `source_official_repo`
- `source_note`

这些字段用于确保每个 baseline 的来源可以在结果表中直接追溯，不需要事后人工补充。
