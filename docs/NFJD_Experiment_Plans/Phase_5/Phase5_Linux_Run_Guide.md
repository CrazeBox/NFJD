# Phase 5 Linux 服务器运行指南

正式 baseline 来源绑定见：`Phase5_Official_Baselines.md`

## 1. 环境准备

```bash
# 创建 conda 环境
conda create -n nfjd python=3.10 -y
conda activate nfjd

# 安装 PyTorch（根据服务器CUDA版本选择）
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 验证
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
```

## 2. 获取代码

```bash
cd /path/to/workspace
git clone https://github.com/CrazeBox/NFJD.git
cd NFJD
```

## 3. 准备数据

### 3.1 MultiMNIST

MultiMNIST 数据会自动从 torchvision 下载 MNIST 并生成叠加图像，无需手动操作。

首次运行时会自动下载到 `data/multimnist/` 目录。

### 3.2 River Flow

需要手动下载 River Flow 数据集：

```bash
mkdir -p data/river_flow

# 方式1：从 UCI 下载
# 访问 https://archive.ics.uci.edu/ 搜索 "River Flow" 或 "Flood"
# 下载 CSV 文件，放置到 data/river_flow/river_flow.csv

# 方式2：如果 UCI 不可用，可从以下备选来源获取
# - Nash-MTL 论文的 GitHub 仓库: https://github.com/AvivNavon/nash-mtl
# - MTL-Benchmark 仓库: https://github.com/jiangfeng1124/MTL-Benchmark
```

**CSV 格式要求**：
- 第一行为表头（会被跳过）
- 前 N 列为输入特征
- 最后 8 列为 8 个河流站点的流量目标值
- 逗号分隔

## 4. 运行实验

```bash
# 激活环境
conda activate nfjd

# 确保在项目根目录
cd /path/to/workspace/NFJD

# 运行全部 Phase 5 实验（117 次，分 3 个数据集脚本顺序执行）
nohup python -m fedjd.experiments.nfjd_phases.run_phase5_suite > p5_stdout.log 2>&1 &

# 查看进度
tail -f results/nfjd_phase5/phase5_suite.log

# 查看进程
ps aux | grep run_phase5_suite
```

### 4.1 分组运行（推荐）

如果服务器资源有限或想分批运行，可以修改脚本中的 `experiments` 列表，只保留特定实验组：

```python
# 只跑 MultiMNIST 实验：注释掉 riverflow 相关的 experiments.append()
# 只跑 River Flow 实验：注释掉 multimnist 相关的 experiments.append()
```

或者使用命令行参数（需要自行添加 argparse 支持）。

### 4.2 GPU 指定

```bash
# 使用第 0 块 GPU
CUDA_VISIBLE_DEVICES=0 python -m fedjd.experiments.nfjd_phases.run_phase5_suite

# 使用第 1 块 GPU
CUDA_VISIBLE_DEVICES=1 python -m fedjd.experiments.nfjd_phases.run_phase5_suite
```

## 5. 需要回传的结果

实验完成后，请将以下文件/目录打包回传：

```bash
# 打包结果
tar czf phase5_results.tar.gz results/nfjd_phase5/
```

### 必须回传的文件：

| 文件 | 说明 |
|------|------|
| `results/nfjd_phase5/multimnist/phase5_multimnist_results.csv` | MultiMNIST 正式 baseline 结果 |
| `results/nfjd_phase5/celeba/phase5_celeba_results.csv` | CelebA 正式 baseline 结果 |
| `results/nfjd_phase5/riverflow/phase5_riverflow_results.csv` | RiverFlow 正式 baseline 结果 |
| `results/nfjd_phase5/phase5_suite.log` | Phase 5 总入口日志 |

### 可选回传的文件：

| 文件 | 说明 |
|------|------|
| `p5_stdout.log` | nohup 标准输出日志 |
| `data/multimnist/` | MultiMNIST 数据（如果需要复现） |
| `data/river_flow/river_flow.csv` | River Flow 原始数据 |

### CSV 关键字段说明：

| 字段 | 说明 | 重要性 |
|------|------|--------|
| `method` | 方法名 (`nfjd` / `fedavg_ls` / `fedavg_mgda` / `fedavg_pcgrad` / `fedavg_cagrad`) | 核心 |
| `method_display_name` | 可展示的方法名 | 核心 |
| `source_paper` | baseline 论文标题 | 核心 |
| `source_official_repo` | baseline 官方实现仓库 | 核心 |
| `dataset` | 数据集 (multimnist/celeba/riverflow) | 核心 |
| `data_split` | 划分方式 (iid/noniid) | 核心 |
| `m` | 目标数 | 核心 |
| `avg_ri` | 平均相对改善 | 核心 |
| `upload_per_client` | 每客户端每轮上传字节数 | 核心 |
| `avg_round_time` | 每轮平均耗时 | 核心 |
| `avg_accuracy` | MultiMNIST 平均准确率 | 核心 |
| `avg_mse` / `max_mse` / `mse_std` | River Flow MSE 指标 | 核心 |
| `total_local_steps` | 总本地步数 | 公平对比 |
| `fair_comparison` | Phase 5 中固定为 `False`，不再使用旧 fair 标记实验 | 配置 |
| `local_epochs` | 本地 epoch 数 | 配置 |
| `num_rounds` | 总通信轮数 | 配置 |

## 6. 结果验证

运行完成后，在服务器上快速验证：

```bash
# 检查实验数量
wc -l results/nfjd_phase5/multimnist/phase5_multimnist_results.csv
wc -l results/nfjd_phase5/celeba/phase5_celeba_results.csv
wc -l results/nfjd_phase5/riverflow/phase5_riverflow_results.csv

# 检查各方法实验数
cut -d',' -f2 results/nfjd_phase5/multimnist/phase5_multimnist_results.csv | sort | uniq -c

# 检查是否有 NaN
grep -i "nan\|inf" results/nfjd_phase5/multimnist/phase5_multimnist_results.csv

# 快速查看 NFJD vs 其他方法的 RI
python -c "
import csv
rows = list(csv.DictReader(open('results/nfjd_phase5/multimnist/phase5_multimnist_results.csv')))
for method in ['nfjd', 'fedavg_ls', 'fedavg_mgda', 'fedavg_pcgrad', 'fedavg_cagrad']:
    sub = [r for r in rows if r['method']==method]
    if sub:
        ri = sum(float(r['avg_ri']) for r in sub)/len(sub)
        print(f'{method:18s} avg_RI={ri:.4f} n={len(sub)}')
"
```
