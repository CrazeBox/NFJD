# FedClient-UPGrad 论文正式大纲（中文稿）

这份文档是论文写作蓝本的中文版本，用来把方法设计、理论论证、实验设置和当前结果组织成一条连贯主线。英文版保留在 `fedjd/docs/paper_outline_formal.md`。

## 论文题目

推荐题目：

```text
FedClient-UPGrad：面向公平联邦学习的客户端级多目标梯度聚合方法
```

备选题目：

1. 基于客户端级多目标优化的联邦学习更新聚合方法
2. FedClient-UPGrad：通过客户端更新空间聚合提升尾部客户端性能
3. 面向异构客户端的冲突感知联邦学习聚合方法

建议：如果论文强调理论和方法框架，用第一个题目；如果投偏应用的联邦学习会议或期刊，可以突出“尾部客户端性能”。

## 论文核心主张

异构联邦学习不应该只被看作一个平均损失最小化问题。每个客户端都有自己的数据分布和局部目标，因此可以把每个客户端的经验损失看作一个单独目标，从而形成客户端级多目标优化问题。

FedClient-UPGrad 的核心思想是：每轮通信中，客户端从同一个全局模型出发做本地训练，上传模型更新量；服务器把这些客户端更新看作各客户端目标的下降方向代理，然后用 UPGrad 在客户端更新空间中计算一个冲突感知的公共方向。这个方向用于更新全局模型，从而在保持平均性能的同时改善弱客户端和尾部客户端表现。

## 论文需要证明和展示的主张

本文要围绕四个主张展开：

1. **问题主张：** 客户端异构会导致目标冲突，只优化平均损失可能牺牲尾部客户端。
2. **方法主张：** 客户端本地更新量可以作为客户端目标梯度或下降方向的实用代理。
3. **理论主张：** UPGrad 可以在客户端更新空间中计算冲突感知的公共方向，并指向多目标一阶稳定性。
4. **实验主张：** FedClient-UPGrad 在 FEMNIST 和 CIFAR10 的客户端级指标上优于 FedAvg、qFedAvg 和 FedMGDA+，尤其改善 worst-10% 客户端准确率。

## 摘要应该写什么

摘要不是简单总结全文，而是压缩回答四个问题：

1. 本文解决什么问题？
异构联邦学习中，平均准确率无法反映尾部客户端和弱客户端的表现。

2. 本文提出什么核心想法？
把每个客户端损失看作一个目标，把联邦学习建模为客户端级多目标优化问题。

3. 本文方法是什么？
客户端上传普通 FedAvg 风格的模型更新量，服务器用这些更新量构造客户端目标方向矩阵，并用 UPGrad 计算公共更新方向。

4. 本文证据是什么？
FEMNIST 三种子实验中，FedClient-UPGrad 同时提升平均客户端准确率、worst-10% 客户端准确率，降低客户端准确率标准差和平均测试损失。CIFAR10 完整结果待补入。

摘要草稿：

```text
异构联邦学习通常以平均准确率作为主要评价指标，但在实际部署中，模型是否能服务低性能客户端同样重要。本文从客户端级多目标优化视角重新审视异构联邦学习，将每个客户端的经验损失视为一个独立目标。基于这一视角，我们提出 FedClient-UPGrad：每个被选客户端从服务器广播的全局模型出发进行普通本地训练，并上传模型更新量；服务器将负更新量作为客户端目标梯度代理，利用 UPGrad 在客户端更新空间中计算冲突感知的公共方向。该方法不需要显式计算客户端 Jacobian，通信形式与 FedAvg 风格的模型更新上传一致。我们在 FEMNIST 和 Dirichlet 划分的 CIFAR10 上比较 FedClient-UPGrad、FedAvg、qFedAvg 和 FedMGDA+。在 FEMNIST 三种子实验中，FedClient-UPGrad 获得 0.7996 ± 0.0094 的平均客户端准确率和 0.5902 ± 0.0058 的 worst-10% 客户端准确率，均显著优于各基线方法。这说明客户端级多目标聚合能够有效改善异构联邦学习中的尾部客户端表现。
```

## 1. Introduction

引言需要回答五个问题。

### 1.1 为什么这个问题重要

要强调的不是“联邦学习很热门”，而是这个具体问题有价值：

1. 联邦学习中的客户端天然异构，例如不同写字人、不同用户、不同设备或不同机构。
2. 非 IID 数据会导致客户端更新方向冲突。
3. 平均准确率高并不代表每个客户端都被优化得好。
4. 尾部客户端表现关系到公平性、鲁棒性和实际部署可靠性。
5. 因此，联邦学习方法应该同时关注平均性能和弱客户端性能。

可写成：

```text
在异构联邦学习中，一个全局模型即使具有较高平均准确率，也可能在部分客户端上表现很差。由于不同客户端的数据分布存在显著差异，优化平均目标可能使模型偏向多数客户端或易优化客户端，从而忽略尾部客户端。这说明仅以平均性能评价联邦模型是不充分的，有必要从客户端级目标平衡的角度重新设计聚合方法。
```

### 1.2 现有方法的代表思路

可以分四类介绍：

1. **FedAvg 类方法：** 简单高效，对客户端更新做加权平均，但本质上优化平均目标。
2. **公平联邦学习 / qFedAvg：** 根据客户端损失进行重加权，让高损失客户端获得更大权重。
3. **多目标联邦学习 / MGDA 类方法：** 试图寻找多个目标的公共下降方向，但可能需要显式梯度或计算较重。
4. **梯度冲突处理方法：** 通过投影、重组或几何约束处理目标冲突，但通常不是专门为客户端级 FL 更新设计。

### 1.3 现有方法的共同局限

共同局限可以概括为：

```text
现有方法要么把客户端目标压缩成一个平均目标，要么只通过标量损失重加权处理公平性，要么需要较昂贵的目标级梯度信息。它们没有充分利用标准联邦学习中已经自然产生的客户端本地更新方向。
```

### 1.4 本文核心思路

本文的桥梁是：

1. 每个客户端损失是一个目标。
2. 客户端本地训练更新量是该客户端目标下降方向的代理。
3. 服务器不再简单平均这些更新，而是在更新空间中用 UPGrad 计算公共方向。
4. 这个公共方向更关注客户端目标之间的冲突，有助于改善尾部客户端。

### 1.5 贡献总结

建议贡献写成：

1. 提出客户端级多目标联邦学习视角，将每个客户端经验损失视为一个目标。
2. 提出 FedClient-UPGrad，使用普通客户端本地更新作为目标方向代理，并用 UPGrad 进行冲突感知聚合。
3. 构建统一实验框架，在 FEMNIST 和 CIFAR10 上公平比较 FedAvg、qFedAvg、FedMGDA+ 和 FedClient-UPGrad。
4. 在 FEMNIST 三种子实验中，FedClient-UPGrad 在平均准确率、worst-10% 准确率、客户端准确率标准差和测试损失上均稳定最优。
5. 分析方法的效率代价，说明其以额外服务器端聚合计算换取更好的尾部客户端性能。

## 2. Related Work

相关工作不要写成文献堆砌，而要服务于本文 gap。

### 2.1 异构联邦优化

介绍 FedAvg 及非 IID 问题。强调 FedAvg 是基础强基线，但其加权平均更新可能被多数客户端主导。

### 2.2 公平联邦学习

介绍 qFedAvg/q-FFL 等方法。指出这类方法通过损失标量重加权改善公平性，而 FedClient-UPGrad 从客户端更新向量几何关系出发。

### 2.3 多目标优化与 MGDA

介绍 Pareto stationarity、公共下降方向、MGDA。说明 FedMGDA+ 是最接近的多目标客户端更新基线。

### 2.4 梯度冲突和投影方法

介绍梯度冲突、投影和 UPGrad 类思想。重点说明本文把这些思想用于联邦客户端更新空间。

## 3. Problem Formulation

### 3.1 联邦学习设置

有 `K` 个客户端。客户端 `i` 的本地目标为：

```text
F_i(theta) = E_{(x,y) in D_i}[ell(f_theta(x), y)]
```

FedAvg 近似优化：

```text
F_avg(theta) = sum_i p_i F_i(theta)
```

其中 `p_i` 通常与客户端样本量相关。

### 3.2 客户端级多目标视角

把所有客户端目标组成向量目标：

```text
F(theta) = [F_1(theta), F_2(theta), ..., F_K(theta)]
```

每轮只采样部分客户端 `S_t`，服务器目标不是只优化平均损失，而是寻找能更好平衡被采样客户端目标的更新方向。

### 3.3 本地更新作为梯度代理

每个客户端从同一个全局模型 `theta_t` 出发，本地训练得到 `theta_i^t`，上传：

```text
Delta_i^t = theta_i^t - theta_t
```

因为本地训练是沿着客户端损失下降，所以：

```text
-Delta_i^t
```

可以被看作客户端目标梯度或下降方向的代理。

### 3.4 评价目标

实验必须对应这个问题定义，因此不能只看平均准确率。主要指标应包括：

1. 平均客户端测试准确率。
2. worst-10% 客户端准确率。
3. 客户端准确率标准差。
4. 平均客户端测试损失。
5. 平均轮次时间和聚合计算时间。

## 4. Method: FedClient-UPGrad

### 4.1 算法流程

一轮通信：

```text
输入：全局模型 theta_t，被采样客户端集合 S_t
服务器向 S_t 中所有客户端广播 theta_t
每个客户端 i：
  从 theta_t 初始化本地模型
  本地训练 E 个 epoch
  上传 Delta_i = theta_i - theta_t
服务器构造 G_t，其行向量为 g_i = -Delta_i
服务器计算 d_t = UPGrad(G_t)
服务器更新 theta_{t+1} = theta_t - scale * d_t
```

### 4.2 UPGrad 聚合

客户端更新矩阵为：

```text
G_t in R^{m x d}
```

其中 `m` 是采样客户端数，`d` 是模型参数维度。UPGrad 使用 Gramian：

```text
H_t = G_t G_t^T
```

根据客户端更新之间的几何关系计算权重 `w_t`，并得到方向：

```text
d_t = G_t^T w_t
```

写作时不要过度声称“必然 Pareto 最优”。更稳妥的表述是：

```text
FedClient-UPGrad 在客户端更新代理上计算冲突感知的公共方向，其动机来自多目标优化中的公共下降方向和 Pareto stationarity。
```

### 4.3 与基线方法的区别

1. **FedAvg：** 对客户端更新按样本数加权平均。
2. **qFedAvg：** 根据客户端训练前损失调整权重。
3. **FedMGDA+：** 用 MGDA/min-norm 思路聚合客户端更新代理。
4. **FedClient-UPGrad：** 用 UPGrad 在客户端更新空间中计算公共方向。

### 4.4 通信和计算代价

通信方面：

```text
FedClient-UPGrad 和 FedAvg 类似，每个客户端上传一个模型 delta。
```

额外代价在服务器端：

```text
需要构造 m x d 更新矩阵，并求解 UPGrad 聚合方向。
```

FEMNIST 当前结果中：

1. FedAvg 平均每轮 `1.012s`。
2. qFedAvg 平均每轮 `1.067s`。
3. FedMGDA+ 平均每轮 `1.103s`。
4. FedClient-UPGrad 平均每轮 `1.480s`。

论文中要正面承认这个代价，并说明它换来了明显的 tail-client performance 提升。

## 5. Experimental Setup

### 5.1 数据集

主实验数据集：

1. **FEMNIST：** LEAF/FEMNIST writer partition，每个 writer 是一个客户端。实验使用 100 个客户端。
2. **CIFAR10 alpha=0.1：** torchvision CIFAR10，使用 Dirichlet label-skew 划分，100 个客户端，异构性强。
3. **CIFAR10 alpha=0.5：** 同上，但异构性较弱。

不纳入本文主实验的数据集：

1. **CelebA：** 虽然代码支持，但它更自然对应多标签属性预测，不是当前“客户端作为目标”的主线。
2. 其他语言类数据集不纳入本文，以保持论文聚焦于视觉场景下清晰的客户端异构。

### 5.2 模型

1. FEMNIST 使用小型 CNN，输入为 28x28 灰度字符图像。
2. CIFAR10 使用小型 CNN，输入为 32x32 RGB 图像。

不要把模型说成 SOTA backbone。它们是为了公平比较聚合方法而选择的受控模型。

### 5.3 对比方法

主对比方法：

1. `fedavg`
2. `qfedavg`
3. `fedmgda_plus`
4. `fedclient_upgrad`

选择理由：

1. FedAvg 是标准平均优化基线。
2. qFedAvg 是公平联邦学习代表方法。
3. FedMGDA+ 是最接近的多目标客户端更新基线。
4. FedClient-UPGrad 是本文方法。

### 5.4 超参数

FEMNIST：

```text
num_clients = 100
num_rounds = 500
local_epochs = 10
participation_rate = 0.5
learning_rate = 0.01
client_test_fraction = 0.2
seeds = {7, 42, 123}
```

CIFAR10：

```text
num_clients = 100
num_rounds = 1000
local_epochs = 2
participation_rate = 0.5
client_test_fraction = 0.2
seeds = {7, 42, 123}
```

CIFAR10 正式实验使用的超参数：

| 方法 | 学习率 | 方法参数 | 选择依据 |
|---|---:|---|---|
| FedAvg | 0.03 | 无 | R100 tuning 中表现最佳 |
| qFedAvg | 0.03 | `q=0.1`, `update_scale=1.0` | R100 tuning 中表现最佳 |
| FedMGDA+ | 0.03 | `update_scale=2.0` | R100 tuning 中表现最佳 |
| FedClient-UPGrad | 0.01 | `update_scale=1.0` | R1000 seed7 稳定性验证通过；`lr=0.03` 在长轮次训练中出现数值发散 |

论文中应说明：CIFAR10 的 baseline 参数来自固定 100 轮 tuning budget；FedClient-UPGrad 在短轮次 tuning 中 `lr=0.03` 早期收敛更快，但在 1000 轮完整训练中出现 NaN 发散，因此正式实验采用经过长轮次稳定性验证的 `lr=0.01, update_scale=1.0`。这属于稳定性筛选，而不是事后更改指标。

### 5.5 指标

主指标：

1. 平均客户端测试准确率。
2. worst-10% 客户端准确率。
3. 客户端准确率标准差。
4. 平均客户端测试损失。

效率指标：

1. 平均每轮时间。
2. 平均上传字节数。
3. 平均聚合/方向求解时间。
4. 总训练时间。

建议图：

1. Mean Acc vs Worst10 Acc 散点图。
2. Sorted Client Accuracy 曲线。
3. Client Accuracy CDF 或箱线图。
4. Efficiency-Performance trade-off 图。

### 5.6 不改变指标的运行优化

全量实验使用了三项不改变最终论文指标的实现优化：

1. `eval_interval=0`：关闭训练期间中间评估，训练结束后仍完整评估所有客户端测试集。
2. 只对 FedAvg 和 FedClient-UPGrad 跳过 `initial_loss`，因为它们的更新规则不使用该值；qFedAvg 和 FedMGDA+ 保留。
3. 每轮复用一个 local model 容器，但每个客户端训练前都用本轮全局模型快照严格重置，保持所有客户端从同一个 `theta_t` 出发。

## 6. Results

### 6.1 FEMNIST 主结果

结果目录：

```text
results/full_femnist_E10R500_pr50_lr001
```

三种子结果：

| 方法 | Mean Acc | Worst10 Acc | Acc Std | Mean Loss |
|---|---:|---:|---:|---:|
| FedAvg | 0.7665 ± 0.0132 | 0.4903 ± 0.0252 | 0.1301 ± 0.0114 | 0.7814 ± 0.0824 |
| qFedAvg | 0.6968 ± 0.0150 | 0.4163 ± 0.0288 | 0.1373 ± 0.0091 | 1.0680 ± 0.0828 |
| FedMGDA+ | 0.7335 ± 0.0146 | 0.4358 ± 0.0367 | 0.1470 ± 0.0072 | 0.9686 ± 0.0703 |
| FedClient-UPGrad | **0.7996 ± 0.0094** | **0.5902 ± 0.0058** | **0.1030 ± 0.0041** | **0.7026 ± 0.0679** |

解释：

1. FedClient-UPGrad 在三个 seed 上四个指标全部排名第一。
2. 相比 FedAvg，平均准确率提升 `+0.0332`，相对提升 `+4.33%`。
3. 相比 FedAvg，worst-10% 准确率提升 `+0.0999`，相对提升 `+20.37%`。
4. 客户端准确率标准差降低 `0.0271`，说明客户端间表现更均衡。
5. 平均测试损失相对 FedAvg 降低 `10.09%`。

可直接写进论文：

```text
在 FEMNIST 上，FedClient-UPGrad 在三种随机种子下均稳定优于所有基线方法。其优势尤其体现在 worst-10% 客户端准确率上，相比 FedAvg 提升 9.99 个百分点。这说明 FedClient-UPGrad 不仅提升了平均客户端性能，也显著改善了尾部客户端表现，与客户端级多目标优化的动机一致。
```

### 6.2 FEMNIST 客户端分布分析

三种子平均客户端准确率分位数：

| 方法 | P10 | P25 | Median | P75 |
|---|---:|---:|---:|---:|
| FedClient-UPGrad | **0.6563** | **0.7398** | **0.8176** | **0.8689** |
| FedAvg | 0.6251 | 0.7019 | 0.7852 | 0.8645 |
| FedMGDA+ | 0.5467 | 0.6517 | 0.7530 | 0.8419 |
| qFedAvg | 0.5130 | 0.6215 | 0.7053 | 0.7930 |

解释：FedClient-UPGrad 不是只提高少数客户端或平均值，而是整体抬高了低分位和中位数客户端表现。

### 6.3 CIFAR10 alpha=0.1 结果

结果目录：

```text
results/full_cifar_alpha0p1_E2R1000_pr50_mixed
```

该设置使用 mixed hyperparameters：FedAvg、qFedAvg 和 FedMGDA+ 使用 R100 tuning 选出的最佳参数，FedClient-UPGrad 使用经过 R1000 seed7 稳定性验证的 `learning_rate=0.01, update_scale=1.0`。

三种子结果：

| 方法 | Mean Acc | Worst10 Acc | Acc Std | Mean Loss |
|---|---:|---:|---:|---:|
| FedAvg | **0.6040 ± 0.0047** | 0.3424 ± 0.0394 | 0.1362 ± 0.0105 | **1.1175 ± 0.0084** |
| qFedAvg | 0.5290 ± 0.0252 | 0.2754 ± 0.0758 | 0.1333 ± 0.0093 | 1.3015 ± 0.0343 |
| FedMGDA+ | 0.4505 ± 0.2848 | 0.2588 ± 0.2289 | 0.1765 ± 0.0651 | 1.2631 ± 0.3054 |
| FedClient-UPGrad | 0.5928 ± 0.0133 | **0.3777 ± 0.0286** | **0.1190 ± 0.0096** | 1.1551 ± 0.0372 |

解释：

1. FedClient-UPGrad 在 CIFAR10 alpha=0.1 上没有全面超过 FedAvg；FedAvg 的平均准确率和平均损失最好。
2. FedClient-UPGrad 获得最高 worst-10% 客户端准确率，相比 FedAvg 提升 `+0.0352`，相对提升 `+10.28%`。
3. FedClient-UPGrad 获得最低客户端准确率标准差，相比 FedAvg 降低 `0.0172`，说明客户端间表现更均衡。
4. FedClient-UPGrad 相比 qFedAvg 和 FedMGDA+ 在 mean、worst10、std 上均更稳定。
5. FedMGDA+ 在 seed123 出现数值发散，导致 `mean_client_test_loss=NaN`，这说明强 label-skew 下多目标聚合方法对长轮次稳定性较敏感。

逐 seed 现象：

| Seed | Mean Acc 最好 | Worst10 最好 | Acc Std 最低 | Mean Loss 最低 |
|---:|---|---|---|---|
| 7 | FedClient-UPGrad | FedClient-UPGrad | qFedAvg | FedAvg |
| 42 | FedMGDA+ | FedMGDA+ | FedClient-UPGrad | FedMGDA+ |
| 123 | FedAvg | FedClient-UPGrad | FedClient-UPGrad | FedAvg |

可直接写进论文：

```text
在 CIFAR10 alpha=0.1 的强 label-skew 设置下，FedClient-UPGrad 没有在平均准确率上超过 FedAvg，但取得了最高的 worst-10% 客户端准确率和最低的客户端准确率标准差。这说明在更尖锐的标签异构场景中，FedClient-UPGrad 主要体现为平均性能与客户端公平性的 trade-off：它牺牲少量平均准确率，换取更好的尾部客户端表现和更小的客户端间差异。
```

### 6.4 CIFAR10 alpha=0.5 结果

状态：完整三种子实验仍在运行。

完成后需要补充：

1. CIFAR10 alpha=0.5 三种子结果表。
2. 相对 tuned baselines 的提升。
3. sorted client accuracy 曲线。
4. 效率-性能 trade-off。

### 6.5 效率 trade-off

FEMNIST 效率：

| 方法 | 平均每轮时间 | 平均聚合计算时间 | 总训练时间 |
|---|---:|---:|---:|
| FedAvg | 1.012s ± 0.011s | 0.0003s | 506.23s ± 5.58s |
| qFedAvg | 1.067s ± 0.032s | 0.0001s | 533.57s ± 16.16s |
| FedMGDA+ | 1.103s ± 0.037s | 0.0353s | 551.90s ± 18.50s |
| FedClient-UPGrad | 1.480s ± 0.055s | 0.4722s | 740.67s ± 27.40s |

解释：FedClient-UPGrad 服务器端聚合更慢，平均每轮时间比 FedAvg 高约 `46%`，但在 FEMNIST 上换来了 `+4.33%` 的平均准确率提升和 `+20.37%` 的 worst-10% 准确率提升。论文中要把它表述为“性能-公平性-计算代价”的 trade-off，而不是回避这个问题。

CIFAR10 alpha=0.1 效率：

| 方法 | 平均每轮时间 | 平均聚合计算时间 | 总训练时间 |
|---|---:|---:|---:|
| FedAvg | 2.788s ± 0.060s | 0.0003s | 2788.30s ± 60.46s |
| qFedAvg | 4.078s ± 0.057s | 0.0001s | 4078.68s ± 56.75s |
| FedMGDA+ | 4.109s ± 0.040s | 0.0318s | 4109.37s ± 39.71s |
| FedClient-UPGrad | 3.229s ± 0.020s | 0.4710s | 3230.03s ± 20.20s |

解释：CIFAR10 alpha=0.1 上 FedClient-UPGrad 比 FedAvg 慢，但比 qFedAvg 和 FedMGDA+ 的实际总训练时间更低。其主要额外成本仍来自 UPGrad 聚合，但稳定 `lr=0.01` 配置避免了 `lr=0.03` 长轮次发散问题。

## 7. Theory and Analysis

理论部分必须和实验形成一条论证链：先证明方法为什么合理，再说明实验为什么看这些指标。

### 7.1 理论要证明什么

理论部分应支持三个递进命题：

1. **Lemma 1：Local Delta Proxy**
客户端本地更新量可以作为该客户端目标梯度或下降方向的近似代理。

2. **Lemma 2：UPGrad Common Direction**
UPGrad 可以把这些客户端方向代理组合成一个冲突感知的公共方向。

3. **Theorem 3：Approximate Pareto-Stationarity**
结合前两个命题，在标准非凸假设下，FedClient-UPGrad 可以收敛到 approximate Pareto-stationary point 附近，误差由本地训练漂移、随机梯度噪声和客户端采样误差决定。

注意：不要声称非凸全局最优收敛。合理目标是“一阶稳定性”或“近似 Pareto 稳定性”。

实事求是地说，当前理论部分还只是“证明框架”，不是完整证明。正式论文至少还需要补齐以下内容：

1. **统一符号系统：** 客户端目标、采样集合、本地 SGD 轨迹、客户端 delta、代理梯度矩阵、UPGrad 权重、服务器更新方向都要明确定义。
2. **明确理论假设：** smoothness、随机梯度无偏性、方差有界、梯度二阶矩有界、客户端采样假设、本地步长和服务器步长限制。
3. **local delta 误差推导：** 不能只说 delta 是 proxy，要给出 `-Delta_i/(eta_l E)` 与 `grad F_i(theta_t)` 的偏差界。
4. **UPGrad 方向性质：** 需要说明 UPGrad 至少满足什么性质，例如有界性、Gramian 几何依赖、正对齐条件下的一步下降，而不是泛泛说“冲突感知”。
5. **目标函数下降或 residual 下降：** 要用 smoothness descent lemma 把更新方向和目标下降联系起来。
6. **采样误差：** 每轮只采样部分客户端，需要说明 sampled residual 和 full-client residual 的关系，或明确理论只证明 sampled-objective stationarity。
7. **最终 theorem：** 给出一个可发表的非凸结论，例如平均 Pareto-stationarity residual 收敛到由 proxy error 和 sampling error 决定的邻域。
8. **理论与实验指标映射：** 说明为什么 theorem 中的“多目标平衡”在实验中对应 worst-10%、client accuracy std 和 sorted client accuracy curve。

如果时间有限，理论优先级应为：Lemma 1 完整证明 > Lemma 2 性质说明 > Theorem 3 给出较保守但自洽的证明草图。不要为了显得理论强而写无法支撑的全局收敛。

### 7.1.1 建议使用的统一符号

正式论文理论部分建议先定义以下符号：

```text
K                         客户端总数
D_i                       客户端 i 的本地数据集或数据分布
F_i(theta)                客户端 i 的经验风险 / 期望风险
F(theta)                  客户端级向量目标 [F_1(theta), ..., F_K(theta)]
S_t                       第 t 轮采样到的客户端集合
m = |S_t|                 每轮参与客户端数量
theta_t                   第 t 轮服务器全局模型
theta_{i,e}^t             客户端 i 第 e 个本地 SGD step 后的模型
eta_l                     客户端本地学习率
E                         本地 step / epoch 数的理论抽象
Delta_i^t                 客户端 i 上传的模型增量 theta_{i,E}^t - theta_t
g_i^t                     客户端 i 的代理梯度，建议定义为 -Delta_i^t / (eta_l E)
G_t                       sampled client proxy matrix，行向量为 g_i^t
H_t = G_t G_t^T           客户端代理梯度 Gramian
w_t                       UPGrad 计算出的组合权重
d_t = G_t^T w_t           服务器聚合方向
alpha_t                   服务器端更新步长或 update scale
R(theta)                  Pareto-stationarity residual
```

注意：代码里 FedClient-UPGrad 实际使用 `g_i = -Delta_i`。理论里写 `g_i = -Delta_i/(eta_l E)` 更接近“平均梯度代理”，两者只差一个正比例尺度。在不改变方向的分析中可以吸收到服务器步长或 update scale 里。论文中必须说明这一点，避免理论符号和代码实现看起来不一致。

### 7.1.2 建议理论假设

可以采用标准非凸 FL 假设：

1. **Smoothness：** 对每个客户端目标 `F_i`，存在 `L > 0`，使得：

```text
||grad F_i(x) - grad F_i(y)|| <= L ||x - y||
```

2. **随机梯度无偏：** 客户端 minibatch 随机梯度满足：

```text
E[g_i(theta; xi)] = grad F_i(theta)
```

3. **方差有界：**

```text
E||g_i(theta; xi) - grad F_i(theta)||^2 <= sigma_i^2
```

4. **二阶矩有界：**

```text
E||g_i(theta; xi)||^2 <= G^2
```

5. **客户端采样有界方差：** 每轮采样集合 `S_t` 是随机采样，sampled client gradients 对全体客户端目标的估计误差有界。

6. **步长条件：** `eta_l`、`E`、`alpha_t` 足够小，使 local drift 和 server update 不破坏一阶下降分析。

7. **UPGrad 方向有界：** 存在常数 `D`，使：

```text
E||d_t||^2 <= D^2
```

8. **正对齐 / 充分下降条件：** 为证明目标下降，需要假设或证明聚合方向与某个多目标 stationarity direction 有足够对齐，例如：

```text
<bar_g_t, d_t> >= c R_S(theta_t) - epsilon_up
```

其中 `bar_g_t` 可以是某个理想多目标公共方向，`R_S(theta_t)` 是 sampled Pareto residual，`epsilon_up` 表示 UPGrad 近似求解误差或 proxy 误差。

最后这个假设最关键，也最难。若无法从 UPGrad 原文性质直接推出，就应诚实写成 theorem assumption 或 proof condition，而不是硬说 UPGrad 必然满足。

### 7.2 Lemma 1：Local Delta Proxy

目的：证明服务器为什么可以把客户端上传的 delta 当作客户端目标方向信息。

客户端 `i` 从 `theta_t` 出发，本地训练 `E` 步后上传：

```text
Delta_i^t = theta_{i,E}^t - theta_t
```

单步 SGD 时：

```text
-Delta_i^t / eta_l = stochastic gradient of F_i(theta_t)
```

多步本地训练时：

```text
-Delta_i^t / (eta_l E) = grad F_i(theta_t) + proxy_error_i^t
```

其中 `proxy_error` 由 smoothness、local learning rate、local steps、随机梯度方差控制。

可以给出的误差形式：

```text
E ||proxy_error_i^t||^2 <= C_1 eta_l^2 E^2 G^2 + C_2 sigma_i^2 / E
```

这个 lemma 直接支撑方法设计：FedClient-UPGrad 不需要显式计算每个客户端完整梯度或 Jacobian，因为客户端本地更新已经携带了方向信息。

证明路线应写得更具体：

1. 写出客户端本地 SGD 轨迹：

```text
theta_{i,e+1}^t = theta_{i,e}^t - eta_l * stochastic_grad_i(theta_{i,e}^t; xi_{i,e})
theta_{i,0}^t = theta_t
```

2. 对 `E` 步求和：

```text
Delta_i^t = theta_{i,E}^t - theta_t
          = -eta_l * sum_{e=0}^{E-1} stochastic_grad_i(theta_{i,e}^t; xi_{i,e})
```

3. 因此平均代理梯度为：

```text
g_i^t = -Delta_i^t / (eta_l E)
      = 1/E * sum_{e=0}^{E-1} stochastic_grad_i(theta_{i,e}^t; xi_{i,e})
```

4. 加减 `grad F_i(theta_t)`：

```text
g_i^t - grad F_i(theta_t)
= 1/E * sum_e [stochastic_grad_i(theta_{i,e}^t; xi_{i,e}) - grad F_i(theta_{i,e}^t)]
  + 1/E * sum_e [grad F_i(theta_{i,e}^t) - grad F_i(theta_t)]
```

第一项是随机梯度噪声，第二项是 local drift。

5. 用方差有界控制随机噪声项，用 smoothness 控制 local drift：

```text
||grad F_i(theta_{i,e}^t) - grad F_i(theta_t)|| <= L ||theta_{i,e}^t - theta_t||
```

6. 再用本地梯度二阶矩有界得到：

```text
E||theta_{i,e}^t - theta_t||^2 <= O(eta_l^2 e^2 G^2)
```

7. 最终得到：

```text
E||g_i^t - grad F_i(theta_t)||^2
<= O(sigma_i^2 / E) + O(L^2 eta_l^2 E^2 G^2)
```

这条推导非常重要，因为它把“本地多步训练造成的 drift”明确放进理论误差项里。CIFAR10 alpha=0.1 上结果不如 FEMNIST 干净，也可以从这个角度解释：强 label-skew 使 local drift 和 proxy error 更大。

### 7.3 Lemma 2：UPGrad Common Direction

目的：证明服务器端 UPGrad 聚合不是任意操作，而是基于客户端更新几何关系的公共方向计算。

定义：

```text
g_i^t = -Delta_i^t
G_t = [g_i^t]_{i in S_t}
H_t = G_t G_t^T
d_t = G_t^T w_t
```

其中 `H_t` 描述客户端更新之间的 pairwise 几何关系。如果两个客户端方向冲突：

```text
<g_i, g_j> < 0
```

简单平均可能抵消方向，或者被多数客户端方向主导。UPGrad 则利用 Gramian 计算更合理的组合权重。

可以证明三种形式之一：

1. **Alignment form：** UPGrad 方向与客户端方向有更好的对齐关系。
2. **Projection form：** UPGrad 等价于求解 Gramian 空间中的约束二次问题。
3. **Descent form：** 如果方向与活跃客户端梯度代理有足够正对齐，则小步长更新会降低这些客户端目标的一阶近似。

建议论文中采用保守表述：

```text
给定客户端代理梯度矩阵 G_t，UPGrad 计算 Gramian-aware 的组合方向 d_t = G_t^T w_t。当客户端更新存在冲突时，该方向区别于简单平均，它显式利用了客户端更新之间的几何关系。在正对齐条件下，d_t 是 sampled proxy objectives 的公共下降方向。
```

这里需要特别谨慎。论文不能只说“UPGrad 更好”，需要说明它具体提供什么理论性质。建议写成三层：

1. **几何层性质：** UPGrad 只依赖 `G_t G_t^T` 的 Gramian 几何，因此显式利用客户端方向之间的内积、冲突和相似性。

2. **尺度层性质：** 主方法使用 raw update proxy，不做单位范数归一化，因此保留 Jacobian Descent / UPGrad 中梯度范数的线性和加权影响。这一点与 STCH 式目标函数归一化不同。

3. **下降层性质：** 在正对齐条件下，`d_t` 对 sampled objectives 是公共下降方向。具体可写成：

```text
<grad F_i(theta_t), d_t> >= gamma_i ||d_t||^2 - error_i
```

或者更弱地写成聚合 residual 形式：

```text
<d_t, G_t^T lambda_t^*> >= c R_S_proxy(theta_t) - epsilon_up
```

其中 `lambda_t^*` 是 sampled Pareto residual 的最优 simplex 权重，`R_S_proxy` 是基于代理梯度的 residual。

如果无法严格从 UPGrad 原文直接推出每个客户端都下降，则不要写“对所有客户端下降”。更稳妥的理论结论是：

```text
UPGrad 给出一个基于 Gramian 的冲突感知方向，并在满足正对齐或充分下降条件时，可以推动 sampled multi-objective residual 下降。
```

这与实验也一致：CIFAR10 alpha=0.1 中 FedClient-UPGrad 没有让所有指标全面第一，但改善了 worst10 和 std，说明它更偏向目标平衡而不是单纯平均最优。

### 7.4 Theorem 3：Approximate Pareto-Stationarity

非凸情况下不证明全局最优，而证明近似一阶稳定性。多目标视角下，合适的 stationarity residual 是：

```text
R(theta) = min_{lambda in simplex} || sum_i lambda_i grad F_i(theta) ||^2
```

或者采样版本：

```text
R_S(theta_t) = min_{lambda in simplex} || sum_{i in S_t} lambda_i grad F_i(theta_t) ||^2
```

目标 theorem 形式：

```text
1/T sum_{t=0}^{T-1} E[R(theta_t)]
  <= optimization_error(T)
   + local_update_proxy_error
   + stochastic_gradient_error
   + client_sampling_error
```

预期收敛形式：

```text
O(1/sqrt(T)) + O(proxy error) + O(sampling error)
```

逻辑依赖：

1. Lemma 1 说明 `G_t` 是客户端真实梯度的近似。
2. Lemma 2 说明 UPGrad 从 `G_t` 中计算结构化公共方向。
3. Theorem 3 说明重复使用该方向更新，可以趋近 approximate Pareto-stationarity，误差来自 proxy 和 sampling。

推荐假设：

1. 每个客户端目标 `F_i` 是 L-smooth。
2. 随机梯度无偏且方差有界。
3. 梯度或本地更新二阶矩有界。
4. 客户端采样无偏或采样方差有界。
5. local learning rate 和 server step size 足够小。
6. UPGrad 方向范数有界，并满足与 proxy objectives 的正对齐或充分下降条件。

更完整的 theorem 应包含下面四个误差项：

1. **优化误差：** 来自有限轮数 `T`，通常是 `O(1/sqrt(T))` 或类似形式。
2. **本地更新代理误差：** 来自 Lemma 1 中 `g_i^t` 与 `grad F_i(theta_t)` 的差距，通常包含 `O(sigma^2/E)` 和 `O(L^2 eta_l^2 E^2 G^2)`。
3. **客户端采样误差：** 来自每轮只采样 `S_t`，而不是全体客户端。
4. **UPGrad 求解/对齐误差：** 来自 UPGrad 方向相对于理想 Pareto common direction 的偏差。

建议 theorem 表述为：

```text
Theorem. 在 Assumption 1-6 成立时，选择合适的 local learning rate eta_l 和 server step size alpha_t，则 FedClient-UPGrad 生成的模型序列满足：

1/T sum_{t=0}^{T-1} E[R(theta_t)]
<= O(1/sqrt(T))
 + O(sigma^2/E)
 + O(L^2 eta_l^2 E^2 G^2)
 + O(sampling_error)
 + O(upgrad_alignment_error)
```

其中 `R(theta)` 是 full-client 或 sampled-client Pareto-stationarity residual。

如果只证明 sampled-client 版本，结论更容易成立：

```text
1/T sum_t E[R_{S_t}(theta_t)] <= ...
```

如果要证明 full-client 版本，则必须额外处理 sampled residual 和 full residual 的偏差：

```text
|R_{S_t}(theta_t) - R(theta_t)| <= sampling_error
```

这一步可能较难，建议论文主定理先用 sampled stationarity，附加 corollary 再说明在无偏采样和足够覆盖客户端时可推广到 full-client residual 的期望界。

证明路线：

1. 用 Lemma 1 将真实客户端梯度替换成代理梯度，并引入 proxy error。
2. 用 UPGrad common direction condition 建立 `d_t` 和 sampled Pareto residual 的下降关系。
3. 用 L-smooth descent lemma：

```text
F_i(theta_{t+1}) <= F_i(theta_t)
  - alpha_t <grad F_i(theta_t), d_t>
  + L alpha_t^2 ||d_t||^2 / 2
```

4. 对 sampled objectives 或 scalarized residual 加权求和。
5. 对随机性、客户端采样和本地 SGD 噪声取期望。
6. telescope sum over `t=0,...,T-1`。
7. 选择步长 `alpha_t = O(1/sqrt(T))` 得到平均 residual 收敛界。

可接受的保守结论：

```text
FedClient-UPGrad converges to a neighborhood of sampled Pareto-stationarity. The neighborhood size is controlled by local-update approximation, stochastic gradient noise, partial participation, and UPGrad alignment error.
```

不可接受的过强结论：

```text
FedClient-UPGrad converges to the global optimum.
FedClient-UPGrad always improves every client.
FedClient-UPGrad always dominates FedAvg.
```

### 7.5 理论和实验的有机联系

理论和实验不是两部分，而是同一条证据链。

| 理论命题 | 含义 | 实验对应 | 指标/图 |
|---|---|---|---|
| Local Delta Proxy | 客户端 delta 含有客户端目标方向信息 | FedClient-UPGrad 只上传 FedAvg-style delta | 通信量与 FedAvg-style 方法一致 |
| UPGrad Common Direction | 聚合应处理客户端方向冲突 | 与 FedAvg、FedMGDA+ 比较 | mean accuracy、worst10 accuracy、sorted client curve |
| Approximate Pareto-Stationarity | 方法目标是多客户端目标平衡 | 看尾部客户端和客户端分布是否改善 | worst10 accuracy、accuracy std、CDF/boxplot |
| proxy/sampling error | 随机性和采样会带来误差 | 三种子实验 | mean ± std、error bar |
| common direction cost | 结构化聚合有额外计算 | 报告每轮时间和聚合时间 | efficiency-performance plot |

这就是理论和实验的有机关联：

1. 理论说客户端 delta 是目标方向代理，所以实验强调方法不需要额外 Jacobian 通信。
2. 理论说 UPGrad 处理客户端冲突，所以实验必须看 worst10 和客户端分布，而不能只看 mean accuracy。
3. 理论说收敛目标是 approximate Pareto-stationarity，所以实验重点是客户端间表现是否更均衡。
4. 理论中有 sampling/proxy error，所以实验必须多 seed。
5. 理论中 UPGrad 有额外计算，所以实验必须报告效率 trade-off。

可直接写进论文的桥接句：

```text
理论分析表明，基于客户端更新几何关系计算的公共方向应比简单平均更能平衡客户端目标。因此，我们不仅报告平均客户端准确率，还报告 worst-10% 客户端准确率和客户端准确率标准差。在 FEMNIST 上，FedClient-UPGrad 相比 FedAvg 将 worst-10% 准确率提升 9.99 个百分点，同时提高平均准确率，说明更新空间中的公共方向确实改善了尾部客户端而没有牺牲整体性能。
```

## 8. Discussion

### 8.1 为什么 FedClient-UPGrad 在 FEMNIST 上有效

可能原因：

1. FEMNIST writer partition 是真实客户端异构，每个 writer 的书写风格不同。
2. 不同 writer 的客户端更新方向存在冲突。
3. FedClient-UPGrad 减少了平均方向对多数客户端的偏向，更好考虑弱客户端目标。

### 8.2 为什么 qFedAvg 和 FedMGDA+ 可能较弱

qFedAvg：

1. 它基于损失标量重加权，无法完整表示方向冲突。
2. loss 高不一定代表该客户端方向应该被如何几何组合。

FedMGDA+：

1. min-norm 方向可能过于保守。
2. 它可能降低冲突，但也可能削弱有效进展方向。
3. 当前 FEMNIST 结果说明 FedClient-UPGrad 在“公共下降”和“有效前进”之间取得了更好平衡。

### 8.3 CIFAR10 的作用

CIFAR10 用来验证方法是否能从 writer heterogeneity 泛化到 label-skew heterogeneity：

1. alpha=0.1 测强异构。
2. alpha=0.5 测较弱异构。
3. 三种子结果决定最终能否强声称跨数据集有效。

## 9. Limitations

### 9.1 运行时间和服务器端开销

FedClient-UPGrad 比 FedAvg 慢，因为服务器端要求解 UPGrad 方向。FEMNIST 上平均每轮 `1.480s`，FedAvg 为 `1.012s`。

推荐表述：

```text
FedClient-UPGrad 用额外服务器端计算换取更好的尾部客户端性能。对于关注公平性和弱客户端可靠性的场景，这一 trade-off 是有意义的；但在服务器计算资源极其受限或每轮采样客户端数很大的场景中，该代价需要进一步优化。
```

### 9.2 客户端更新只是近似梯度

多步本地训练下，delta 不是精确梯度，而是 trajectory-integrated proxy。理论和实验都应承认这一点。

### 9.3 数据集范围

本文主要证据来自视觉任务。未来需要扩展到更大模型、更大规模客户端和更多模态。

### 9.4 代码实现成熟度

当前是研究型 PyTorch 实现，基线为统一框架下复现。论文中可以说明为了公平比较，所有方法使用同一数据划分、模型、训练轮数和评估协议。

## 10. Conclusion

结论应强调：

1. 异构联邦学习不能只看平均目标。
2. 客户端级多目标视角能更好解释和处理 tail-client 问题。
3. FedClient-UPGrad 用标准客户端更新实现了实用的多目标聚合。
4. FEMNIST 结果显示该方法稳定提升平均性能、尾部性能并降低客户端差异。
5. CIFAR10 结果将进一步决定跨 label-skew 设置的结论强度。
6. 后续工作应降低服务器端聚合开销并扩展到更大规模场景。

## 推荐图表

主文：

1. 三数据集三种子 `mean ± std` 主结果表。
2. Mean Acc vs Worst10 Acc 散点图。
3. Sorted Client Accuracy 曲线。
4. Worst10 Accuracy 柱状图，带 error bar。
5. Efficiency-Performance trade-off 图。

附录：

1. CIFAR10 R100 tuning 表。
2. learning rate / update scale 敏感性分析。
3. 每个 seed 的完整结果。
4. 运行优化说明。

## 当前证据状态

已完成且支持本文方法：

1. FEMNIST 三种子完整实验强支持 FedClient-UPGrad。
2. CIFAR10 alpha=0.1 三种子 mixed 参数完整实验支持 FedClient-UPGrad 作为尾部客户端/公平性改进方法：worst-10% 准确率最高、客户端准确率标准差最低，但平均准确率不是最高。
3. CIFAR10 R100 调参为 baseline 提供了强参数，同时说明短轮次调参需要配合长轮次稳定性检查，尤其是 FedClient-UPGrad 和 FedMGDA+。
4. 旧 CIFAR10 seed7 R1000 的 `lr=0.01` 结果已经显示 FedClient-UPGrad 稳定且有优势，因此用于确定正式 full-run 的稳定配置。

待完成：

1. CIFAR10 alpha=0.5 三种子 mixed/stable 参数完整结果。
2. 跨数据集总表。
3. 最终论文图。

## 写作策略

建议重点：

1. 以客户端级多目标问题和尾部客户端性能作为主线。
2. 把 FedClient-UPGrad 写成实用 update-space 方法，而不是依赖显式 Jacobian 的重型理论方法。
3. FEMNIST 是当前最强证据，因为 writer partition 天然对应客户端异构。
4. CIFAR10 用来证明方法能推广到 label-skew 异构。
5. 正面承认运行时间代价，并把它解释成计算-公平性 trade-off。

避免过度声称：

1. 不声称非凸全局最优收敛。
2. 不声称通信量低于 FedAvg。
3. 不声称所有数据集上全面支配；当前 CIFAR10 alpha=0.1 结果更支持“公平性-平均性能 trade-off”。
4. 不引入与主线无关的数据集结果。
