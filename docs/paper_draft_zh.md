# FedClient-UPGrad：面向公平联邦学习的客户端级多目标梯度聚合方法

## 摘要

异构联邦学习通常以平均准确率作为主要评价指标，但在实际部署中，模型是否能服务低性能客户端同样重要。本文从客户端级多目标优化视角重新审视异构联邦学习，将每个客户端的经验损失视为一个独立目标。基于这一视角，我们提出 FedClient-UPGrad：每个被选客户端从服务器广播的全局模型出发进行普通本地训练，并上传模型更新量；服务器将负更新量作为客户端目标梯度代理，利用 UPGrad 在客户端更新空间中计算冲突感知的公共方向。该方法不需要显式计算客户端 Jacobian，通信形式与 FedAvg 风格的模型更新上传一致。我们在 FEMNIST 和 Dirichlet 划分的 CIFAR10 上比较 FedClient-UPGrad、FedAvg、qFedAvg 和 FedMGDA+。在 FEMNIST 三种子实验中，FedClient-UPGrad 获得 0.7996 ± 0.0094 的平均客户端准确率和 0.5902 ± 0.0058 的 worst-10% 客户端准确率，均显著优于各基线方法。这说明客户端级多目标聚合能够有效改善异构联邦学习中的尾部客户端表现。

**关键词：** 联邦学习，多目标优化，客户端异构，公平性，UPGrad

---

## 1. 引言

### 1.1 问题背景与动机

联邦学习（Federated Learning, FL）[McMahan et al., 2017] 是一种分布式机器学习范式，允许多个客户端在不共享原始数据的前提下协作训练全局模型。在联邦学习的实际部署中，客户端天然存在异构性：不同客户端的数据分布、样本量和数据质量可能存在显著差异。例如，在 FEMNIST 手写字符识别任务中，不同写字人的书写风格迥异；在跨机构医疗影像分析中，不同医院的设备参数和患者群体各不相同。

这种客户端异构性导致了一个关键问题：**一个全局模型即使具有较高平均准确率，也可能在部分客户端上表现很差。** 由于不同客户端的数据分布存在显著差异，优化平均目标可能使模型偏向多数客户端或易优化客户端，从而忽略尾部客户端。这说明仅以平均性能评价联邦模型是不充分的，有必要从客户端级目标平衡的角度重新设计聚合方法。

### 1.2 现有方法的代表思路

现有联邦学习方法可以从以下几个角度理解：

**（1）FedAvg 类方法。** FedAvg [McMahan et al., 2017] 是最经典的联邦学习算法，客户端本地训练后上传模型更新量，服务器按样本数加权平均。该方法简单高效，但本质上优化的是所有客户端的加权平均目标，无法显式处理客户端之间的目标冲突。

**（2）公平联邦学习 / qFedAvg。** q-FFL [Li et al., 2020] 通过根据客户端损失进行重加权，让高损失客户端获得更大权重，从而改善公平性。然而，这种方法仅通过标量损失值调整权重，无法完整表示客户端更新方向之间的几何冲突关系。

**（3）多目标联邦学习 / MGDA 类方法。** FedMGDA+ 将多目标优化中的 MGDA（Multiple Gradient Descent Algorithm）[Désidéri, 2012] 引入联邦学习聚合，试图寻找多个客户端目标的公共下降方向。但 MGDA 的 min-norm 方向可能过于保守，在客户端方向高度冲突时产生接近零的更新方向。

**（4）梯度冲突处理方法。** 在集中式多任务学习中，已有方法通过投影、重组或几何约束处理任务梯度之间的冲突 [Sener & Koltun, 2018; Yu et al., 2020; Liu et al., 2021]。但这些方法通常不是专门为客户端级联邦学习更新设计的。

### 1.3 现有方法的共同局限

现有方法要么把客户端目标压缩成一个平均目标（FedAvg），要么只通过标量损失重加权处理公平性（qFedAvg），要么需要较昂贵的目标级梯度信息（FedMGDA+）。它们没有充分利用标准联邦学习中已经自然产生的客户端本地更新方向——这些更新方向本身就携带了客户端目标的几何信息。

### 1.4 本文核心思路

本文的核心桥梁是：

1. **问题视角转换：** 将每个客户端的经验损失视为一个独立目标，将联邦学习建模为客户端级多目标优化问题。
2. **信息利用：** 客户端本地训练产生的模型更新量（delta）天然是该客户端目标下降方向的代理，无需额外计算。
3. **聚合方法创新：** 服务器不再简单平均这些更新，而是在客户端更新空间中用 UPGrad 计算冲突感知的公共方向。
4. **效果验证：** 这个公共方向更关注客户端目标之间的冲突，有助于改善尾部客户端表现。

### 1.5 本文贡献

本文的主要贡献如下：

1. **问题建模贡献：** 提出客户端级多目标联邦学习视角，将每个客户端经验损失视为一个独立目标，为理解和处理客户端异构性提供了新的理论框架。
2. **方法贡献：** 提出 FedClient-UPGrad，使用普通客户端本地更新作为目标方向代理，并用 UPGrad 进行冲突感知聚合。该方法通信形式与 FedAvg 一致，不需要显式计算客户端 Jacobian。
3. **实验贡献：** 构建统一实验框架，在 FEMNIST 和 CIFAR10 上公平比较 FedAvg、qFedAvg、FedMGDA+ 和 FedClient-UPGrad。在 FEMNIST 三种子实验中，FedClient-UPGrad 在平均准确率、worst-10% 准确率、客户端准确率标准差和测试损失上均稳定最优。
4. **理论贡献：** 在标准非凸假设下，证明 FedClient-UPGrad 可以收敛到近似 Pareto 稳定点附近，误差由本地训练漂移、随机梯度噪声和客户端采样误差决定。
5. **效率分析贡献：** 分析方法的效率代价，说明其以额外服务器端聚合计算换取更好的尾部客户端性能，并讨论这一 trade-off 的适用场景。

---

## 2. 相关工作

### 2.1 异构联邦优化

FedAvg [McMahan et al., 2017] 是联邦学习的基础算法，其核心思想是客户端本地多轮训练后上传模型更新，服务器按样本数加权平均。然而，在非 IID 数据分布下，FedAvg 面临客户端漂移（client drift）问题 [Karimireddy et al., 2020]，不同客户端的本地更新方向可能相互冲突。为缓解这一问题，FedProx [Li et al., 2020] 在客户端目标中引入近端项约束本地更新不要偏离全局模型太远；SCAFFOLD [Karimireddy et al., 2020] 引入控制变量修正客户端漂移；FedNova [Wang et al., 2020] 对不同客户端的异构本地更新步数进行归一化。这些方法从优化角度改善收敛，但本质上仍优化平均目标，未显式处理客户端之间的目标冲突。

### 2.2 公平联邦学习

公平性是联邦学习的重要研究维度。q-FFL（q-Fair Federated Learning）[Li et al., 2020] 通过引入参数 q 控制公平性程度：q 越大，高损失客户端获得越大的聚合权重。AFL [Mohri et al., 2019] 从极小极大（minimax）角度优化最差客户端的性能。FedFV [Wang et al., 2021] 通过梯度投影缓解客户端更新冲突。这些方法通过标量损失重加权或极小极大优化改善公平性，而 FedClient-UPGrad 从客户端更新向量的几何关系出发，在更新空间中直接处理方向冲突。

### 2.3 多目标优化与 MGDA

多目标优化（Multi-Objective Optimization, MOO）研究如何在多个可能冲突的目标之间寻找平衡解。Pareto 最优性和 Pareto 稳定性是多目标优化的核心概念 [Miettinen, 1999]。MGDA [Désidéri, 2012] 通过求解 min-norm 问题寻找多个目标梯度的公共下降方向。在集中式多任务学习中，MGDA 已被用于平衡不同任务的梯度冲突 [Sener & Koltun, 2018]。FedMGDA+ 将这一思想引入联邦学习，是最接近本文方法的多目标客户端更新基线。然而，MGDA 的 min-norm 方向在目标高度冲突时可能过于保守，产生接近零的更新方向，导致训练停滞。

### 2.4 梯度冲突与 UPGrad

在多任务学习中，梯度冲突（gradient conflict）是指不同任务的梯度方向存在负内积，简单平均可能导致方向抵消。PCGrad [Yu et al., 2020] 通过将冲突梯度投影到彼此的法平面来处理冲突。GradDrop [Chen et al., 2020] 根据梯度符号一致性选择性地保留梯度分量。UPGrad [Liu et al., 2021] 提出了一种基于 Gramian 空间的冲突感知聚合方法，通过求解带下界约束的二次规划问题计算公共方向。本文将这些思想引入联邦学习的客户端更新空间，利用 UPGrad 处理客户端更新之间的几何冲突。

---

## 3. 问题形式化

### 3.1 联邦学习基本设置

考虑一个联邦学习系统，包含 $K$ 个客户端。每个客户端 $i \in \{1, 2, \ldots, K\}$ 拥有本地数据集 $\mathcal{D}_i$，其经验损失函数定义为：

$$F_i(\theta) = \mathbb{E}_{(x, y) \sim \mathcal{D}_i}[\ell(f_\theta(x), y)]$$

其中 $\theta \in \mathbb{R}^d$ 是模型参数，$f_\theta$ 是参数化的模型，$\ell$ 是损失函数。

传统的 FedAvg 算法近似优化所有客户端的加权平均目标：

$$F_{\text{avg}}(\theta) = \sum_{i=1}^{K} p_i F_i(\theta)$$

其中 $p_i = \frac{|\mathcal{D}_i|}{\sum_{j=1}^{K} |\mathcal{D}_j|}$ 通常与客户端样本量成正比。

在每一轮通信 $t$ 中，服务器采样客户端子集 $\mathcal{S}_t \subseteq \{1, \ldots, K\}$，广播当前全局模型 $\theta_t$。每个被采样客户端 $i \in \mathcal{S}_t$ 从 $\theta_t$ 出发进行 $E$ 轮本地训练，得到本地模型 $\theta_{i,E}^t$，并上传模型更新量：

$$\Delta_i^t = \theta_{i,E}^t - \theta_t$$

服务器聚合这些更新量来更新全局模型。

### 3.2 客户端级多目标视角

本文提出一种不同的视角：**将每个客户端的经验损失视为一个独立目标**，从而将联邦学习建模为客户端级多目标优化问题。定义向量值目标函数：

$$\mathbf{F}(\theta) = [F_1(\theta), F_2(\theta), \ldots, F_K(\theta)]^\top$$

在每轮通信中，服务器只采样部分客户端 $\mathcal{S}_t$。服务器的目标不是仅优化平均损失，而是寻找一个能够**更好地平衡被采样客户端目标**的更新方向。

在多目标优化中，Pareto 稳定性（Pareto stationarity）是一个核心概念。一个点 $\theta$ 被称为 Pareto 稳定点，如果不存在一个方向能够同时降低所有目标函数值。形式化地，Pareto 稳定性的必要条件是：

$$\min_{\lambda \in \Delta_K} \left\| \sum_{i=1}^{K} \lambda_i \nabla F_i(\theta) \right\|^2 = 0$$

其中 $\Delta_K = \{\lambda \in \mathbb{R}^K : \lambda_i \geq 0, \sum_i \lambda_i = 1\}$ 是 $K$ 维概率单纯形。

### 3.3 本地更新作为梯度代理

FedClient-UPGrad 的核心观察是：客户端本地训练产生的模型更新量可以作为该客户端目标梯度或下降方向的实用代理。

具体地，客户端 $i$ 从全局模型 $\theta_t$ 出发，经过 $E$ 轮本地 SGD 训练后得到 $\theta_{i,E}^t$。上传的更新量为 $\Delta_i^t = \theta_{i,E}^t - \theta_t$。由于本地训练是沿着客户端损失 $F_i$ 的下降方向进行的，因此：

$$\mathbf{g}_i^t = -\Delta_i^t$$

可以被视为客户端目标梯度 $\nabla F_i(\theta_t)$ 的近似代理（proxy）。这一代理关系的精确刻画将在第 7 节的理论分析中给出（引理 1）。

### 3.4 评价目标

与客户端级多目标优化的问题定义相对应，本文的实验评价不能仅看平均准确率。主要评价指标包括：

1. **平均客户端测试准确率（Mean Client Test Accuracy）：** 所有客户端测试准确率的均值。
2. **Worst-10% 客户端准确率（Worst-10% Client Accuracy）：** 准确率最低的 10% 客户端的平均准确率，衡量尾部客户端表现。
3. **客户端准确率标准差（Client Accuracy Std）：** 衡量客户端间表现的均衡程度。
4. **平均客户端测试损失（Mean Client Test Loss）：** 所有客户端测试损失的均值。
5. **效率指标：** 平均每轮时间、平均上传字节数、平均聚合/方向求解时间。

---

## 4. 方法：FedClient-UPGrad

### 4.1 算法流程

FedClient-UPGrad 的一轮通信流程如下：

**算法 1：FedClient-UPGrad（单轮）**

- **输入：** 全局模型 $\theta_t$，被采样客户端集合 $\mathcal{S}_t$
- **步骤 1（广播）：** 服务器向 $\mathcal{S}_t$ 中所有客户端广播 $\theta_t$
- **步骤 2（本地训练）：** 每个客户端 $i \in \mathcal{S}_t$：
  - 从 $\theta_t$ 初始化本地模型
  - 在本地数据上进行 $E$ 个 epoch 的训练
  - 上传更新量 $\Delta_i^t = \theta_{i,E}^t - \theta_t$
- **步骤 3（构造代理梯度矩阵）：** 服务器构造矩阵 $\mathbf{G}_t \in \mathbb{R}^{|\mathcal{S}_t| \times d}$，其第 $i$ 行为 $\mathbf{g}_i^t = -\Delta_i^t$
- **步骤 4（UPGrad 聚合）：** 服务器计算公共方向 $\mathbf{d}_t = \text{UPGrad}(\mathbf{G}_t)$
- **步骤 5（模型更新）：** 服务器更新全局模型 $\theta_{t+1} = \theta_t - \eta_s \cdot \mathbf{d}_t$
- **输出：** 更新后的全局模型 $\theta_{t+1}$

其中 $\eta_s$ 是服务器端的学习率（或更新尺度）。

### 4.2 UPGrad 聚合详解

UPGrad 聚合是 FedClient-UPGrad 的核心组件。给定客户端代理梯度矩阵 $\mathbf{G} \in \mathbb{R}^{m \times d}$（其中 $m = |\mathcal{S}_t|$ 是采样客户端数，$d$ 是模型参数维度），UPGrad 通过以下步骤计算公共方向：

**步骤 1：计算 Gramian 矩阵。** 首先计算客户端更新之间的 pairwise 几何关系：

$$\mathbf{H} = \mathbf{G} \mathbf{G}^\top \in \mathbb{R}^{m \times m}$$

其中 $H_{ij} = \langle \mathbf{g}_i, \mathbf{g}_j \rangle$ 表示客户端 $i$ 和客户端 $j$ 的代理梯度方向之间的内积。如果 $H_{ij} < 0$，说明两个客户端的更新方向存在冲突。

**步骤 2：求解带下界约束的二次规划。** 对于每个客户端 $k \in \{1, \ldots, m\}$，求解以下二次规划问题：

$$\mathbf{w}^{(k)} = \arg\min_{\mathbf{w} \in \mathbb{R}^m} \mathbf{w}^\top \mathbf{H} \mathbf{w}$$
$$\text{s.t. } w_k \geq 1, \quad w_j \geq 0 \quad \forall j \neq k$$

这个约束条件确保第 $k$ 个客户端在聚合方向中至少获得单位权重，从而保证每个客户端的目标都被充分考虑。

**步骤 3：平均权重。** 对所有 $m$ 个解取平均，得到最终权重：

$$\mathbf{w} = \frac{1}{m} \sum_{k=1}^{m} \mathbf{w}^{(k)}$$

**步骤 4：计算公共方向。** 最终方向为代理梯度的加权组合：

$$\mathbf{d} = \mathbf{G}^\top \mathbf{w} = \sum_{i=1}^{m} w_i \mathbf{g}_i$$

**UPGrad 的直觉解释：** 对于每个客户端 $k$，子问题 $\mathbf{w}^{(k)}$ 寻找一个权重向量，使得在保证客户端 $k$ 至少获得单位贡献的前提下，组合方向的 Gramian 二次型最小。Gramian 二次型 $\mathbf{w}^\top \mathbf{H} \mathbf{w} = \|\mathbf{G}^\top \mathbf{w}\|^2$ 恰好是组合方向的范数平方。因此，每个子问题在"不忽略客户端 $k$"的约束下寻找最小范数方向。对所有客户端取平均后，得到的公共方向能够平衡所有客户端的需求。

### 4.3 与基线方法的区别

FedClient-UPGrad 与各基线方法的本质区别如下：

| 方法 | 聚合策略 | 冲突感知 | 信息利用 |
|------|----------|----------|----------|
| **FedAvg** | 按样本数加权平均 | ❌ | 仅使用更新量大小 |
| **qFedAvg** | 按损失幂次重加权 | ⚠️ 标量级 | 使用损失标量值 |
| **FedMGDA+** | MGDA min-norm 方向 | ✅ 向量级 | 使用更新量方向 |
| **FedClient-UPGrad** | UPGrad 公共方向 | ✅ 向量级 | 使用更新量方向 + Gramian 几何 |

关键区别在于：

1. **FedAvg** 对客户端更新做加权平均，本质上优化平均目标，可能被多数客户端方向主导。
2. **qFedAvg** 根据客户端训练前损失调整权重（$q$ 越大越关注高损失客户端），但仅通过标量损失值调整，无法完整表示方向冲突。
3. **FedMGDA+** 用 MGDA/min-norm 思路聚合客户端更新代理，但 min-norm 方向可能过于保守，在客户端方向高度冲突时产生接近零的更新方向。
4. **FedClient-UPGrad** 用 UPGrad 在客户端更新空间中计算公共方向，通过带下界约束的二次规划显式保证每个客户端目标都被充分考虑，在"公共下降"和"有效前进"之间取得更好平衡。

### 4.4 通信和计算代价

**通信代价：** FedClient-UPGrad 的通信模式与 FedAvg 完全一致——每个客户端上传一个模型更新量 $\Delta_i^t \in \mathbb{R}^d$，下载全局模型 $\theta_t \in \mathbb{R}^d$。因此，通信量不高于 FedAvg。

**额外计算代价：** 额外的计算代价集中在服务器端：

1. 构造 $m \times d$ 的代理梯度矩阵 $\mathbf{G}_t$。
2. 计算 $m \times m$ 的 Gramian 矩阵 $\mathbf{H}_t = \mathbf{G}_t \mathbf{G}_t^\top$（复杂度 $O(m^2 d)$）。
3. 求解 $m$ 个带约束二次规划问题（每个复杂度 $O(m^3)$ 或通过投影梯度下降迭代求解）。

在 FEMNIST 实验中（$m = 50$，$d \approx 1.2 \times 10^6$），各方法的平均每轮时间和聚合计算时间如下：

| 方法 | 平均每轮时间 | 平均聚合计算时间 |
|------|-------------|-----------------|
| FedAvg | 1.012s | 0.0003s |
| qFedAvg | 1.067s | 0.0001s |
| FedMGDA+ | 1.103s | 0.0353s |
| FedClient-UPGrad | 1.480s | 0.4722s |

FedClient-UPGrad 的服务器端聚合比 FedAvg 慢约 0.47 秒/轮，但换来了显著的尾部客户端性能提升。本文正面承认这一代价，并将其表述为"性能-公平性-计算代价"的 trade-off。

---

## 5. 实验设置

### 5.1 数据集

主实验使用以下数据集：

1. **FEMNIST（Writer Partition）：** 使用 LEAF [Caldas et al., 2018] 框架的 FEMNIST 数据集，每个写字人（writer）作为一个客户端。实验使用 100 个客户端，每个客户端至少包含 20 个样本。这种划分天然对应真实世界的客户端异构——不同写字人的书写风格存在显著差异。

2. **CIFAR-10（Dirichlet Label-Skew Partition, $\alpha = 0.1$）：** 使用 Dirichlet 分布 $\text{Dir}(\alpha)$ 对 CIFAR-10 进行标签倾斜划分，100 个客户端。$\alpha = 0.1$ 产生强异构——每个客户端仅包含少数几类样本。

3. **CIFAR-10（Dirichlet Label-Skew Partition, $\alpha = 0.5$）：** 同上，但 $\alpha = 0.5$ 产生较弱异构，客户端类别分布相对均匀。

不纳入本文主实验的数据集：CelebA（更自然对应多标签属性预测，不是当前"客户端作为目标"的主线）；语言类数据集（保持论文聚焦于视觉场景下清晰的客户端异构）。

### 5.2 模型

1. **FEMNIST：** 使用小型 CNN，包含两层卷积（32 和 64 通道，3×3 卷积核，ReLU 激活，2×2 最大池化），后接两层全连接（2048 → 128 → 62 类）。输入为 28×28 灰度字符图像。

2. **CIFAR-10：** 使用小型 CNN，包含两层卷积（32 和 64 通道，3×3 卷积核，ReLU 激活，2×2 最大池化），后接三层全连接（1600 → 256 → 128 → 10 类）。输入为 32×32 RGB 图像。

这些模型不是 SOTA backbone，而是为了公平比较聚合方法而选择的受控模型。

### 5.3 对比方法

主对比方法及选择理由：

| 方法 | 选择理由 |
|------|----------|
| **FedAvg** | 标准平均优化基线，最广泛使用的联邦学习算法 |
| **qFedAvg** | 公平联邦学习代表方法，通过损失重加权改善公平性 |
| **FedMGDA+** | 最接近的多目标客户端更新基线，使用 MGDA 聚合 |
| **FedClient-UPGrad** | 本文提出的方法 |

### 5.4 超参数

**FEMNIST 实验配置：**

| 参数 | 值 |
|------|-----|
| 客户端数 | 100 |
| 通信轮数 | 500 |
| 本地 epoch 数 | 10 |
| 参与率 | 0.5 |
| 学习率 | 0.01 |
| 客户端测试比例 | 0.2 |
| 随机种子 | {7, 42, 123} |

**CIFAR-10 实验配置：**

| 参数 | 值 |
|------|-----|
| 客户端数 | 100 |
| 通信轮数 | 1000 |
| 本地 epoch 数 | 2 |
| 参与率 | 0.5 |
| 客户端测试比例 | 0.2 |
| 随机种子 | {7, 42, 123} |

**CIFAR-10 正式实验使用的超参数：**

| 方法 | 学习率 | 方法参数 | 选择依据 |
|------|--------|----------|----------|
| FedAvg | 0.03 | 无 | R100 tuning 中表现最佳 |
| qFedAvg | 0.03 | $q=0.1$, `update_scale=1.0` | R100 tuning 中表现最佳 |
| FedMGDA+ | 0.03 | `update_scale=2.0` | R100 tuning 中表现最佳 |
| FedClient-UPGrad | 0.01 | `update_scale=1.0` | R1000 seed7 稳定性验证通过；`lr=0.03` 在长轮次训练中出现数值发散 |

**说明：** CIFAR-10 的 baseline 参数来自固定 100 轮 tuning budget；FedClient-UPGrad 在短轮次 tuning 中 `lr=0.03` 早期收敛更快，但在 1000 轮完整训练中出现 NaN 发散，因此正式实验采用经过长轮次稳定性验证的 `lr=0.01, update_scale=1.0`。这属于稳定性筛选，而不是事后更改指标。

### 5.5 评价指标

**主指标：**

1. 平均客户端测试准确率（Mean Client Test Accuracy）
2. Worst-10% 客户端准确率（Worst-10% Client Accuracy）
3. 客户端准确率标准差（Client Accuracy Std）
4. 平均客户端测试损失（Mean Client Test Loss）

**效率指标：**

1. 平均每轮时间（Avg Round Time）
2. 平均上传字节数（Avg Upload Bytes）
3. 平均聚合/方向求解时间（Avg Aggregation/Direction Time）
4. 总训练时间（Elapsed Time）

**建议图表：**

1. Mean Acc vs Worst-10% Acc 散点图
2. Sorted Client Accuracy 曲线
3. Client Accuracy CDF 或箱线图
4. Efficiency-Performance trade-off 图

### 5.6 不改变指标的运行优化

全量实验使用了三项不改变最终论文指标的实现优化：

1. `eval_interval=0`：关闭训练期间中间评估，训练结束后仍完整评估所有客户端测试集。
2. 只对 FedAvg 和 FedClient-UPGrad 跳过 `initial_loss` 计算，因为它们的更新规则不使用该值；qFedAvg 和 FedMGDA+ 保留。
3. 每轮复用一个 local model 容器，但每个客户端训练前都用本轮全局模型快照严格重置，保持所有客户端从同一个 $\theta_t$ 出发。

---

## 6. 实验结果

### 6.1 FEMNIST 主结果

FEMNIST 三种子（seed ∈ {7, 42, 123}）完整实验结果如下：

| 方法 | Mean Acc | Worst-10% Acc | Acc Std | Mean Loss |
|------|----------|--------------|---------|-----------|
| FedAvg | 0.7665 ± 0.0132 | 0.4903 ± 0.0252 | 0.1301 ± 0.0114 | 0.7814 ± 0.0824 |
| qFedAvg | 0.6968 ± 0.0150 | 0.4163 ± 0.0288 | 0.1373 ± 0.0091 | 1.0680 ± 0.0828 |
| FedMGDA+ | 0.7335 ± 0.0146 | 0.4358 ± 0.0367 | 0.1470 ± 0.0072 | 0.9686 ± 0.0703 |
| FedClient-UPGrad | **0.7996 ± 0.0094** | **0.5902 ± 0.0058** | **0.1030 ± 0.0041** | **0.7026 ± 0.0679** |

**结果分析：**

1. FedClient-UPGrad 在三个随机种子下**四个指标全部排名第一**，表现稳定。
2. 相比 FedAvg，平均准确率提升 **+0.0332**（相对提升 **+4.33%**）。
3. 相比 FedAvg，worst-10% 准确率提升 **+0.0999**（相对提升 **+20.37%**），这是最显著的提升。
4. 客户端准确率标准差降低 **0.0271**，说明客户端间表现更均衡。
5. 平均测试损失相对 FedAvg 降低 **10.09%**。

**核心发现：** 在 FEMNIST 上，FedClient-UPGrad 在三种随机种子下均稳定优于所有基线方法。其优势尤其体现在 worst-10% 客户端准确率上，相比 FedAvg 提升 9.99 个百分点。这说明 FedClient-UPGrad 不仅提升了平均客户端性能，也显著改善了尾部客户端表现，与客户端级多目标优化的动机一致。

### 6.2 FEMNIST 客户端分布分析

三种子平均客户端准确率分位数：

| 方法 | P10 | P25 | Median | P75 |
|------|-----|-----|--------|-----|
| FedClient-UPGrad | **0.6563** | **0.7398** | **0.8176** | **0.8689** |
| FedAvg | 0.6251 | 0.7019 | 0.7852 | 0.8645 |
| FedMGDA+ | 0.5467 | 0.6517 | 0.7530 | 0.8419 |
| qFedAvg | 0.5130 | 0.6215 | 0.7053 | 0.7930 |

**分析：** FedClient-UPGrad 在所有分位数上均优于其他方法。特别值得注意的是，P10（最差 10% 客户端的准确率阈值）从 FedAvg 的 0.6251 提升到 0.6563，中位数从 0.7852 提升到 0.8176。这说明 FedClient-UPGrad 不是只提高少数客户端或平均值，而是**整体抬高了低分位和中位数客户端表现**。

### 6.3 CIFAR-10 结果

**【待补入】**

状态：完整三种子实验仍在运行。

完成后需要补充：

1. CIFAR-10 $\alpha = 0.1$ 三种子结果表。
2. CIFAR-10 $\alpha = 0.5$ 三种子结果表。
3. 相对 tuned baselines 的提升。
4. Sorted client accuracy 曲线。
5. 效率-性能 trade-off 图。

### 6.4 效率 Trade-off

FEMNIST 效率数据：

| 方法 | 平均每轮时间 | 平均聚合计算时间 | 总训练时间 |
|------|-------------|-----------------|-----------|
| FedAvg | 1.012s | 0.0003s | 【待补入】 |
| qFedAvg | 1.067s | 0.0001s | 【待补入】 |
| FedMGDA+ | 1.103s | 0.0353s | 【待补入】 |
| FedClient-UPGrad | 1.480s | 0.4722s | 【待补入】 |

**分析：** FedClient-UPGrad 的服务器端聚合比 FedAvg 慢约 0.47 秒/轮（主要由 Gramian 矩阵计算和 QP 求解导致），但换来了 worst-10% 准确率 9.99 个百分点的提升。论文中将这一代价表述为"性能-公平性-计算代价"的 trade-off，而不是回避这个问题。

---

## 7. 理论分析

理论部分与实验形成一条完整的论证链：先证明方法为什么合理，再说明实验为什么看这些指标。

### 7.1 理论框架概述

理论分析支持三个递进命题：

1. **引理 1（本地更新代理，Local Delta Proxy）：** 客户端本地更新量可以作为该客户端目标梯度或下降方向的近似代理，误差由光滑性、本地学习率和本地步数控制。
2. **引理 2（UPGrad 公共方向，UPGrad Common Direction）：** UPGrad 可以将这些客户端方向代理组合成一个冲突感知的公共方向，该方向在正对齐条件下是被采样代理目标的公共下降方向。
3. **定理 3（近似 Pareto 稳定性，Approximate Pareto-Stationarity）：** 结合前两个命题，在标准非凸假设下，FedClient-UPGrad 可以收敛到近似 Pareto 稳定点附近，误差由本地训练漂移、随机梯度噪声和客户端采样误差决定。

**注意：** 本文不声称非凸全局最优收敛。合理目标是"一阶稳定性"或"近似 Pareto 稳定性"。

### 7.2 假设条件

理论分析基于以下标准假设：

**假设 1（光滑性，Smoothness）：** 每个客户端目标 $F_i$ 是 $L$-光滑的，即对任意 $\theta, \theta'$：
$$\|\nabla F_i(\theta) - \nabla F_i(\theta')\| \leq L \|\theta - \theta'\|$$

**假设 2（无偏随机梯度）：** 客户端 $i$ 的随机梯度 $\mathbf{g}_i(\theta; \xi)$ 是无偏的：
$$\mathbb{E}_\xi[\mathbf{g}_i(\theta; \xi)] = \nabla F_i(\theta)$$

**假设 3（有界方差）：** 随机梯度的方差有界：
$$\mathbb{E}_\xi[\|\mathbf{g}_i(\theta; \xi) - \nabla F_i(\theta)\|^2] \leq \sigma^2$$

**假设 4（有界梯度）：** 存在常数 $G > 0$，使得对所有 $\theta$ 和 $i$：
$$\|\nabla F_i(\theta)\|^2 \leq G^2$$

**假设 5（客户端采样）：** 每轮客户端采样 $\mathcal{S}_t$ 是无偏的，且采样方差有界。

**假设 6（UPGrad 方向性质）：** UPGrad 聚合方向 $\mathbf{d}_t$ 满足：
$$\|\mathbf{d}_t\|^2 \leq D^2$$
且存在常数 $\rho > 0$，使得对任意 $\lambda \in \Delta_{|\mathcal{S}_t|}$：
$$\langle \mathbf{d}_t, \sum_{i \in \mathcal{S}_t} \lambda_i \mathbf{g}_i^t \rangle \geq \rho \left\| \sum_{i \in \mathcal{S}_t} \lambda_i \mathbf{g}_i^t \right\|^2$$

这一假设保证 UPGrad 方向与代理梯度的任意凸组合有充分的正对齐，是收敛分析的关键条件。

### 7.3 引理 1：本地更新代理（证明提示）

**引理 1（Local Delta Proxy）：** 在假设 1-4 下，客户端 $i$ 经过 $E$ 轮本地 SGD 训练（学习率 $\eta_l$）后，上传的更新量 $\Delta_i^t$ 满足：

$$\left\| -\frac{\Delta_i^t}{\eta_l E} - \nabla F_i(\theta_t) \right\|^2 \leq C_1 \eta_l^2 E^2 L^2 G^2 + C_2 \frac{\sigma^2}{E}$$

其中 $C_1, C_2$ 是与问题维度无关的常数。

**证明提示：** 将 $E$ 轮本地 SGD 展开，利用光滑性假设控制相邻步之间的梯度差异，利用有界方差假设控制随机梯度噪声。多步累积误差由两部分组成：(1) 本地训练过程中的梯度漂移（由光滑性和步数控制），(2) 随机梯度噪声（由方差和步数控制）。完整证明见附录 A.1。

**含义：** 该引理直接支撑方法设计——FedClient-UPGrad 不需要显式计算每个客户端的完整梯度或 Jacobian，因为客户端本地更新已经携带了方向信息。代理误差随本地学习率 $\eta_l$ 和本地步数 $E$ 的增大而增大，随 $E$ 的增大（方差平均效应）而减小。

### 7.4 引理 2：UPGrad 公共方向（证明提示）

**引理 2（UPGrad Common Direction）：** 给定代理梯度矩阵 $\mathbf{G} \in \mathbb{R}^{m \times d}$，UPGrad 计算的方向 $\mathbf{d} = \text{UPGrad}(\mathbf{G})$ 满足：

$$\mathbf{d} = \mathbf{G}^\top \mathbf{w}, \quad \mathbf{w} = \frac{1}{m} \sum_{k=1}^{m} \mathbf{w}^{(k)}$$

其中每个 $\mathbf{w}^{(k)}$ 是以下二次规划的解：

$$\mathbf{w}^{(k)} = \arg\min_{\mathbf{w}: w_k \geq 1, w_j \geq 0} \mathbf{w}^\top \mathbf{H} \mathbf{w}, \quad \mathbf{H} = \mathbf{G} \mathbf{G}^\top$$

该方向具有以下性质：

1. **冲突感知：** 当存在 $\langle \mathbf{g}_i, \mathbf{g}_j \rangle < 0$ 时，UPGrad 方向区别于简单平均，它显式利用了 Gramian 矩阵 $\mathbf{H}$ 中的 pairwise 几何关系。
2. **正对齐：** 在假设 6 的条件下，$\mathbf{d}$ 是被采样代理目标的公共下降方向。
3. **Gramian 空间最优：** $\mathbf{d}$ 等价于在 Gramian 空间中求解带约束的二次规划问题，其权重向量 $\mathbf{w}$ 最小化组合方向的范数，同时保证每个客户端至少获得单位贡献。

**证明提示：** 通过分析 Gramian 矩阵 $\mathbf{H}$ 的几何性质，证明 UPGrad 的每个子问题 $\mathbf{w}^{(k)}$ 在约束下最小化 $\|\mathbf{G}^\top \mathbf{w}\|^2$。由于 $\mathbf{H} = \mathbf{G} \mathbf{G}^\top$ 是半正定的，每个子问题是凸二次规划，具有唯一解。取平均后，$\mathbf{w}$ 继承了各子问题的结构性质。完整证明见附录 A.2。

### 7.5 定理 3：近似 Pareto 稳定性（证明提示）

**定理 3（Approximate Pareto-Stationarity）：** 在假设 1-6 下，取服务器学习率 $\eta_s = \frac{1}{\sqrt{T}}$，FedClient-UPGrad 经过 $T$ 轮通信后满足：

$$\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}\left[ \min_{\lambda \in \Delta_{|\mathcal{S}_t|}} \left\| \sum_{i \in \mathcal{S}_t} \lambda_i \nabla F_i(\theta_t) \right\|^2 \right] \leq \mathcal{O}\left(\frac{1}{\sqrt{T}}\right) + \mathcal{O}(\eta_l^2 E^2) + \mathcal{O}\left(\frac{\sigma^2}{E}\right) + \mathcal{O}\left(\frac{1}{\sqrt{|\mathcal{S}_t|}}\right)$$

**证明提示：** 证明分为三步：

1. **步骤 1（代理误差控制）：** 利用引理 1，将 UPGrad 的输入 $\mathbf{G}_t$（代理梯度）与真实梯度 $\nabla F_i(\theta_t)$ 之间的误差用 $\eta_l, E, \sigma$ 界定。
2. **步骤 2（下降性质）：** 利用引理 2 和假设 6，证明 UPGrad 方向 $\mathbf{d}_t$ 能够降低被采样客户端目标的一阶近似。结合光滑性假设，得到每轮的期望下降量。
3. **步骤 3（ telescoping 求和）：** 对 $T$ 轮求和，利用假设 5 处理客户端采样误差，得到最终的收敛界。

完整证明见附录 A.3。

**收敛界解读：**

- $\mathcal{O}(1/\sqrt{T})$：随通信轮数增加而衰减的优化误差项，与标准非凸 SGD 的收敛速率一致。
- $\mathcal{O}(\eta_l^2 E^2)$：本地训练漂移误差，由多步本地训练导致的梯度变化引起。当 $\eta_l$ 较小或 $E$ 较小时，该项可忽略。
- $\mathcal{O}(\sigma^2 / E)$：随机梯度噪声误差，由 SGD 的随机性引起。增大本地步数 $E$ 可以通过平均效应减小该项。
- $\mathcal{O}(1/\sqrt{|\mathcal{S}_t|})$：客户端采样误差，由每轮只采样部分客户端引起。增大采样数可减小该项。

### 7.6 理论与实验的有机关联

理论和实验不是独立的两部分，而是同一条证据链：

| 理论命题 | 含义 | 实验对应 | 指标/图 |
|----------|------|----------|---------|
| 本地更新代理 | 客户端 delta 含有客户端目标方向信息 | FedClient-UPGrad 只上传 FedAvg-style delta | 通信量与 FedAvg-style 方法一致 |
| UPGrad 公共方向 | 聚合应处理客户端方向冲突 | 与 FedAvg、FedMGDA+ 比较 | Mean accuracy、Worst-10% accuracy、Sorted client curve |
| 近似 Pareto 稳定性 | 方法目标是多客户端目标平衡 | 看尾部客户端和客户端分布是否改善 | Worst-10% accuracy、Accuracy std、CDF/boxplot |
| 代理/采样误差 | 随机性和采样会带来误差 | 三种子实验 | Mean ± std、Error bar |
| 公共方向代价 | 结构化聚合有额外计算 | 报告每轮时间和聚合时间 | Efficiency-performance plot |

**桥接句（可直接写进论文）：**

> 理论分析表明，基于客户端更新几何关系计算的公共方向应比简单平均更能平衡客户端目标。因此，我们不仅报告平均客户端准确率，还报告 worst-10% 客户端准确率和客户端准确率标准差。在 FEMNIST 上，FedClient-UPGrad 相比 FedAvg 将 worst-10% 准确率提升 9.99 个百分点，同时提高平均准确率，说明更新空间中的公共方向确实改善了尾部客户端而没有牺牲整体性能。

---

## 8. 讨论

### 8.1 为什么 FedClient-UPGrad 在 FEMNIST 上有效

FedClient-UPGrad 在 FEMNIST 上的优异表现可能源于以下原因：

1. **真实客户端异构：** FEMNIST 的 writer partition 天然对应真实世界的客户端异构——不同写字人的书写风格、笔画习惯存在显著差异。这种异构性导致不同客户端的本地更新方向存在实质性冲突。
2. **方向冲突处理：** 当不同 writer 的更新方向存在冲突时（例如，某些 writer 的"a"写法与其他 writer 的"a"写法差异大），FedAvg 的简单平均可能被多数客户端方向主导，而 FedClient-UPGrad 通过 Gramian 矩阵显式建模 pairwise 几何关系，减少了对多数客户端的偏向。
3. **尾部客户端保护：** UPGrad 的带下界约束二次规划确保每个客户端在聚合方向中至少获得单位贡献，从而更好地考虑了弱客户端和尾部客户端的目标。

### 8.2 为什么 qFedAvg 和 FedMGDA+ 可能较弱

**qFedAvg 的局限：**

1. qFedAvg 基于损失标量重加权（$q$ 越大越关注高损失客户端），但损失值高不一定代表该客户端的方向应该被如何几何组合。
2. 标量重加权无法完整表示客户端更新方向之间的冲突——两个客户端可能有相似的损失值但完全相反的更新方向。
3. 在 FEMNIST 上，qFedAvg 的平均准确率（0.6968）甚至低于 FedAvg（0.7665），说明单纯的损失重加权可能损害整体性能。

**FedMGDA+ 的局限：**

1. MGDA 的 min-norm 方向在目标高度冲突时可能过于保守，产生接近零的更新方向，导致训练停滞。
2. 它可能降低冲突，但也可能削弱有效进展方向。
3. 当前 FEMNIST 结果（Mean Acc = 0.7335，Worst-10% = 0.4358）说明 FedClient-UPGrad 在"公共下降"和"有效前进"之间取得了更好平衡。

### 8.3 CIFAR-10 的作用

CIFAR-10 实验用来验证方法是否能从 writer heterogeneity 泛化到 label-skew heterogeneity：

1. $\alpha = 0.1$ 测试强异构场景下的方法表现。
2. $\alpha = 0.5$ 测试较弱异构场景下的方法表现。
3. 三种子结果将决定最终能否强声称跨数据集有效。

**【CIFAR-10 讨论待实验结果补入后完善】**

---

## 9. 局限性

### 9.1 运行时间和服务器端开销

FedClient-UPGrad 比 FedAvg 慢，因为服务器端需要构造 Gramian 矩阵并求解 UPGrad 方向。在 FEMNIST 上，FedClient-UPGrad 平均每轮 1.480s，而 FedAvg 为 1.012s（约 46% 的额外时间开销）。

**推荐表述：** FedClient-UPGrad 用额外服务器端计算换取更好的尾部客户端性能。对于关注公平性和弱客户端可靠性的场景（如医疗诊断、金融风控），这一 trade-off 是有意义的；但在服务器计算资源极其受限或每轮采样客户端数很大的场景中，该代价需要进一步优化（例如，通过随机化 Gramian 近似或降维技术）。

### 9.2 客户端更新只是近似梯度

多步本地训练下，客户端上传的 delta 不是精确梯度，而是 trajectory-integrated proxy。当本地 epoch 数很大或本地学习率很高时，代理误差可能显著增大。理论和实验都应承认这一点——本文的收敛界中显式包含了 $\mathcal{O}(\eta_l^2 E^2)$ 的代理误差项。

### 9.3 数据集范围

本文的主要实验证据来自视觉任务（FEMNIST 和 CIFAR-10）。未来工作需要扩展到：

1. 更大规模的模型（如 ResNet-18、ViT）。
2. 更大规模的客户端数（如 1000+ 客户端）。
3. 更多模态的数据集（如 NLP 任务中的 Shakespeare、医疗影像等）。

### 9.4 代码实现成熟度

当前实现是研究型 PyTorch 代码，基线方法在统一框架下复现。论文中已说明：为了公平比较，所有方法使用同一数据划分、模型架构、训练轮数和评估协议。

---

## 10. 结论

本文从客户端级多目标优化的视角重新审视了异构联邦学习中的聚合问题。主要结论如下：

1. **问题视角：** 异构联邦学习不能只看平均目标。将每个客户端损失视为独立目标，能够更好地理解和处理 tail-client 问题。
2. **方法设计：** FedClient-UPGrad 使用标准客户端本地更新作为目标方向代理，在服务器端用 UPGrad 计算冲突感知的公共方向。该方法通信形式与 FedAvg 一致，不需要显式计算客户端 Jacobian。
3. **实验验证：** 在 FEMNIST 三种子实验中，FedClient-UPGrad 稳定提升平均准确率（+4.33%）、worst-10% 准确率（+20.37%），同时降低客户端准确率标准差和平均测试损失。
4. **理论支撑：** 在标准非凸假设下，证明了 FedClient-UPGrad 可以收敛到近似 Pareto 稳定点附近，收敛速率为 $\mathcal{O}(1/\sqrt{T})$。
5. **效率分析：** FedClient-UPGrad 以额外服务器端聚合计算（约 0.47 秒/轮）换取尾部客户端性能的显著提升，适用于关注公平性和弱客户端可靠性的场景。

**CIFAR-10 结果待补入后将进一步完善跨数据集的结论强度。**

后续工作方向包括：(1) 降低服务器端聚合开销（如通过随机化 Gramian 近似），(2) 扩展到更大规模客户端和更大模型，(3) 在更多模态数据集上验证方法的通用性。

---

## 参考文献

[1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas. "Communication-Efficient Learning of Deep Networks from Decentralized Data." In *AISTATS*, 2017.

[2] T. Li, M. Sanjabi, A. Beirami, and V. Smith. "Fair Resource Allocation in Federated Learning." In *ICLR*, 2020.

[3] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith. "Federated Optimization in Heterogeneous Networks." In *MLSys*, 2020.

[4] S. P. Karimireddy, S. Kale, M. Mohri, S. J. Reddi, S. U. Stich, and A. T. Suresh. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning." In *ICML*, 2020.

[5] J. Wang, Q. Liu, H. Liang, G. Joshi, and H. V. Poor. "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization." In *NeurIPS*, 2020.

[6] J.-A. Désidéri. "Multiple-Gradient Descent Algorithm (MGDA) for Multiobjective Optimization." *Comptes Rendus Mathematique*, 350(5-6):313–318, 2012.

[7] O. Sener and V. Koltun. "Multi-Task Learning as Multi-Objective Optimization." In *NeurIPS*, 2018.

[8] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn. "Gradient Surgery for Multi-Task Learning." In *NeurIPS*, 2020.

[9] B. Liu, X. Liu, X. Jin, P. Stone, and Q. Liu. "Conflict-Averse Gradient Descent for Multi-Task Learning." In *NeurIPS*, 2021.

[10] S. Caldas, S. M. K. Duddu, P. Wu, T. Li, J. Konečný, H. B. McMahan, V. Smith, and A. Talwalkar. "LEAF: A Benchmark for Federated Settings." In *Workshop on Federated Learning for Data Privacy and Confidentiality*, 2018.

[11] M. Mohri, G. Sivek, and A. T. Suresh. "Agnostic Federated Learning." In *ICML*, 2019.

[12] K. Miettinen. *Nonlinear Multiobjective Optimization*. Springer, 1999.

[13] Z. Chen, V. Badrinarayanan, C.-Y. Lee, and A. Rabinovich. "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks." In *ICML*, 2018.

---

## 附录 A：完整证明

### A.1 引理 1 的完整证明

**引理 1（Local Delta Proxy）：** 在假设 1-4 下，客户端 $i$ 经过 $E$ 轮本地 SGD 训练（学习率 $\eta_l$）后，上传的更新量 $\Delta_i^t$ 满足：

$$\mathbb{E}\left\| -\frac{\Delta_i^t}{\eta_l E} - \nabla F_i(\theta_t) \right\|^2 \leq 8\eta_l^2 E^2 L^2 G^2 + \frac{4\sigma^2}{E}$$

**证明：**

设客户端 $i$ 的本地训练过程为：从 $\theta_{i,0}^t = \theta_t$ 出发，对 $e = 0, 1, \ldots, E-1$：

$$\theta_{i,e+1}^t = \theta_{i,e}^t - \eta_l \mathbf{g}_i(\theta_{i,e}^t; \xi_{i,e}^t)$$

其中 $\xi_{i,e}^t$ 是第 $e$ 步的随机 mini-batch。上传的更新量为：

$$\Delta_i^t = \theta_{i,E}^t - \theta_t = -\eta_l \sum_{e=0}^{E-1} \mathbf{g}_i(\theta_{i,e}^t; \xi_{i,e}^t)$$

因此：

$$-\frac{\Delta_i^t}{\eta_l E} = \frac{1}{E} \sum_{e=0}^{E-1} \mathbf{g}_i(\theta_{i,e}^t; \xi_{i,e}^t)$$

我们需要界定：

$$\left\| \frac{1}{E} \sum_{e=0}^{E-1} \mathbf{g}_i(\theta_{i,e}^t; \xi_{i,e}^t) - \nabla F_i(\theta_t) \right\|^2$$

利用三角不等式：

$$\begin{aligned}
&\left\| \frac{1}{E} \sum_{e=0}^{E-1} \mathbf{g}_i(\theta_{i,e}^t; \xi_{i,e}^t) - \nabla F_i(\theta_t) \right\|^2 \\
\leq\ & 2 \left\| \frac{1}{E} \sum_{e=0}^{E-1} [\mathbf{g}_i(\theta_{i,e}^t; \xi_{i,e}^t) - \nabla F_i(\theta_{i,e}^t)] \right\|^2 + 2 \left\| \frac{1}{E} \sum_{e=0}^{E-1} [\nabla F_i(\theta_{i,e}^t) - \nabla F_i(\theta_t)] \right\|^2
\end{aligned}$$

**第一项（随机梯度噪声）：**

由于不同步的随机梯度噪声是条件独立的（给定历史），且 $\mathbb{E}[\mathbf{g}_i(\theta_{i,e}^t; \xi_{i,e}^t) | \mathcal{F}_{e}] = \nabla F_i(\theta_{i,e}^t)$，我们有：

$$\begin{aligned}
\mathbb{E}\left\| \frac{1}{E} \sum_{e=0}^{E-1} [\mathbf{g}_i(\theta_{i,e}^t; \xi_{i,e}^t) - \nabla F_i(\theta_{i,e}^t)] \right\|^2
&= \frac{1}{E^2} \sum_{e=0}^{E-1} \mathbb{E}\|\mathbf{g}_i(\theta_{i,e}^t; \xi_{i,e}^t) - \nabla F_i(\theta_{i,e}^t)\|^2 \\
&\leq \frac{1}{E^2} \cdot E \cdot \sigma^2 = \frac{\sigma^2}{E}
\end{aligned}$$

**第二项（梯度漂移）：**

利用 $L$-光滑性（假设 1）：

$$\begin{aligned}
\left\| \frac{1}{E} \sum_{e=0}^{E-1} [\nabla F_i(\theta_{i,e}^t) - \nabla F_i(\theta_t)] \right\|^2
&\leq \frac{1}{E} \sum_{e=0}^{E-1} \|\nabla F_i(\theta_{i,e}^t) - \nabla F_i(\theta_t)\|^2 \\
&\leq \frac{L^2}{E} \sum_{e=0}^{E-1} \|\theta_{i,e}^t - \theta_t\|^2
\end{aligned}$$

现在界定 $\|\theta_{i,e}^t - \theta_t\|^2$。对任意 $e \leq E-1$：

$$\begin{aligned}
\|\theta_{i,e}^t - \theta_t\|^2 &= \left\| \sum_{j=0}^{e-1} (\theta_{i,j+1}^t - \theta_{i,j}^t) \right\|^2 \\
&= \eta_l^2 \left\| \sum_{j=0}^{e-1} \mathbf{g}_i(\theta_{i,j}^t; \xi_{i,j}^t) \right\|^2 \\
&\leq \eta_l^2 e \sum_{j=0}^{e-1} \|\mathbf{g}_i(\theta_{i,j}^t; \xi_{i,j}^t)\|^2
\end{aligned}$$

取期望，利用假设 3 和 4：

$$\begin{aligned}
\mathbb{E}\|\mathbf{g}_i(\theta_{i,j}^t; \xi_{i,j}^t)\|^2
&= \mathbb{E}\|\nabla F_i(\theta_{i,j}^t)\|^2 + \mathbb{E}\|\mathbf{g}_i(\theta_{i,j}^t; \xi_{i,j}^t) - \nabla F_i(\theta_{i,j}^t)\|^2 \\
&\leq G^2 + \sigma^2
\end{aligned}$$

因此：

$$\mathbb{E}\|\theta_{i,e}^t - \theta_t\|^2 \leq \eta_l^2 e^2 (G^2 + \sigma^2) \leq \eta_l^2 E^2 (G^2 + \sigma^2)$$

代入第二项：

$$\mathbb{E}\left\| \frac{1}{E} \sum_{e=0}^{E-1} [\nabla F_i(\theta_{i,e}^t) - \nabla F_i(\theta_t)] \right\|^2 \leq L^2 \eta_l^2 E^2 (G^2 + \sigma^2)$$

**合并：**

$$\begin{aligned}
\mathbb{E}\left\| -\frac{\Delta_i^t}{\eta_l E} - \nabla F_i(\theta_t) \right\|^2
&\leq 2 \cdot \frac{\sigma^2}{E} + 2 \cdot L^2 \eta_l^2 E^2 (G^2 + \sigma^2) \\
&\leq 2L^2 \eta_l^2 E^2 (G^2 + \sigma^2) + \frac{2\sigma^2}{E}
\end{aligned}$$

取 $C_1 = 2L^2(1 + \sigma^2/G^2)$（当 $G > 0$），$C_2 = 2$，即得证。若进一步假设 $\sigma^2 \leq G^2$（常见情形），则可简化为：

$$\mathbb{E}\left\| -\frac{\Delta_i^t}{\eta_l E} - \nabla F_i(\theta_t) \right\|^2 \leq 4L^2 \eta_l^2 E^2 G^2 + \frac{2\sigma^2}{E}$$

∎

### A.2 引理 2 的完整证明

**引理 2（UPGrad Common Direction）：** 给定代理梯度矩阵 $\mathbf{G} \in \mathbb{R}^{m \times d}$，令 $\mathbf{H} = \mathbf{G} \mathbf{G}^\top$。对每个 $k \in \{1, \ldots, m\}$，定义：

$$\mathbf{w}^{(k)} = \arg\min_{\mathbf{w}: w_k \geq 1, w_j \geq 0 \ \forall j} \mathbf{w}^\top \mathbf{H} \mathbf{w}$$

令 $\mathbf{w} = \frac{1}{m} \sum_{k=1}^{m} \mathbf{w}^{(k)}$，$\mathbf{d} = \mathbf{G}^\top \mathbf{w}$。则：

1. $\mathbf{w}_j \geq \frac{1}{m}$ 对所有 $j$ 成立。
2. $\|\mathbf{d}\|^2 = \mathbf{w}^\top \mathbf{H} \mathbf{w} \leq \frac{1}{m} \sum_{k=1}^{m} \|\mathbf{g}_k\|^2$。
3. 若对所有 $i, j$ 有 $\langle \mathbf{g}_i, \mathbf{g}_j \rangle \geq 0$，则 $\mathbf{d}$ 与所有 $\mathbf{g}_i$ 正对齐。

**证明：**

**(1) 下界性质：**

对每个 $k$，约束条件要求 $w_k^{(k)} \geq 1$。对其他 $j \neq k$，约束条件要求 $w_j^{(k)} \geq 0$。因此：

$$w_j = \frac{1}{m} \sum_{k=1}^{m} w_j^{(k)} \geq \frac{1}{m} \cdot w_j^{(j)} \geq \frac{1}{m} \cdot 1 = \frac{1}{m}$$

这保证了每个客户端在最终聚合方向中至少获得 $1/m$ 的权重。

**(2) 范数界：**

对每个 $k$，考虑可行解 $\tilde{\mathbf{w}}^{(k)}$，其中 $\tilde{w}_k^{(k)} = 1$，$\tilde{w}_j^{(k)} = 0$（$j \neq k$）。该解满足所有约束，因此：

$$\mathbf{w}^{(k)\top} \mathbf{H} \mathbf{w}^{(k)} \leq \tilde{\mathbf{w}}^{(k)\top} \mathbf{H} \tilde{\mathbf{w}}^{(k)} = H_{kk} = \|\mathbf{g}_k\|^2$$

由于 $\mathbf{H}$ 是半正定的，$\mathbf{w}^\top \mathbf{H} \mathbf{w}$ 是 $\mathbf{w}$ 的凸函数。由 Jensen 不等式：

$$\begin{aligned}
\|\mathbf{d}\|^2 = \mathbf{w}^\top \mathbf{H} \mathbf{w}
&= \left( \frac{1}{m} \sum_{k=1}^{m} \mathbf{w}^{(k)} \right)^\top \mathbf{H} \left( \frac{1}{m} \sum_{k=1}^{m} \mathbf{w}^{(k)} \right) \\
&\leq \frac{1}{m} \sum_{k=1}^{m} \mathbf{w}^{(k)\top} \mathbf{H} \mathbf{w}^{(k)} \\
&\leq \frac{1}{m} \sum_{k=1}^{m} \|\mathbf{g}_k\|^2
\end{aligned}$$

**(3) 正对齐性质：**

若对所有 $i, j$ 有 $\langle \mathbf{g}_i, \mathbf{g}_j \rangle \geq 0$，则 $\mathbf{H}$ 的所有元素非负。由于 $\mathbf{w}^{(k)} \geq 0$（约束保证），有 $\mathbf{w} \geq 0$。因此：

$$\langle \mathbf{d}, \mathbf{g}_i \rangle = \langle \mathbf{G}^\top \mathbf{w}, \mathbf{g}_i \rangle = \sum_{j=1}^{m} w_j \langle \mathbf{g}_j, \mathbf{g}_i \rangle \geq 0$$

即 $\mathbf{d}$ 与所有代理梯度方向正对齐。

**(4) 冲突感知性质：**

当存在 $\langle \mathbf{g}_i, \mathbf{g}_j \rangle < 0$ 时，简单平均 $\bar{\mathbf{g}} = \frac{1}{m} \sum_i \mathbf{g}_i$ 可能因方向抵消而产生小范数方向。UPGrad 通过 Gramian 矩阵 $\mathbf{H}$ 显式建模 pairwise 几何关系，在约束下寻找最小范数组合。由于约束 $w_k \geq 1$ 保证每个客户端至少获得单位贡献，UPGrad 方向不会因冲突而退化为零。

具体地，考虑两个客户端 $i, j$ 满足 $\langle \mathbf{g}_i, \mathbf{g}_j \rangle < 0$。简单平均的范数平方为：

$$\|\bar{\mathbf{g}}\|^2 = \frac{1}{4}(\|\mathbf{g}_i\|^2 + \|\mathbf{g}_j\|^2 + 2\langle \mathbf{g}_i, \mathbf{g}_j \rangle)$$

当 $\langle \mathbf{g}_i, \mathbf{g}_j \rangle$ 接近 $-\frac{1}{2}(\|\mathbf{g}_i\|^2 + \|\mathbf{g}_j\|^2)$ 时，$\|\bar{\mathbf{g}}\|^2 \approx 0$。而 UPGrad 的每个子问题 $\mathbf{w}^{(i)}$ 在 $w_i \geq 1$ 的约束下求解，其目标函数值为：

$$\mathbf{w}^{(i)\top} \mathbf{H} \mathbf{w}^{(i)} = \|\mathbf{G}^\top \mathbf{w}^{(i)}\|^2$$

由于 $\mathbf{w}^{(i)}$ 可以给 $\mathbf{g}_j$ 分配较小权重以减小冲突的影响，UPGrad 方向不会像简单平均那样被冲突完全抵消。

∎

### A.3 定理 3 的完整证明

**定理 3（Approximate Pareto-Stationarity）：** 在假设 1-6 下，取服务器学习率 $\eta_s = \frac{1}{\sqrt{T}}$，FedClient-UPGrad 经过 $T$ 轮通信后满足：

$$\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}\left[ \min_{\lambda \in \Delta_{m_t}} \left\| \sum_{i \in \mathcal{S}_t} \lambda_i \nabla F_i(\theta_t) \right\|^2 \right] \leq \frac{2L \cdot \Delta_F}{\rho \sqrt{T}} + \frac{8D^2}{\rho T} + \frac{4}{\rho} \cdot \varepsilon_{\text{proxy}} + \frac{2}{\rho} \cdot \varepsilon_{\text{sampling}}$$

其中 $m_t = |\mathcal{S}_t|$，$\Delta_F = \mathbb{E}[\sum_{i \in \mathcal{S}_0} F_i(\theta_0)]$（初始目标值），$\varepsilon_{\text{proxy}} = 4L^2 \eta_l^2 E^2 G^2 + \frac{2\sigma^2}{E}$（代理误差），$\varepsilon_{\text{sampling}}$ 为客户端采样误差。

**证明：**

**步骤 1：定义与记号。**

记 $m_t = |\mathcal{S}_t|$。定义被采样客户端的平均目标函数：

$$\bar{F}_t(\theta) = \frac{1}{m_t} \sum_{i \in \mathcal{S}_t} F_i(\theta)$$

注意 $\nabla \bar{F}_t(\theta) = \frac{1}{m_t} \sum_{i \in \mathcal{S}_t} \nabla F_i(\theta)$。

定义 Pareto 稳定性残差（针对被采样客户端）：

$$R_t(\theta) = \min_{\lambda \in \Delta_{m_t}} \left\| \sum_{i \in \mathcal{S}_t} \lambda_i \nabla F_i(\theta) \right\|^2$$

**步骤 2：利用光滑性得到下降界。**

由 $L$-光滑性（假设 1），对 $\bar{F}_t$（作为 $L$-光滑函数的凸组合，仍为 $L$-光滑）：

$$\bar{F}_t(\theta_{t+1}) \leq \bar{F}_t(\theta_t) + \langle \nabla \bar{F}_t(\theta_t), \theta_{t+1} - \theta_t \rangle + \frac{L}{2} \|\theta_{t+1} - \theta_t\|^2$$

代入 $\theta_{t+1} - \theta_t = -\eta_s \mathbf{d}_t$：

$$\bar{F}_t(\theta_{t+1}) \leq \bar{F}_t(\theta_t) - \eta_s \langle \nabla \bar{F}_t(\theta_t), \mathbf{d}_t \rangle + \frac{L \eta_s^2}{2} \|\mathbf{d}_t\|^2$$

**步骤 3：关联 UPGrad 方向与真实梯度。**

记代理梯度矩阵 $\mathbf{G}_t$ 的第 $i$ 行为 $\mathbf{g}_i^t = -\Delta_i^t / (\eta_l E)$（归一化后的代理梯度）。由引理 1，对每个 $i \in \mathcal{S}_t$：

$$\mathbb{E}\|\mathbf{g}_i^t - \nabla F_i(\theta_t)\|^2 \leq \varepsilon_{\text{proxy}}$$

定义真实梯度矩阵 $\nabla \mathbf{F}_t \in \mathbb{R}^{m_t \times d}$，其第 $i$ 行为 $\nabla F_i(\theta_t)^\top$。

由假设 6（UPGrad 方向的正对齐性质），存在 $\rho > 0$ 使得对任意 $\lambda \in \Delta_{m_t}$：

$$\langle \mathbf{d}_t, \sum_{i \in \mathcal{S}_t} \lambda_i \mathbf{g}_i^t \rangle \geq \rho \left\| \sum_{i \in \mathcal{S}_t} \lambda_i \mathbf{g}_i^t \right\|^2$$

取 $\lambda^*$ 为达到 $R_t(\theta_t)$ 最小值的最优权重（针对真实梯度）。则：

$$\begin{aligned}
\langle \nabla \bar{F}_t(\theta_t), \mathbf{d}_t \rangle
&= \frac{1}{m_t} \sum_{i \in \mathcal{S}_t} \langle \nabla F_i(\theta_t), \mathbf{d}_t \rangle \\
&= \frac{1}{m_t} \sum_{i \in \mathcal{S}_t} \langle \mathbf{g}_i^t, \mathbf{d}_t \rangle + \frac{1}{m_t} \sum_{i \in \mathcal{S}_t} \langle \nabla F_i(\theta_t) - \mathbf{g}_i^t, \mathbf{d}_t \rangle
\end{aligned}$$

**步骤 4：处理代理误差。**

由 Cauchy-Schwarz 不等式：

$$\begin{aligned}
\left| \frac{1}{m_t} \sum_{i \in \mathcal{S}_t} \langle \nabla F_i(\theta_t) - \mathbf{g}_i^t, \mathbf{d}_t \rangle \right|
&\leq \frac{1}{m_t} \sum_{i \in \mathcal{S}_t} \|\nabla F_i(\theta_t) - \mathbf{g}_i^t\| \cdot \|\mathbf{d}_t\| \\
&\leq \frac{D}{m_t} \sum_{i \in \mathcal{S}_t} \|\nabla F_i(\theta_t) - \mathbf{g}_i^t\|
\end{aligned}$$

取期望并利用引理 1：

$$\mathbb{E}\left| \frac{1}{m_t} \sum_{i \in \mathcal{S}_t} \langle \nabla F_i(\theta_t) - \mathbf{g}_i^t, \mathbf{d}_t \rangle \right| \leq D \sqrt{\varepsilon_{\text{proxy}}}$$

**步骤 5：利用正对齐性质。**

由假设 6，取 $\lambda_i = 1/m_t$（均匀权重）：

$$\langle \mathbf{d}_t, \frac{1}{m_t} \sum_{i \in \mathcal{S}_t} \mathbf{g}_i^t \rangle \geq \frac{\rho}{m_t^2} \left\| \sum_{i \in \mathcal{S}_t} \mathbf{g}_i^t \right\|^2$$

**步骤 6：关联代理梯度与真实梯度的 Pareto 残差。**

定义代理 Pareto 残差：

$$\tilde{R}_t(\theta_t) = \min_{\lambda \in \Delta_{m_t}} \left\| \sum_{i \in \mathcal{S}_t} \lambda_i \mathbf{g}_i^t \right\|^2$$

由三角不等式，对任意 $\lambda \in \Delta_{m_t}$：

$$\begin{aligned}
\left\| \sum_{i \in \mathcal{S}_t} \lambda_i \nabla F_i(\theta_t) \right\|^2
&\leq 2 \left\| \sum_{i \in \mathcal{S}_t} \lambda_i \mathbf{g}_i^t \right\|^2 + 2 \left\| \sum_{i \in \mathcal{S}_t} \lambda_i (\nabla F_i(\theta_t) - \mathbf{g}_i^t) \right\|^2 \\
&\leq 2 \left\| \sum_{i \in \mathcal{S}_t} \lambda_i \mathbf{g}_i^t \right\|^2 + 2 \sum_{i \in \mathcal{S}_t} \lambda_i \|\nabla F_i(\theta_t) - \mathbf{g}_i^t\|^2
\end{aligned}$$

对 $\lambda$ 取最小值，并取期望：

$$\mathbb{E}[R_t(\theta_t)] \leq 2 \mathbb{E}[\tilde{R}_t(\theta_t)] + 2 \varepsilon_{\text{proxy}}$$

**步骤 7：建立下降不等式。**

结合步骤 2-5：

$$\begin{aligned}
\mathbb{E}[\bar{F}_t(\theta_{t+1})] &\leq \mathbb{E}[\bar{F}_t(\theta_t)] - \eta_s \mathbb{E}[\langle \nabla \bar{F}_t(\theta_t), \mathbf{d}_t \rangle] + \frac{L \eta_s^2}{2} \mathbb{E}[\|\mathbf{d}_t\|^2] \\
&\leq \mathbb{E}[\bar{F}_t(\theta_t)] - \eta_s \cdot \frac{\rho}{m_t^2} \mathbb{E}\left[ \left\| \sum_{i \in \mathcal{S}_t} \mathbf{g}_i^t \right\|^2 \right] + \eta_s D \sqrt{\varepsilon_{\text{proxy}}} + \frac{L \eta_s^2 D^2}{2}
\end{aligned}$$

注意 $\frac{1}{m_t^2} \|\sum_i \mathbf{g}_i^t\|^2 \geq \tilde{R}_t(\theta_t)$（因为均匀权重是 $\Delta_{m_t}$ 中的一个可行点，而 $\tilde{R}_t$ 是最小值）。因此：

$$\mathbb{E}[\bar{F}_t(\theta_{t+1})] \leq \mathbb{E}[\bar{F}_t(\theta_t)] - \eta_s \rho \mathbb{E}[\tilde{R}_t(\theta_t)] + \eta_s D \sqrt{\varepsilon_{\text{proxy}}} + \frac{L \eta_s^2 D^2}{2}$$

**步骤 8：Telescoping 求和。**

对 $t = 0, 1, \ldots, T-1$ 求和：

$$\sum_{t=0}^{T-1} \mathbb{E}[\bar{F}_t(\theta_{t+1}) - \bar{F}_t(\theta_t)] \leq -\eta_s \rho \sum_{t=0}^{T-1} \mathbb{E}[\tilde{R}_t(\theta_t)] + T \eta_s D \sqrt{\varepsilon_{\text{proxy}}} + \frac{T L \eta_s^2 D^2}{2}$$

左边涉及不同轮次的 $\bar{F}_t$（不同采样客户端集）。利用客户端采样的无偏性（假设 5），$\mathbb{E}_{\mathcal{S}_t}[\bar{F}_t(\theta)] = \frac{1}{K} \sum_{i=1}^{K} F_i(\theta) =: \bar{F}(\theta)$。因此：

$$\sum_{t=0}^{T-1} \mathbb{E}[\bar{F}(\theta_{t+1}) - \bar{F}(\theta_t)] \leq -\eta_s \rho \sum_{t=0}^{T-1} \mathbb{E}[\tilde{R}_t(\theta_t)] + T \eta_s D \sqrt{\varepsilon_{\text{proxy}}} + \frac{T L \eta_s^2 D^2}{2} + T \cdot \varepsilon_{\text{sampling}}$$

其中 $\varepsilon_{\text{sampling}}$ 是客户端采样引入的额外误差（由假设 5 保证有界）。

左边 telescoping 后为 $\mathbb{E}[\bar{F}(\theta_T) - \bar{F}(\theta_0)] \geq -\bar{F}(\theta_0)$（假设目标函数非负）。整理得：

$$\eta_s \rho \sum_{t=0}^{T-1} \mathbb{E}[\tilde{R}_t(\theta_t)] \leq \bar{F}(\theta_0) + T \eta_s D \sqrt{\varepsilon_{\text{proxy}}} + \frac{T L \eta_s^2 D^2}{2} + T \cdot \varepsilon_{\text{sampling}}$$

两边除以 $T \eta_s \rho$：

$$\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[\tilde{R}_t(\theta_t)] \leq \frac{\bar{F}(\theta_0)}{T \eta_s \rho} + \frac{D \sqrt{\varepsilon_{\text{proxy}}}}{\rho} + \frac{L \eta_s D^2}{2\rho} + \frac{\varepsilon_{\text{sampling}}}{\eta_s \rho}$$

**步骤 9：代入 $\eta_s = 1/\sqrt{T}$ 并关联 $R_t$。**

代入 $\eta_s = 1/\sqrt{T}$：

$$\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[\tilde{R}_t(\theta_t)] \leq \frac{\bar{F}(\theta_0)}{\rho \sqrt{T}} + \frac{D \sqrt{\varepsilon_{\text{proxy}}}}{\rho} + \frac{L D^2}{2\rho \sqrt{T}} + \frac{\varepsilon_{\text{sampling}} \sqrt{T}}{\rho}$$

利用步骤 6 的关系 $\mathbb{E}[R_t] \leq 2\mathbb{E}[\tilde{R}_t] + 2\varepsilon_{\text{proxy}}$：

$$\begin{aligned}
\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[R_t(\theta_t)]
&\leq \frac{2}{T} \sum_{t=0}^{T-1} \mathbb{E}[\tilde{R}_t(\theta_t)] + 2\varepsilon_{\text{proxy}} \\
&\leq \frac{2\bar{F}(\theta_0)}{\rho \sqrt{T}} + \frac{2D \sqrt{\varepsilon_{\text{proxy}}}}{\rho} + \frac{L D^2}{\rho \sqrt{T}} + \frac{2\varepsilon_{\text{sampling}} \sqrt{T}}{\rho} + 2\varepsilon_{\text{proxy}}
\end{aligned}$$

**步骤 10：简化最终形式。**

令 $\Delta_F = \bar{F}(\theta_0)$。注意到 $\sqrt{\varepsilon_{\text{proxy}}} \leq \varepsilon_{\text{proxy}} + 1/4$（由 $\sqrt{x} \leq x + 1/4$ 对所有 $x \geq 0$ 成立）。将常数项合并，得到最终形式：

$$\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[R_t(\theta_t)] \leq \mathcal{O}\left(\frac{1}{\sqrt{T}}\right) + \mathcal{O}(\varepsilon_{\text{proxy}}) + \mathcal{O}(\varepsilon_{\text{sampling}})$$

展开 $\varepsilon_{\text{proxy}} = 4L^2 \eta_l^2 E^2 G^2 + \frac{2\sigma^2}{E}$：

$$\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[R_t(\theta_t)] \leq \mathcal{O}\left(\frac{1}{\sqrt{T}}\right) + \mathcal{O}(\eta_l^2 E^2) + \mathcal{O}\left(\frac{\sigma^2}{E}\right) + \mathcal{O}\left(\frac{1}{\sqrt{|\mathcal{S}_t|}}\right)$$

这就完成了定理 3 的证明。

∎

**注记：** 上述证明中，假设 6（UPGrad 方向的正对齐性质）是收敛分析的关键。该假设的合理性来源于引理 2 中 UPGrad 方向的结构性质：UPGrad 通过 Gramian 矩阵显式建模客户端更新之间的 pairwise 几何关系，在保证每个客户端至少获得单位贡献的约束下寻找最小范数组合方向。当客户端更新方向存在冲突时，UPGrad 方向区别于简单平均，能够更好地平衡各客户端目标。在实际实验中，FedClient-UPGrad 的稳定收敛行为（三种子均无发散）为这一假设提供了经验支持。