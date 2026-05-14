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

本文要围绕五个主张展开：

1. **问题主张：** 客户端异构会导致目标冲突，只优化平均损失可能牺牲尾部客户端。
2. **方法主张：** 客户端本地更新量可以作为客户端目标梯度或下降方向的实用代理。
3. **轨迹主张：** FedClient-UPGrad 不是逼近 FedAvg 的平均风险轨迹，而是追踪一条随机客户端级 Pareto/Jacobian descent 轨迹；在强异构下，该轨迹可能更曲折，但会把尾部客户端保护累积到同通信轮次的模型切片中。
4. **理论主张：** UPGrad 可以在客户端更新空间中计算冲突感知的公共方向，并指向多目标一阶稳定性。
5. **实验主张：** FedClient-UPGrad 在 FEMNIST 上稳定改善平均性能、尾部客户端和客户端差异；在 CIFAR10 alpha=0.1 的强 label-skew 下体现为公平性-平均性能 trade-off；在 CIFAR10 alpha=0.5 的较弱异构下，FedAvg 的平均风险轨迹更强，FedClient-UPGrad 仅相对 FedAvg 略低客户端差异。这说明本文方法不是全面支配 FedAvg，而是在客户端目标冲突更突出时更有价值。

## 摘要应该写什么

摘要不是简单总结全文，而是压缩回答四个问题：

1. 本文解决什么问题？
异构联邦学习中，平均准确率无法反映尾部客户端和弱客户端的表现。

2. 本文提出什么核心想法？
把每个客户端损失看作一个目标，把联邦学习建模为客户端级多目标优化问题。

3. 本文方法是什么？
客户端上传普通 FedAvg 风格的模型更新量，服务器用这些更新量构造客户端目标方向矩阵，并用 UPGrad 计算公共更新方向。

4. 本文证据是什么？
FEMNIST 三种子实验中，FedClient-UPGrad 同时提升平均客户端准确率、worst-10% 客户端准确率，降低客户端准确率标准差和平均测试损失。CIFAR10 alpha=0.1 结果显示 FedClient-UPGrad 牺牲少量平均准确率换取更好的 worst-10% 准确率和更低客户端差异；CIFAR10 alpha=0.5 结果则显示较弱异构下 FedAvg 仍是更强平均风险基线。

摘要草稿：

```text
异构联邦学习通常以平均准确率作为主要评价指标，但在实际部署中，模型是否能服务低性能客户端同样重要。本文从客户端级多目标优化视角重新审视异构联邦学习，将每个客户端的经验损失视为一个独立目标。基于这一视角，我们提出 FedClient-UPGrad：每个被选客户端从服务器广播的全局模型出发进行普通本地训练，并上传模型更新量；服务器将负更新量作为客户端目标梯度代理，利用 UPGrad 在客户端更新空间中计算冲突感知的公共方向。该方法不需要显式计算客户端 Jacobian，通信形式与 FedAvg 风格的模型更新上传一致。我们在 FEMNIST 和 Dirichlet 划分的 CIFAR10 上比较 FedClient-UPGrad、FedAvg、qFedAvg 和 FedMGDA+。在 FEMNIST 三种子实验中，FedClient-UPGrad 获得 0.7996 ± 0.0094 的平均客户端准确率和 0.5902 ± 0.0058 的 worst-10% 客户端准确率，均显著优于各基线方法。在 CIFAR10 alpha=0.1 上，FedClient-UPGrad 获得最高 worst-10% 准确率和最低客户端准确率标准差，但平均准确率略低于 FedAvg；在 alpha=0.5 上，FedAvg 在平均准确率和 worst-10% 上均领先。这些结果表明，客户端级多目标聚合尤其适合目标冲突更突出的异构设置，并体现出平均风险与客户端公平性的可解释 trade-off。
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
3. 从优化轨迹角度解释 partial participation：每轮 sampled client-Jacobian 不是共同恢复某个固定矩阵，而是在当前 `theta_t` 上给出 full client-Jacobian field 的随机切片；这些切片经 UPGrad 聚合形成一条公平性导向的随机轨迹。
4. 构建统一实验框架，在 FEMNIST 和 CIFAR10 上公平比较 FedAvg、qFedAvg、FedMGDA+ 和 FedClient-UPGrad。
5. 在 FEMNIST 三种子实验中，FedClient-UPGrad 在平均准确率、worst-10% 准确率、客户端准确率标准差和测试损失上均稳定最优；在 CIFAR10 alpha=0.1 上，它体现为平均性能与客户端公平性的 trade-off；在 CIFAR10 alpha=0.5 上，FedAvg 的平均风险轨迹更强，形成本文方法的边界案例。
6. 分析方法的效率代价，说明其以额外服务器端聚合计算换取更好的尾部客户端性能。

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

理想情况下，如果每一轮都能访问全部客户端，则可以构造完整客户端级 Jacobian：

```text
J(theta) =
[ grad F_1(theta)
  grad F_2(theta)
  ...
  grad F_K(theta) ] in R^{K x d}
```

其中 `d` 是模型参数维度。这个矩阵的每一行对应一个客户端目标的梯度。对 `J(theta)` 做 Jacobian Descent / UPGrad 式聚合，可以直接得到平衡所有客户端目标的公共方向。

但在联邦学习中，每轮让所有客户端参与并构造完整 `K x d` client-Jacobian 通常不现实。原因包括：

1. 联邦学习本身通常是 partial participation，每轮只有部分客户端在线或被采样。
2. 完整 client-Jacobian 的通信和服务器端计算成本随客户端数 `K` 增长。
3. 真实联邦场景中客户端数据和设备状态动态变化，很难每轮获得全体客户端方向。

因此，FedClient-UPGrad 采用随机客户端采样。每轮只对参与客户端 `S_t` 构造 sampled client-Jacobian：

\[
J_{S_t}(\theta_t) = [\nabla F_i(\theta_t)]_{i \in S_t} \in \mathbb{R}^{m \times d}, \quad m = |S_t|.
\]

多轮通信中，不同客户端会被反复采样。于是，每轮的 sampled client-Jacobian 可以被看作当前 full client-Jacobian 几何结构的随机切片。这里要特别强调：多轮通信得到的一系列子 Jacobian 不是共同逼近某一个固定矩阵，而是在训练轨迹上分别对应当前模型点的完整 client-Jacobian field：

\[
J(\theta_0), J(\theta_1), \ldots, J(\theta_{T-1}).
\]

第 `t` 轮的近似关系应写成：

\[
G_t \approx J_{S_t}(\theta_t) \subset J(\theta_t).
\]

因此，FedClient-UPGrad 近似的不是 FedAvg 轨迹，也不是某个训练后的全 Jacobian，而是一条不可直接计算的 full-client Jacobian descent / UPGrad 轨迹。这个思想类似 SGD 中 mini-batch gradient 对当前 full-batch gradient 的随机近似：第 `t` 步的 mini-batch gradient 近似的是 `theta_t` 处的 full gradient，而不是一个固定初始梯度。

本文的核心理解是：

```text
FedClient-UPGrad = stochastic client-level Jacobian descent
```

即用多轮 sampled client-Jacobian 近似完整 client-Jacobian，从而逐步逼近所有客户端目标之间的平衡，而不是只优化单轮被采样客户端的短期表现。

进一步地，FedAvg 与 FedClient-UPGrad 的轨迹目标不同。FedAvg 主要追踪平均风险下降轨迹：

\[
\min_\theta \sum_{i=1}^K p_i F_i(\theta),
\]

而 FedClient-UPGrad 追踪客户端级多目标轨迹：

\[
\min_\theta [F_1(\theta), \ldots, F_K(\theta)].
\]

在强异构或极端 label-skew 下，不同轮次采样到的 `S_t` 可能具有差异很大的冲突结构，因此 FedClient-UPGrad 的随机 Pareto 轨迹可能比 FedAvg 的平均风险轨迹更曲折、更震荡。这种不平滑性不应被简单解释为方法失败，而应被解释为平均性能与客户端公平性之间的优化轨迹 trade-off：方法可能牺牲一部分 mean accuracy 或 mean loss，换取 higher worst10 和 lower client accuracy std。

论文中可以把“同通信轮次模型切片”作为解释核心。第 `t` 轮模型切片 `theta_t` 的公平性由客户端性能分布衡量：

\[
\{F_i(\theta_t)\}_{i=1}^K
\]

或：

\[
\{\operatorname{Acc}_i(\theta_t)\}_{i=1}^K.
\]

FedClient-UPGrad 的切片更公平，并不是因为任意单个 `G_t` 都完整代表全体客户端，而是因为每轮 UPGrad 对参与客户端施加一阶不冲突倾向，这种局部公平性通过状态递推累积：

\[
\text{non-conflicting sampled direction}
\rightarrow
\text{less harm to sampled clients}
\rightarrow
\text{repeated protection under random participation}
\rightarrow
\text{more balanced } \theta_t.
\]

这个解释应贯穿方法、理论和实验讨论，用来说明为什么 CIFAR10 alpha=0.1 中全局平均指标可能不如 FedAvg，但 worst10 和 std 更好。

### 3.3 本地更新作为 sampled client-Jacobian 的行向量代理

每个客户端从同一个全局模型 `theta_t` 出发，本地训练得到 `theta_i^t`，上传：

```text
Delta_i^t = theta_i^t - theta_t
```

因为本地训练是沿着客户端损失下降，所以：

```text
-Delta_i^t
```

可以被看作客户端目标梯度或下降方向的代理。因此实际算法并不显式计算：

```text
grad F_i(theta_t)
```

而是使用：

```text
g_i^t ≈ -Delta_i^t
```

作为 sampled client-Jacobian 的第 `i` 行。于是服务器每轮实际构造的是代理矩阵：

```text
G_t = [g_i^t]_{i in S_t} ≈ J_{S_t}(theta_t)
```

这个设计同时解决两个问题：一是避免显式计算完整客户端梯度/Jacobian，二是保持 FedAvg 风格的客户端通信形式，即每个客户端只上传一个模型更新量。

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

### 4.2 Sampled Client-Jacobian 作为随机轨迹近似

FedClient-UPGrad 的方法动机不是“只优化当前采样到的 50 个客户端”，也不是用多轮子 Jacobian 恢复某个固定完整 Jacobian。更准确地说，它用当前采样客户端构造一个随机子问题，在训练轨迹上持续获得当前 full client-Jacobian field 的随机切片，从而追踪一条客户端级随机 Pareto/Jacobian descent 轨迹。

若完整客户端级多目标方向需要基于：

```text
J(theta_t) in R^{K x d}
```

则实际每轮只构造：

```text
G_t ≈ J_{S_t}(theta_t) in R^{m x d}
```

其中 `S_t` 是随机采样客户端集合。每一轮的 `G_t` 只覆盖当前 `theta_t` 处的部分客户端目标，但在多轮通信中，随机采样会沿着模型轨迹提供对完整客户端集合几何结构的重复观测。因此，FedClient-UPGrad 可以被理解为一种 stochastic trajectory approximation：

```text
full client-Jacobian     J(theta_t)
sampled client-Jacobian  J_{S_t}(theta_t)
proxy sampled matrix     G_t from client deltas
```

写作时要避免过强表述。不要说“每个 sampled Jacobian 都能准确逼近完整 Jacobian”。更准确的说法是：

```text
在随机采样和多轮通信覆盖下，sampled client-Jacobian 在每个当前模型点 theta_t 上随机反映完整客户端目标集合的局部几何结构。
```

这也解释了为什么实验需要多轮通信和多随机种子：采样带来近似误差，长期训练和多 seed 用来检验这种随机轨迹是否稳定地转化为尾部客户端收益。同通信轮次的模型切片更公平，来自前面多轮 sampled protection 的状态累积，而不是来自某个单独 `G_t` 对完整 `J(theta_t)` 的精确恢复。

采样率 `participation_rate` 是这一路径中的关键控制量。设每轮参与客户端数为 `m = |S_t|`。当采样率更高时，服务器构造的 `G_t` 包含更多客户端行，因此 sampled client-Jacobian 对当前 full client-Jacobian `J(theta_t)` 的局部几何覆盖更充分，采样方差通常更小，UPGrad 方向也更接近理想 full-client UPGrad 方向。代价是每轮本地训练客户端更多，上传总量更大，服务器端需要处理更大的 `m x d` 更新矩阵和 `m x m` Gramian，聚合计算更慢。

相反，当采样率更低时，`G_t` 更小，通信、客户端训练和服务器聚合都更便宜；但每轮只看到更少客户端目标，随机切片对 full-client 轨迹的近似更粗糙，方向方差更大，可能削弱尾部客户端收益并降低最终训练指标。论文中应把它写成采样率带来的三方 trade-off：

```text
participation rate ↑  ->  better client-Jacobian coverage / lower sampling error  ->  potentially better fairness and stability, but higher per-round cost
participation rate ↓  ->  cheaper rounds / smaller Gramian  ->  noisier trajectory approximation and potentially weaker metrics
```

注意不要过强声称“采样率越高效果必然越好”。更严谨的说法是：在其他超参数固定时，提高采样率通常会改善 sampled Jacobian 对 full-client geometry 的代表性，但实际指标还受异构强度、学习率、本地 epoch、模型容量和 UPGrad 数值稳定性影响。

### 4.3 UPGrad 聚合

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

### 4.4 与基线方法的区别

1. **FedAvg：** 对客户端更新按样本数加权平均。
2. **qFedAvg：** 根据客户端训练前损失调整权重。
3. **FedMGDA+：** 用 MGDA/min-norm 思路聚合客户端更新代理。
4. **FedClient-UPGrad：** 用 UPGrad 在客户端更新空间中计算公共方向。

### 4.5 通信和计算代价

通信方面：

```text
FedClient-UPGrad 和 FedAvg 类似，每个客户端上传一个模型 delta。
```

额外代价在服务器端：

```text
需要构造 m x d 更新矩阵，并求解 UPGrad 聚合方向。
```

FEMNIST 结果中：

1. FedAvg 平均每轮 `1.012s`。
2. qFedAvg 平均每轮 `1.067s`。
3. FedMGDA+ 平均每轮 `1.103s`。
4. FedClient-UPGrad 平均每轮 `1.480s`。

论文中要正面承认这个代价，并说明它换来了明显的 tail-client performance 提升。

采样率会放大或缓解这一代价。FedAvg 的服务器端聚合几乎只是对客户端 delta 求平均，服务器端成本随 `m` 增长较温和；FedClient-UPGrad 需要显式构造客户端更新矩阵并计算 Gramian：

\[
H_t = G_t G_t^\top \in \mathbb{R}^{m \times m}.
\]

因此，当 `participation_rate` 增大时，FedClient-UPGrad 的服务器端聚合时间会比 FedAvg 更敏感。论文中可以把采样率视为一个系统层面的旋钮：较高采样率提高 sampled Jacobian 质量和潜在公平性收益，较低采样率降低每轮参与客户端数、通信量和服务器端计算，但会增加轨迹近似噪声。

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

主实验固定 `participation_rate = 0.5`，是为了在轨迹近似质量和每轮计算成本之间取一个中间点。采样率本身是一个重要消融维度：更高采样率可能降低 sampled Jacobian 误差、提高尾部客户端和稳定性指标，但会增加每轮客户端训练、上传和服务器端 Gramian/UPGrad 成本；更低采样率更快，但可能使随机 Pareto 轨迹更噪声化。由于本文重点是验证客户端级 UPGrad 聚合思想，采样率消融可作为附录或后续实验。

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
5. 优化轨迹/模型切片示意图：画出 FedAvg 的平滑平均风险轨迹和 FedClient-UPGrad 的更曲折随机 Pareto 轨迹，并在同一通信轮次 `theta_t` 上标注客户端性能分布切片，直观展示 mean/loss 与 worst10/std 的 trade-off。
6. 采样率消融图：横轴为 `participation_rate`，同时画 worst10/std/mean loss 和 avg round time，用来展示 sampled Jacobian 覆盖质量与计算效率之间的 trade-off。

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

可以加入更有理论味道的轨迹解释：

```text
FedAvg 在该设置下沿着平均客户端目标的随机梯度轨迹前进，因此更容易获得较好的 mean accuracy 和 mean loss。FedClient-UPGrad 则沿着客户端级随机 Pareto/Jacobian 轨迹前进；每轮 sampled client-Jacobian 的冲突结构随参与客户端集合变化，因此轨迹可能更曲折，但该轨迹在每个模型切片上累积了对参与客户端的一阶不伤害倾向。结果上，这种轨迹没有最小化平均风险，却产生了更公平的同轮次模型切片，即 worst-10% accuracy 更高且 client accuracy std 更低。
```

图解提醒：扩写论文时建议画一张二维直观图。横轴可以表示平均风险下降方向，纵轴表示客户端公平性或尾部客户端改善方向；用较平滑曲线表示 FedAvg，用更曲折曲线表示 FedClient-UPGrad；在同一个通信轮次画两个模型切片的客户端准确率分布，展示 FedAvg 的平均点较优但尾部较低，FedClient-UPGrad 的尾部更高、分布更窄。

### 6.4 CIFAR10 alpha=0.5 结果

结果目录：

```text
results/full_cifar_alpha0p5_E2R1000_pr50_mixed
```

该设置仍使用 mixed hyperparameters：FedAvg、qFedAvg 和 FedMGDA+ 使用 R100 tuning 选出的最佳参数，FedClient-UPGrad 使用经过长轮次稳定性验证的 `learning_rate=0.01, update_scale=1.0`。

三种子结果：

| 方法 | Mean Acc | Worst10 Acc | Acc Std | Mean Loss |
|---|---:|---:|---:|---:|
| FedAvg | **0.6379 ± 0.0105** | **0.5075 ± 0.0273** | 0.0705 ± 0.0056 | **1.0747 ± 0.0461** |
| qFedAvg | 0.6022 ± 0.0041 | 0.4735 ± 0.0185 | **0.0674 ± 0.0060** | 1.1372 ± 0.0156 |
| FedMGDA+ | 0.1008 ± 0.0030 | 0.0000 ± 0.0000 | 0.1154 ± 0.0101 | NaN |
| FedClient-UPGrad | 0.5999 ± 0.0025 | 0.4704 ± 0.0043 | 0.0681 ± 0.0039 | 1.1501 ± 0.0119 |

解释：

1. FedAvg 在 CIFAR10 alpha=0.5 上全面领先 mean accuracy、worst10 accuracy 和 mean loss。
2. FedClient-UPGrad 相比 FedAvg：mean accuracy 低 `0.0380`（相对 `-5.95%`），worst10 低 `0.0371`（相对 `-7.31%`），mean loss 高 `0.0754`（相对 `+7.02%`）。
3. FedClient-UPGrad 的客户端准确率标准差略低于 FedAvg（`0.0681` vs `0.0705`），但不是最低；qFedAvg 的 std 最低。
4. FedMGDA+ 三个 seed 均出现严重退化，mean accuracy 接近随机水平，mean loss 为 `NaN`，说明该配置在 CIFAR10 alpha=0.5 长轮次训练中不稳定。
5. alpha=0.5 是本文方法的边界案例：当 label-skew 较弱、平均风险轨迹已经能兼顾多数客户端时，FedAvg 更有效；FedClient-UPGrad 的公平性导向轨迹没有带来 worst10 收益。

逐 seed 现象：

| Seed | Mean Acc 最好 | Worst10 最好 | Acc Std 最低 | Mean Loss 最低 |
|---:|---|---|---|---|
| 7 | FedAvg | FedAvg | qFedAvg | FedAvg |
| 42 | FedAvg | FedAvg | qFedAvg | FedAvg |
| 123 | FedAvg | FedAvg | FedClient-UPGrad | FedAvg |

客户端准确率分位数（三种子合并）：

| 方法 | P10 | P25 | Median | P75 | P90 |
|---|---:|---:|---:|---:|---:|
| FedAvg | **0.5571** | **0.5938** | **0.6405** | **0.6838** | **0.7223** |
| qFedAvg | 0.5205 | 0.5647 | 0.6045 | 0.6436 | 0.6813 |
| FedMGDA+ | 0.0000 | 0.0109 | 0.0606 | 0.1468 | 0.2679 |
| FedClient-UPGrad | 0.5118 | 0.5597 | 0.6042 | 0.6472 | 0.6809 |

可直接写进论文：

```text
In the milder CIFAR10 alpha=0.5 setting, FedAvg remains the strongest method in terms of mean accuracy, worst-10% accuracy, and mean loss. FedClient-UPGrad slightly reduces client accuracy variance relative to FedAvg, but this does not translate into tail-client improvement. This boundary case suggests that client-level UPGrad is most useful when client objectives exhibit stronger conflict; when heterogeneity is moderate, the smoother average-risk trajectory of FedAvg can be preferable.
```

中文解释：

```text
CIFAR10 alpha=0.5 表明，FedClient-UPGrad 并不是对 FedAvg 的全面替代。在较弱 label-skew 下，FedAvg 的平均风险轨迹已经足够稳定且有效，因此在 mean、worst10 和 loss 上均领先。这个结果反而强化了本文的边界条件：FedClient-UPGrad 的价值主要出现在客户端目标冲突更强、平均优化容易牺牲尾部客户端的场景。
```

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

CIFAR10 alpha=0.5 效率：

| 方法 | 平均每轮时间 | 平均聚合计算时间 | 总训练时间 |
|---|---:|---:|---:|
| FedAvg | 2.806s ± 0.041s | 0.0003s | 2806.67s ± 41.12s |
| qFedAvg | 4.074s ± 0.028s | 0.0001s | 4073.91s ± 28.47s |
| FedMGDA+ | 4.093s ± 0.052s | 0.0297s | 4093.28s ± 52.45s |
| FedClient-UPGrad | 3.296s ± 0.025s | 0.4784s | 3297.12s ± 24.79s |

解释：alpha=0.5 上 FedClient-UPGrad 比 FedAvg 慢约 `17.46%`，但没有带来 mean/worst10 收益，因此该设置下计算-公平性 trade-off 不划算。这个结果应作为局限和边界案例报告，而不是隐藏。

### 6.6 跨数据集综合解读

三组主结果形成一条更稳健、但不夸大的结论链：

1. **FEMNIST 是强正结果。** writer partition 是自然客户端异构，FedClient-UPGrad 在 mean、worst10、std 和 loss 上全面优于基线，说明客户端级多目标聚合在真实客户端异构场景下有效。
2. **CIFAR10 alpha=0.1 是公平性 trade-off 结果。** FedClient-UPGrad 不如 FedAvg 的 mean/loss，但获得最高 worst10 和最低 std，说明强 label-skew 下方法更偏向切片公平性而不是平均风险最优。
3. **CIFAR10 alpha=0.5 是边界案例。** FedAvg 全面领先 mean、worst10 和 loss，说明在较弱 label-skew 下，平均风险轨迹可能已经足够好，UPGrad 的冲突感知方向不一定带来收益。

论文主结论应据此写成：FedClient-UPGrad 是一种面向客户端目标冲突和尾部客户端改善的联邦聚合方法，而不是所有异构强度下都支配 FedAvg 的通用替代品。

## 7. Theory and Analysis

理论部分必须和实验形成一条论证链：先证明方法为什么合理，再说明实验为什么看这些指标。

### 7.1 理论要证明什么

理论部分应支持四个递进命题：

1. **Lemma 1：Local Delta Proxy**
客户端本地更新量可以作为该客户端目标梯度或下降方向的近似代理。

2. **Lemma 2：UPGrad Common Direction**
UPGrad 可以把这些客户端方向代理组合成一个冲突感知的公共方向。

3. **Lemma 3：Trajectory Slice Fairness Inheritance**
每轮 sampled UPGrad 方向对参与客户端具有一阶不冲突或少伤害倾向；在随机参与覆盖下，这种局部性质通过 `theta_{t+1} = theta_t - alpha_t d_t` 的状态递推累积，使同通信轮次的模型切片更可能呈现更均衡的客户端性能分布。

4. **Theorem 4：Approximate Pareto-Stationarity**
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

如果时间有限，理论优先级应为：Lemma 1 完整证明 > Lemma 2 性质说明 > Lemma 3 给出一阶切片公平性解释 > Theorem 4 给出较保守但自洽的证明草图。不要为了显得理论强而写无法支撑的全局收敛。

### 7.1.1 建议使用的统一符号

正式论文理论部分建议先定义以下符号：

```text
K                         客户端总数
D_i                       客户端 i 的本地数据集或数据分布
F_i(theta)                客户端 i 的经验风险 / 期望风险
F(theta)                  客户端级向量目标 [F_1(theta), ..., F_K(theta)]
J_full(theta)             完整 client-Jacobian，行向量为所有客户端梯度
S_t                       第 t 轮采样到的客户端集合
m = |S_t|                 每轮参与客户端数量
J_{S_t}(theta)            sampled client-Jacobian，只包含 S_t 中客户端梯度
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
R_{S_t}(theta)            sampled Pareto-stationarity residual
Q(theta)                  客户端切片公平性指标，例如 worst10、client gap 或 client accuracy variance
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

5. **客户端采样有界方差：** 每轮采样集合 `S_t` 是随机采样，sampled client-Jacobian 对完整 client-Jacobian 的几何结构提供有界方差的随机近似。

采样误差应显式依赖每轮参与客户端数 `m`。在均匀无放回采样且客户端梯度二阶矩有界的理想化设置下，sampled client-Jacobian 或 sampled residual 相对 full-client 对象的方差通常随 `m` 增大而下降，形式上可以写成：

```text
E || sampling_error_t ||^2 <= C_sample / m
```

或更保守地写成某个随 `m` 单调下降的项 `epsilon_sample(m)`。这里不必承诺严格 `1/m` 速率，除非正式证明中能满足相应独立性和有界方差条件。

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

### 7.4 Lemma 3：Trajectory Slice Fairness Inheritance

目的：把“每轮局部不冲突方向”与“同通信轮次模型切片更公平”联系起来。这一节是本文理论叙事中最有辨识度的部分：FedClient-UPGrad 不是在每轮恢复完整 Jacobian，也不是在逼近 FedAvg 轨迹，而是在一条随机 Pareto 轨迹上累积局部公平性。

第 `t` 轮更新为：

\[
\theta_{t+1} = \theta_t - \alpha_t d_t.
\]

如果 `d_t` 对参与客户端 `i in S_t` 满足一阶非冲突条件：

\[
\langle \nabla F_i(\theta_t), d_t \rangle \ge -\epsilon_{i,t},
\]

则由 smoothness 得到：

\[
F_i(\theta_{t+1})
\le
F_i(\theta_t)
- \alpha_t \langle \nabla F_i(\theta_t), d_t \rangle
+ \frac{L\alpha_t^2}{2}\|d_t\|^2.
\]

当 `epsilon_{i,t}` 和二阶项较小时，该更新在一阶近似下不会显著伤害当前参与客户端。对比 FedAvg，平均方向可能对某些客户端满足：

\[
\langle \nabla F_j(\theta_t), d_t^{\mathrm{FedAvg}} \rangle < 0,
\]

即平均目标下降时仍可能增加客户端 `j` 的 loss。

因此可以提出一个保守 lemma：

```text
Lemma. 在客户端均匀随机参与、UPGrad 方向满足 sampled non-conflict up to error、且步长足够小时，每个客户端在其参与轮次中获得一阶保护。若每个客户端在 T 轮内以正概率重复参与，则这种 sampled protection 沿优化轨迹累积，使模型序列的客户端损失分布相对于平均方向更新具有更小的尾部恶化项。
```

这不是要证明 FedClient-UPGrad 永远支配 FedAvg，而是要证明一种更弱、更合理的性质：

\[
\text{sampled first-order protection}
\Rightarrow
\text{trajectory-level bias toward client balance}.
\]

可以定义一个客户端切片公平性泛函 `Q(theta)`，例如：

\[
Q_{\mathrm{gap}}(\theta) = \max_i F_i(\theta) - \frac{1}{K}\sum_{i=1}^K F_i(\theta),
\]

或用实验中的 accuracy 版本：

\[
Q_{\mathrm{tail}}(\theta) = \operatorname{Worst10Acc}(\theta),
\]

\[
Q_{\mathrm{std}}(\theta) = \operatorname{Std}(\{\operatorname{Acc}_i(\theta)\}_{i=1}^K).
\]

理论上可以不直接证明 worst10 accuracy，而是证明 loss-side tail gap 或 client loss variance 的下降倾向；实验中再用 worst10 accuracy、client accuracy std 和 sorted client accuracy curve 对应验证。

证明路线：

1. 对任意参与客户端使用 smoothness descent lemma。
2. 用 UPGrad non-conflict / positive alignment 条件控制一阶项。
3. 对客户端随机参与事件取期望，得到每个客户端被保护的概率项。
4. 把每轮保护项沿 `t=0,...,T-1` 累积，得到 trajectory-level bound。
5. 将 loss-side bound 映射到实验中的同轮次模型切片指标：worst10、std、CDF。

论文中的谨慎表述：

```text
The fairer checkpoints are inherited from the trajectory rather than from any single sampled sub-Jacobian. Each sampled UPGrad step reduces conflict for the participating clients in a first-order sense. Under repeated random participation, this local protection accumulates along the model trajectory, yielding checkpoints with a more balanced client performance distribution.
```

这部分也解释 CIFAR10 alpha=0.1 的现象：当 label-skew 极强时，`J_{S_t}(theta_t)` 的几何结构在轮次间变化很大，UPGrad 轨迹可能比 FedAvg 更曲折；但这种曲折来自其追踪公平性导向的随机 Pareto 轨迹，而不是追踪平均风险轨迹。因此 mean/loss 可能不如 FedAvg，而 worst10/std 更优。

### 7.5 Theorem 4：Approximate Pareto-Stationarity

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
3. Lemma 3 说明 sampled protection 会沿训练轨迹累积到同通信轮次模型切片。
4. Theorem 4 说明重复使用该方向更新，可以趋近 approximate Pareto-stationarity，误差来自 proxy 和 sampling。

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
这个误差应写成依赖采样率或参与客户端数的项，例如 `O(epsilon_sample(m))`。直观上，`m` 越大，sampled client-Jacobian 对 full-client geometry 的覆盖越充分；`m` 越小，轨迹近似噪声越大。
4. **UPGrad 求解/对齐误差：** 来自 UPGrad 方向相对于理想 Pareto common direction 的偏差。

建议 theorem 表述为：

```text
Theorem. 在 Assumption 1-6 成立时，选择合适的 local learning rate eta_l 和 server step size alpha_t，则 FedClient-UPGrad 生成的模型序列满足：

1/T sum_{t=0}^{T-1} E[R(theta_t)]
<= O(1/sqrt(T))
 + O(sigma^2/E)
  + O(L^2 eta_l^2 E^2 G^2)
  + O(epsilon_sample(m))
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

### 7.6 理论和实验的有机联系

理论和实验不是两部分，而是同一条证据链。

| 理论命题 | 含义 | 实验对应 | 指标/图 |
|---|---|---|---|
| Local Delta Proxy | 客户端 delta 含有客户端目标方向信息 | FedClient-UPGrad 只上传 FedAvg-style delta | 通信量与 FedAvg-style 方法一致 |
| UPGrad Common Direction | 聚合应处理客户端方向冲突 | 与 FedAvg、FedMGDA+ 比较 | mean accuracy、worst10 accuracy、sorted client curve |
| Trajectory Slice Fairness | 局部 sampled protection 沿训练轨迹累积到模型切片 | 同通信轮次 checkpoint 的客户端分布比较 | trajectory sketch、same-round client distribution、worst10/std |
| Approximate Pareto-Stationarity | 方法目标是多客户端目标平衡 | 看尾部客户端和客户端分布是否改善 | worst10 accuracy、accuracy std、CDF/boxplot |
| proxy/sampling error | 随机性和采样会带来误差 | 三种子实验 | mean ± std、error bar |
| participation-rate trade-off | 采样率控制 Jacobian 切片质量和每轮成本 | 不同参与率的消融实验 | participation-rate vs metrics/time |
| common direction cost | 结构化聚合有额外计算 | 报告每轮时间和聚合时间 | efficiency-performance plot |

这就是理论和实验的有机关联：

1. 理论说客户端 delta 是目标方向代理，所以实验强调方法不需要额外 Jacobian 通信。
2. 理论说 UPGrad 处理客户端冲突，所以实验必须看 worst10 和客户端分布，而不能只看 mean accuracy。
3. 理论说同通信轮次模型切片的公平性来自轨迹继承，所以实验应比较相同 round budget 下的客户端性能分布，而不是只比较最终平均准确率。
4. 理论说收敛目标是 approximate Pareto-stationarity，所以实验重点是客户端间表现是否更均衡。
5. 理论中有 sampling/proxy error，所以实验必须多 seed。
6. 理论中采样误差依赖参与客户端数，所以 participation rate 是重要消融维度。
7. 理论中 UPGrad 有额外计算，所以实验必须报告效率 trade-off。

可直接写进论文的桥接句：

```text
理论分析表明，基于客户端更新几何关系计算的公共方向应比简单平均更能平衡客户端目标。因此，我们不仅报告平均客户端准确率，还报告 worst-10% 客户端准确率和客户端准确率标准差。在 FEMNIST 上，FedClient-UPGrad 相比 FedAvg 将 worst-10% 准确率提升 9.99 个百分点，同时提高平均准确率，说明更新空间中的公共方向确实改善了尾部客户端而没有牺牲整体性能。
```

CIFAR10 alpha=0.1 的桥接句要更强调 trade-off：

```text
在更极端的 Dirichlet label-skew 下，FedClient-UPGrad 的优势主要表现为切片公平性而非平均风险最优。FedAvg 的平均风险轨迹更平滑，因此取得更好的 mean accuracy 和 mean loss；FedClient-UPGrad 的随机 Pareto 轨迹更受 sampled client-Jacobian 冲突结构影响，可能更曲折，但其 repeated sampled protection 累积到最终 checkpoint，使 worst-10% accuracy 更高且 client accuracy std 更低。
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
3. 两个 alpha 共同说明 FedClient-UPGrad 的适用范围：强冲突下更容易体现尾部客户端收益，较弱异构下 FedAvg 可能仍是更优选择。

alpha=0.1 的结果应作为“公平性-平均性能 trade-off”的正面证据，而不是负面结果。它说明：

1. 当客户端目标冲突极强时，FedAvg 的平均风险轨迹可能更适合优化 mean accuracy 和 mean loss。
2. FedClient-UPGrad 的随机 Pareto 轨迹可能更曲折，因为每轮 sampled client-Jacobian 的几何结构变化更大。
3. 这种更曲折的轨迹仍然有意义，因为同通信轮次的模型切片呈现更好的 worst10 和更低 std。
4. 因此论文应避免声称 FedClient-UPGrad 全面支配 FedAvg，而应强调它在极端异构下提供尾部客户端改善。

alpha=0.5 的结果应作为边界案例。它说明：

1. 当 label-skew 较弱时，客户端目标冲突不足以让 UPGrad 的公平性导向方向带来 tail improvement。
2. FedAvg 的平均风险轨迹更平滑且更高效，在 mean、worst10 和 loss 上均领先。
3. FedClient-UPGrad 只略微降低客户端准确率标准差，但没有获得 worst10 收益，因此该设置下不应声称方法有效。
4. 这要求论文把 CIFAR10 结论写成“异构强度相关的 trade-off”，而不是“跨所有 label-skew 设置全面有效”。

### 8.4 优化轨迹与同轮次切片解释

这一节建议作为 Discussion 的理论解释亮点。

核心观点：

```text
FedAvg and FedClient-UPGrad should not be compared as two approximations to the same trajectory. FedAvg follows an average-risk trajectory, while FedClient-UPGrad follows a stochastic client-level Pareto trajectory. Therefore, a FedClient-UPGrad checkpoint may have slightly worse mean loss but better client distribution at the same communication round.
```

中文表述：

```text
FedClient-UPGrad 的目标不是复制 FedAvg 的平滑平均风险轨迹，而是在 partial participation 下用 sampled client-Jacobian 构造一条随机客户端级 Pareto 轨迹。由于每轮参与客户端集合不同，这条轨迹在强异构场景中可能更曲折；但每一步都倾向于减少对当前参与客户端的方向冲突。随着客户端反复随机参与，这种局部保护被写入模型状态，使同通信轮次的模型切片呈现更好的尾部客户端性能和更低的客户端间差异。
```

这段讨论应放在 Results 后、Limitations 前，用来解释为什么“全局平均指标不总是最好”并不削弱论文主线。

### 8.5 未来方向：平滑随机 Pareto 轨迹

本文不把轨迹平滑作为当前贡献，也不新增相关实验。它可以作为后续工作提出：如果 FedClient-UPGrad 在强异构下因为 sampled client-Jacobian 方差较大而轨迹更曲折，那么一个自然问题是能否在不明显损失尾部客户端公平性和通信效率的前提下平滑该随机 Pareto 轨迹，从而追回甚至提升 mean accuracy 和 mean loss。

可以列出的后续方向：

1. **FedAvg-UPGrad 混合方向：** 将平均风险方向和 UPGrad 公平性方向做凸组合，在 mean performance 和 tail fairness 之间自适应折中。
2. **服务器端方向动量：** 对 `d_t = UPGrad(G_t)` 做指数滑动平均，降低跨轮次 sampled Jacobian 变化带来的方向方差。
3. **冲突触发式 UPGrad：** 当 sampled clients 之间冲突弱时使用 FedAvg，当冲突强时启用 UPGrad，或用连续 gate 控制两者权重。
4. **冲突自适应步长：** 根据 `G_t` 的负余弦相似度、Gramian 条件数或方向方差调节 server update scale，使高冲突轮次更保守。
5. **采样率自适应：** 根据训练阶段、冲突强度或尾部客户端指标动态调整 participation rate；早期或高冲突阶段使用更高采样率改善 Jacobian 切片质量，稳定阶段降低采样率节省计算。

论文中可直接写成：

```text
An interesting future direction is to smooth the stochastic Pareto trajectory induced by partial participation. Server-side momentum, FedAvg-UPGrad interpolation, conflict-triggered aggregation, conflict-adaptive step sizes, or adaptive participation rates may reduce the variance of sampled UPGrad directions. Such extensions could improve mean accuracy under severe heterogeneity while preserving the tail-client gains observed in this work. We leave this bias-variance-fairness-efficiency trade-off to future research.
```

注意：这部分只作为 Discussion/Future Work，不要写成本文已经验证的结论。

## 9. Limitations

### 9.1 运行时间和服务器端开销

FedClient-UPGrad 比 FedAvg 慢，因为服务器端要求解 UPGrad 方向。FEMNIST 上平均每轮 `1.480s`，FedAvg 为 `1.012s`。

推荐表述：

```text
FedClient-UPGrad 用额外服务器端计算换取更好的尾部客户端性能。对于关注公平性和弱客户端可靠性的场景，这一 trade-off 是有意义的；但在服务器计算资源极其受限或每轮采样客户端数很大的场景中，该代价需要进一步优化。
```

### 9.2 客户端更新只是近似梯度

多步本地训练下，delta 不是精确梯度，而是 trajectory-integrated proxy。理论和实验都应承认这一点。

### 9.3 极端异构下轨迹更曲折

在强 label-skew 或客户端目标冲突极强时，FedClient-UPGrad 的 sampled Pareto 轨迹可能比 FedAvg 更震荡。这可能导致 mean accuracy 或 mean loss 不如 FedAvg。论文中应把它表述为方法的已知 trade-off：FedClient-UPGrad 优先改善客户端级公平性和尾部表现，而不是保证平均风险指标总是最优。

### 9.4 数据集范围

本文主要证据来自视觉任务。未来需要扩展到更大模型、更大规模客户端和更多模态。

### 9.5 代码实现成熟度

当前是研究型 PyTorch 实现，基线为统一框架下复现。论文中可以说明为了公平比较，所有方法使用同一数据划分、模型、训练轮数和评估协议。

## 10. Conclusion

结论应强调：

1. 异构联邦学习不能只看平均目标。
2. 客户端级多目标视角能更好解释和处理 tail-client 问题。
3. FedClient-UPGrad 用标准客户端更新实现了实用的多目标聚合。
4. FEMNIST 结果显示该方法稳定提升平均性能、尾部性能并降低客户端差异。
5. CIFAR10 结果显示该方法的收益依赖异构强度：alpha=0.1 支持公平性 trade-off，alpha=0.5 则显示 FedAvg 在较弱异构下更强。
6. 后续工作应降低服务器端聚合开销、扩展到更大规模场景，并研究如何平滑 partial participation 下的随机 Pareto 轨迹。

## 推荐图表

主文：

1. FEMNIST、CIFAR10 alpha=0.1、CIFAR10 alpha=0.5 三组 `mean ± std` 主结果表。
2. Mean Acc vs Worst10 Acc 散点图。
3. Sorted Client Accuracy 曲线。
4. Worst10 Accuracy 柱状图，带 error bar。
5. Efficiency-Performance trade-off 图。
6. 优化轨迹与切片公平性示意图：FedAvg 平均风险轨迹 vs FedClient-UPGrad 随机 Pareto 轨迹；在相同通信轮次标出客户端性能分布切片，突出 FedClient-UPGrad 可能 mean/loss 稍弱但 worst10/std 更优。
7. Participation-rate ablation 图：展示采样率升高时 sampled Jacobian 质量、worst10/std/mean loss 和每轮时间之间的关系。

附录：

1. CIFAR10 R100 tuning 表。
2. learning rate / update scale 敏感性分析。
3. participation rate 消融表或图。
4. 每个 seed 的完整结果。
5. 运行优化说明。

## 当前证据状态

已完成且支持本文方法：

1. FEMNIST 三种子完整实验强支持 FedClient-UPGrad。
2. CIFAR10 alpha=0.1 三种子 mixed 参数完整实验支持 FedClient-UPGrad 作为尾部客户端/公平性改进方法：worst-10% 准确率最高、客户端准确率标准差最低，但平均准确率不是最高。
3. CIFAR10 alpha=0.5 三种子 mixed 参数完整实验显示 FedAvg 在较弱 label-skew 下全面领先 mean、worst10 和 loss；FedClient-UPGrad 仅略低客户端准确率 std。这是本文方法的边界案例。
4. CIFAR10 R100 调参为 baseline 提供了强参数，同时说明短轮次调参需要配合长轮次稳定性检查，尤其是 FedClient-UPGrad 和 FedMGDA+。
5. 旧 CIFAR10 seed7 R1000 的 `lr=0.01` 结果已经显示 FedClient-UPGrad 稳定且有优势，因此用于确定正式 full-run 的稳定配置。

待完成：

1. 跨数据集总表和最终论文图。
2. 若时间允许，补一个 participation-rate 消融，用来验证采样率对轨迹近似质量、最终指标和效率的影响。
3. 若需要更强自然客户端证据，可继续评估 Sent140 或 CelebA identity，但不应分散当前主线。

## 写作策略

建议重点：

1. 以客户端级多目标问题和尾部客户端性能作为主线。
2. 把 FedClient-UPGrad 写成实用 update-space 方法，而不是依赖显式 Jacobian 的重型理论方法。
3. FEMNIST 是当前最强证据，因为 writer partition 天然对应客户端异构。
4. CIFAR10 用来展示 label-skew 异构下的适用边界：alpha=0.1 支持公平性 trade-off，alpha=0.5 表明弱异构下 FedAvg 更优。
5. 正面承认运行时间代价，并把它解释成计算-公平性-采样率 trade-off。

避免过度声称：

1. 不声称非凸全局最优收敛。
2. 不声称通信量低于 FedAvg。
3. 不声称所有数据集上全面支配；当前 CIFAR10 alpha=0.1 支持“公平性-平均性能 trade-off”，alpha=0.5 明确显示 FedAvg 更强。
4. 不引入与主线无关的数据集结果。
