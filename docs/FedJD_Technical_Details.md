# FedJD 联邦多目标优化架构：技术白皮书

## 1. 核心设计理念

FedJD（Federated Jacobian Descent）是一个**服务器中心化**的联邦多目标优化框架，其核心思想是：在联邦学习场景中，当多个目标函数之间存在冲突时，客户端上传完整的Jacobian矩阵（而非仅上传标量损失或单一梯度），使服务器能够在参数空间中找到Pareto最优的下降方向。

### 1.1 设计动机

传统联邦学习方法（如FedAvg）本质上处理单目标优化。在多目标场景下，常见的做法是将多个目标加权求和为标量，但这存在两个根本问题：

1. **权重选择困难**：不同权重产生不同的Pareto最优解，预先确定权重需要领域知识
2. **负迁移风险**：当目标梯度方向冲突时，简单加权可能导致所有目标都无法有效优化

FedJD通过上传Jacobian矩阵，让服务器端看到每个目标对每个参数的独立梯度，从而使用MGDA（Multiple Gradient Descent Algorithm）类的算法找到**最小范数方向**——即在所有目标梯度凸组合中范数最小的方向，保证该方向是Pareto驻点方向。

### 1.2 与现有联邦学习架构的差异化特点

| 特性 | FedAvg | FedProx | MOO-Single | FedJD |
|------|--------|---------|------------|-------|
| 上传内容 | 模型参数/梯度 | 模型参数+近端项 | 加权梯度 | **完整Jacobian矩阵** |
| 多目标处理 | 加权求和 | 加权求和 | 加权求和 | **MinNorm方向** |
| Pareto最优性 | 无保证 | 无保证 | 无保证 | **理论保证** |
| 通信开销/客户端 | O(d) | O(d) | O(d) | **O(m×d)** |
| 服务器计算 | 平均 | 平均+近端 | 加权平均 | **MinNorm优化** |

---

## 2. 整体框架结构

```
┌─────────────────────────────────────────────────────────────────┐
│                         FedJDTrainer                            │
│  (训练循环编排：轮次管理、日志、检查点、CSV输出)                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ 每轮调用 server.run_round()
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         FedJDServer                             │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐    │
│  │ 客户端采样 │→│ Jacobian聚合  │→│ MinNorm方向寻找         │    │
│  │ (随机比例) │  │ (样本加权平均) │  │ (投影梯度下降到单纯形)  │    │
│  └──────────┘  └──────────────┘  └────────────────────────┘    │
│                           │                                     │
│  ┌────────────────────────▼────────────────────────────────┐    │
│  │ 压缩/解压 (float16, TopK, LowRank, Sketch)              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                     │
│  ┌────────────────────────▼────────────────────────────────┐    │
│  │ 模型更新: θ ← θ - η·d                                   │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ 广播模型，接收Jacobian
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FedJDClient (×K)                          │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ compute_jacobian():                                    │      │
│  │   1. 前向传播 → predictions                            │      │
│  │   2. 对每个目标 j:                                      │      │
│  │      - 反向传播 → ∇f_j(θ)                              │      │
│  │      - 拼接为行向量                                     │      │
│  │   3. 堆叠为 Jacobian J = [∇f_1; ∇f_2; ...; ∇f_m]     │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 各组件功能与交互流程

### 3.1 FedJDClient — Jacobian计算引擎

客户端的核心职责是计算Jacobian矩阵。对于m个目标函数和d维参数空间，Jacobian矩阵的形状为[m, d]。

```python
# core/client.py — compute_jacobian 核心逻辑
def compute_jacobian(self, model, objective_fn):
    loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    batch_inputs, batch_targets = next(iter(loader))
    
    model.zero_grad(set_to_none=True)
    predictions = model(batch_inputs)
    objective_values = objective_fn(predictions, batch_targets, batch_inputs)
    
    rows = []
    for index, objective in enumerate(objective_values):
        retain_graph = index < len(objective_values) - 1
        model.zero_grad(set_to_none=True)
        objective.backward(retain_graph=retain_graph)
        rows.append(flatten_gradients(model.parameters()))
    
    jacobian = torch.stack(rows, dim=0)  # shape: [m, d]
    return ClientResult(client_id=self.client_id, jacobian=jacobian, ...)
```

**关键设计决策**：
- 使用`retain_graph=True`（除最后一个目标外），避免重复前向传播
- 每个目标反向传播前调用`zero_grad(set_to_none=True)`，确保梯度独立
- 使用mini-batch（而非全数据集）计算Jacobian，平衡计算开销和梯度质量

### 3.2 FedJDServer — 中心化聚合与方向寻找

服务器端的核心流程分为5步：

```python
# core/server.py — run_round 核心逻辑（简化版）
def run_round(self, round_idx):
    # Step 1: 采样客户端
    sampled_clients = self.sample_clients()
    
    # Step 2: 收集并聚合Jacobian
    for client in sampled_clients:
        result = client.compute_jacobian(self._clone_model(), self.objective_fn)
        weight = result.num_examples / total_examples
        compressed_jac, meta = self.compressor.compress(result.jacobian)
        decompressed_jac = self.compressor.decompress(compressed_jac, meta)
        weighted_jac = decompressed_jac * weight
        aggregated_jacobian += weighted_jac
    
    # Step 3: MinNorm方向寻找
    direction = self.aggregator(aggregated_jacobian)
    
    # Step 4: 模型更新
    current_flat = flatten_parameters(self.model.parameters())
    current_flat = current_flat - self.learning_rate * direction
    assign_flat_parameters(self.model.parameters(), current_flat)
    
    # Step 5: 评估全局目标
    objective_values = self.evaluate_global_objectives()
```

**关键设计决策**：
- 服务器使用`_clone_model()`为每个客户端创建模型副本，避免并发修改
- 聚合使用样本数加权平均，与FedAvg一致
- 支持压缩/解压管线，在聚合前解压，保证聚合精度
- 非同步轮次复用上一轮方向，减少通信

### 3.3 MinNormAggregator — Pareto方向寻找器

这是FedJD的核心算法组件，实现MGDA-style的最小范数方向寻找：

```python
# aggregators/__init__.py — MinNormAggregator 核心逻辑
class MinNormAggregator(JacobianAggregator):
    def __call__(self, jacobian):
        # jacobian shape: [m, d]
        # 计算 Gram 矩阵 G = J @ J^T, shape: [m, m]
        gramian = jacobian @ jacobian.T
        
        # 初始化等权重
        lambdas = torch.full((m,), 1.0 / m)
        
        # 投影梯度下降到单纯形
        for _ in range(self.max_iters):
            gradient = gramian @ lambdas
            candidate = _project_simplex(lambdas - lr * gradient)
            if converged: break
            lambdas = candidate
        
        # 最小范数方向 = J^T @ λ*
        direction = jacobian.T @ lambdas
        return direction
```

**算法原理**：
- 目标：min ||J^T λ||² s.t. λ ∈ Δ^m（m维单纯形）
- 等价于：min λ^T G λ s.t. λ ≥ 0, Σλ_i = 1，其中 G = JJ^T
- 使用投影梯度下降求解，投影到单纯形使用经典算法

**单纯形投影算法**：
```python
def _project_simplex(vector):
    sorted_vector, _ = torch.sort(vector, descending=True)
    cumsum = torch.cumsum(sorted_vector, dim=0)
    steps = torch.arange(1, len(vector) + 1)
    support = sorted_vector - (cumsum - 1.0) / steps > 0
    rho = last_true_index(support)
    theta = (cumsum[rho] - 1.0) / (rho + 1)
    return torch.clamp(vector - theta, min=0.0)
```

### 3.4 基线方法

三种基线方法与FedJD共享客户端实现，但服务器端策略不同：

| 方法 | 上传内容 | 方向计算 | 通信量/客户端 |
|------|----------|----------|--------------|
| FedJD | Jacobian (m×d) | MinNorm(J) | m×d×4 B |
| FMGDA | Jacobian (m×d) | J^T @ (1/m,...,1/m) | m×d×4 B |
| WeightedSum | 梯度 (d) | 本地加权梯度 | d×4 B |
| DirectionAvg | 方向 (d) | 本地Jacobian行均值 | d×4 B |

```python
# core/baselines.py — WeightedSumServer 核心逻辑
class WeightedSumServer:
    def run_round(self, round_idx):
        for client in sampled_clients:
            result = client.compute_jacobian(self._clone_model(), self.objective_fn)
            # 本地立即将Jacobian降维为梯度
            local_gradient = result.jacobian.T @ self.weights  # shape: [d]
            all_gradients.append(local_gradient)
        
        # 服务器端只需平均梯度
        direction = stack(all_gradients).mean(dim=0)
```

### 3.5 压缩管线

FedJD支持7种压缩方案，通过`JacobianCompressor`接口统一：

```python
# compressors/__init__.py — 压缩器接口
class JacobianCompressor(ABC):
    @abstractmethod
    def compress(self, jacobian): -> (compressed, metadata)
    
    @abstractmethod
    def decompress(self, compressed, metadata): -> jacobian
```

| 压缩器 | 压缩比 | 质量损失 | 适用场景 |
|--------|--------|----------|----------|
| NoCompressor | 1.0x | 无 | 基线 |
| Float16Compressor | 2.0x | 极小 | **推荐** |
| TopKCompressor | 可调 | 中等 | 稀疏梯度 |
| RowTopKCompressor | 可调 | 中等 | 每目标稀疏 |
| LowRankCompressor | 高 | 较大 | 低秩结构 |
| RandomSketchCompressor | 可调 | 中等 | 快速近似 |

---

## 4. 关键算法实现逻辑

### 4.1 完整训练流程伪代码

```
Algorithm: FedJD Training
Input: K clients, m objectives, T rounds, η learning rate
Output: Trained global model θ_T

Initialize: θ_0 ← random parameters
For t = 0, 1, ..., T-1:
    S_t ← SampleClients(K, participation_rate)
    
    For each client k ∈ S_t (parallel):
        J_k ← ComputeJacobian(θ_t, D_k)  // [m × d] matrix
        J̃_k ← Compress(J_k)              // optional compression
    
    // Server aggregation
    Ĵ ← Σ_{k∈S_t} (n_k/n_S) · Decompress(J̃_k)  // sample-weighted average
    
    // MinNorm direction finding
    G ← Ĵ · Ĵ^T                                   // [m × m] Gram matrix
    λ* ← argmin_{λ∈Δ^m} λ^T G λ                   // projected gradient descent
    d_t ← Ĵ^T · λ*                                 // [d] direction vector
    
    // Model update
    θ_{t+1} ← θ_t - η · d_t
    
    // Evaluate global objectives
    f_j(θ_{t+1}) ← Σ_{k=1}^{K} (n_k/n) · L_j(θ_{t+1}, D_k)  for j=1..m
```

### 4.2 Jacobian计算的时间复杂度

- 前向传播：O(d)（单次）
- 反向传播：O(m × d)（每个目标一次）
- 总计：O((m+1) × d) ≈ O(m × d)

### 4.3 MinNorm方向寻找的收敛性

MinNormAggregator使用投影梯度下降求解凸优化问题：
- 目标函数 λ^T G λ 是凸函数（G半正定）
- 单纯形是凸集
- 投影梯度下降保证收敛到全局最优
- 收敛速率：O(1/t)（次线性收敛）

---

## 5. 数据处理流程

### 5.1 合成回归数据

```python
# data/synthetic.py — 冲突机制
# 目标j的权重：w_j = -w_0/m + noise（反相关）
# 冲突强度：conflict_strength参数控制反相关系数
true_weights = [base_weight]
for obj_idx in range(1, m):
    conflict_component = -base_weight * conflict_strength
    orthogonal = random_orthogonal_to(base_weight)
    w = conflict_component + orthogonal * diversity_scale
    true_weights.append(w)
```

**数据分布特点**：
- 输入：8维标准正态 + 客户端偏移
- 输出：m维，每维由不同权重向量生成
- 客户端异质性：通过特征偏移实现（0.3 × client_idx）

### 5.2 合成分类数据

```python
# data/classification.py — Non-IID机制
# 每个任务的标签分布按Dirichlet分布采样
# noniid_strength控制Dirichlet浓度参数
label_distribution = Dirichlet(alpha=1.0/(1.0+noniid_strength))
```

### 5.3 高冲突数据

```python
# data/synthetic.py — make_high_conflict_federated_regression
# 核心差异：conflict_strength=1.0时，w_1 ≈ -w_0
# 实测余弦相似度：cos(w_0, w_1) ≈ -0.96（强冲突）
```

---

## 6. 通信机制设计

### 6.1 通信流程

```
Client                          Server
  │                               │
  │     ← broadcast model θ_t     │
  │                               │
  │  compute J_k                  │
  │  compress J̃_k                 │
  │                               │
  │     upload J̃_k ──────────→   │
  │                               │  decompress & aggregate
  │                               │  MinNorm direction finding
  │                               │  update model
  │                               │
  │     ← broadcast model θ_{t+1} │
```

### 6.2 通信量分析

| 方法 | 上传量/客户端/轮 | 下载量/客户端/轮 | 总通信量/T轮 |
|------|-----------------|-----------------|-------------|
| FedJD | m×d×4 B | d×4 B | T×(m×d+d)×4 |
| FedJD+fp16 | m×d×2 B | d×4 B | T×(m×d/2+d)×4 |
| WeightedSum | d×4 B | d×4 B | T×2d×4 |
| DirectionAvg | d×4 B | d×4 B | T×2d×4 |

**关键结论**：FedJD的通信开销是基线方法的m倍。当m=10时，FedJD上传量是WeightedSum的10倍。

### 6.3 非同步轮次优化

FedJD支持`full_sync_interval`参数，在非同步轮次复用上一轮方向：

```python
if not is_full_sync and self._last_direction is not None:
    direction = self._last_direction  # 零通信
    current_flat = current_flat - lr * direction
```

---

## 7. 隐私保护措施

### 7.1 当前实现

FedJD遵循标准联邦学习的隐私模型：
- **数据不离开客户端**：原始数据始终保留在客户端，仅上传Jacobian矩阵
- **服务器不可见数据**：服务器仅接收聚合后的梯度信息
- **客户端采样**：每轮仅部分客户端参与，降低信息泄露风险

### 7.2 潜在隐私风险

1. **Jacobian矩阵信息量更大**：相比单一梯度，Jacobian包含m个目标的独立梯度，理论上泄露更多信息
2. **梯度反转攻击**：从Jacobian可能重构更精确的输入数据
3. **当前未实现差分隐私**：没有梯度裁剪和噪声添加机制

### 7.3 可扩展的隐私增强方案

- **梯度裁剪**：在客户端上传前裁剪Jacobian的每行范数
- **差分隐私噪声**：在Jacobian上添加高斯噪声
- **安全聚合**：使用Secure Aggregation协议，服务器仅看到聚合结果

---

## 8. 评估指标体系

### 8.1 优化质量指标

| 指标 | 定义 | 含义 |
|------|------|------|
| NHV | 归一化超体积 | Pareto前沿质量（0-1，越高越好） |
| NPG | 归一化Pareto Gap | 距理想点距离（0-1，越低越好） |
| Avg RI | 平均相对改善率 | 收敛速度（越高越好） |

### 8.2 系统效率指标

| 指标 | 定义 | 含义 |
|------|------|------|
| Upload/Client | 每客户端每轮上传字节数 | 通信开销 |
| Round Time | 每轮训练时间 | 计算效率 |
| Compression Ratio | 压缩后/原始大小 | 压缩效果 |

### 8.3 鲁棒性指标

| 指标 | 定义 | 含义 |
|------|------|------|
| All Decreased | 所有目标是否下降 | 收敛稳定性 |
| NaN/Inf Count | 数值异常次数 | 数值稳定性 |
| Seed Std | 跨种子标准差 | 可复现性 |
