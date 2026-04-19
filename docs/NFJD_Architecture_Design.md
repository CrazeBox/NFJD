# NFJD 架构设计文档

## 1. 概述

NFJD（New Federated Jacobian Descent）是对FedJD的全面架构改进，解决原FedJD的三个核心缺陷：
1. **收敛速度慢**：MinNorm方向范数过小
2. **通信开销大**：上传m×d的Jacobian矩阵
3. **计算效率低**：m次独立backward

### 改进方案映射

| 缺陷 | 改进方案 | 文档章节 |
|------|----------|----------|
| 收敛速度慢 | AdaptiveRescaling + 动量 | §2.5, §2.6 |
| 通信开销大 | Δθ上传机制 | §2.4 |
| 计算效率低 | 单次backward + λ*复用 | §2.7, §2.2 |
| 高m可扩展性 | StochasticGramianSolver | §2.3 |

---

## 2. 核心组件设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         NFJDTrainer                             │
│  (训练循环编排：轮次管理、日志、检查点)                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │ 每轮调用 server.run_round()
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         NFJDServer                              │
│  ┌──────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │ 客户端采样 │→│ Δθ加权平均(FedAvg)│→│ 全局动量更新          │  │
│  │ (随机比例) │  │ θ_new = Σw_k·Δθ_k│  │ v = β·v + (1-β)·Δθ  │  │
│  └──────────┘  └──────────────────┘  └──────────────────────┘  │
│                           │                                     │
│  ┌────────────────────────▼────────────────────────────────┐    │
│  │ 模型更新: θ ← θ + v                                     │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ 广播模型 θ_t，接收 Δθ_k
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      NFJDClient (×K)                            │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ local_update(): E轮本地训练                            │      │
│  │   For e = 1..E:                                       │      │
│  │     1. 前向传播 → L_1,...,L_m                         │      │
│  │     2. λ* ← 上一步MinNorm权重（首步等权重）            │      │
│  │     3. L_total = Σ λ_i · L_i                          │      │
│  │     4. 单次 backward → 联合梯度                        │      │
│  │     5. [可选] torch.autograd.grad → 独立梯度           │      │
│  │     6. MinNorm/StochasticGramian → 新λ*               │      │
│  │     7. AdaptiveRescaling → d_scaled                   │      │
│  │     8. 本地动量更新 → v_local                          │      │
│  │     9. θ_local ← θ_local - lr · v_local               │      │
│  │   Δθ = θ_final - θ_init                               │      │
│  │   上传: 仅 Δθ（此处仅传模型差值以保护通信带宽）          │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 客户端侧：JD + 自适应缩放

每个客户端在本地执行E轮多目标优化训练：

```python
class NFJDClient:
    def local_update(self, model, objective_fn, num_local_epochs):
        theta_init = copy_parameters(model)
        lambda_star = self.prev_lambda  # 上一步的MinNorm权重，首步为等权重
        
        for e in range(num_local_epochs):
            predictions = model(batch_inputs)
            losses = objective_fn(predictions, batch_targets, batch_inputs)
            
            # 使用上一步λ*构建联合损失，单次backward
            L_total = sum(lambda_star[i] * losses[i] for i in range(m))
            L_total.backward()
            
            # 提取独立梯度（用于MinNorm求解）
            independent_grads = extract_independent_grads(model, losses)
            jacobian = torch.stack(independent_grads)  # [m, d]
            
            # MinNorm方向求解
            lambda_star_new = minnorm_solve(jacobian)
            direction = jacobian.T @ lambda_star_new
            
            # AdaptiveRescaling
            direction = adaptive_rescale(direction, jacobian)
            
            # 本地动量更新
            self.local_momentum = beta_local * self.local_momentum + direction
            update_parameters(model, -lr * self.local_momentum)
            
            lambda_star = lambda_star_new
        
        delta_theta = copy_parameters(model) - theta_init
        return delta_theta  # 此处仅传模型差值以保护通信带宽
```

### 2.3 StochasticGramianSolver

当m > threshold（默认5）时，随机采样子集构建Gram矩阵：

```python
class StochasticGramianSolver:
    def __init__(self, subset_size=4, seed=None):
        self.subset_size = subset_size
        self.rng = random.Random(seed)
    
    def solve(self, jacobian):
        m = jacobian.shape[0]
        if m <= self.subset_size:
            return minnorm_solve(jacobian)
        
        # 随机采样子集
        indices = sorted(self.rng.sample(range(m), self.subset_size))
        sub_jacobian = jacobian[indices]
        
        # 求解子集上的MinNorm
        sub_lambda = minnorm_solve(sub_jacobian)
        
        # 扩展到完整权重（未采样目标权重为0）
        full_lambda = torch.zeros(m)
        for i, idx in enumerate(indices):
            full_lambda[idx] = sub_lambda[i]
        
        # rescale_alpha仍基于完整梯度集合的范数计算
        direction = jacobian.T @ full_lambda
        return direction, indices
```

### 2.4 通信效率保护机制

**核心约束**：客户端严禁将Jacobian矩阵、各目标独立梯度或中间权重λ发送给服务器。上传给服务器的数据必须且仅有Δθ。

```python
class NFJDServer:
    def run_round(self, round_idx):
        # 广播当前模型
        theta_current = flatten_parameters(self.model.parameters())
        
        # 收集客户端更新
        delta_thetas = []
        for client in sampled_clients:
            # 客户端本地训练，返回Δθ
            delta_theta = client.local_update(self._clone_model(), self.objective_fn, self.num_local_epochs)
            weight = client.num_examples / total_examples
            delta_thetas.append((delta_theta, weight))
        
        # FedAvg式加权平均
        aggregated_delta = sum(w * dt for dt, w in delta_thetas)
        
        # 全局动量
        self.global_momentum = self.beta * self.global_momentum + (1 - self.beta) * aggregated_delta
        
        # 模型更新
        theta_new = theta_current + self.global_momentum
        assign_flat_parameters(self.model.parameters(), theta_new)
```

**通信量分析**：
| 方法 | 上传量/客户端 | 与m的关系 |
|------|-------------|-----------|
| FedJD (旧) | m × d × 4 B | 线性增长 |
| NFJD (新) | d × 4 B | **完全解耦** |

### 2.5 AdaptiveRescaling 标准化层

在MinNorm方向求解后、应用更新前插入：

```python
class AdaptiveRescaling:
    def __init__(self, epsilon=1e-8, max_scale=10.0):
        self.epsilon = epsilon
        self.max_scale = max_scale
    
    def __call__(self, direction, jacobian):
        # 参考范数：各目标梯度均值的L2范数
        mean_grad = jacobian.mean(dim=0)
        N_raw = torch.norm(mean_grad, p=2).item()
        
        # 当前方向范数
        N_d = torch.norm(direction, p=2).item() + self.epsilon
        
        # 缩放因子
        scale = min(N_raw / N_d, self.max_scale)
        
        # 缩放后的方向
        scaled_direction = direction * scale
        return scaled_direction
```

**设计意图**：即使在目标维度极高、Gram矩阵病态导致d的范数趋近于零时，此机制也能强行将步长恢复至与简单等权重平均法相当的量级。这既保留了JD算法"不伤害任何目标"的方向优势，又彻底解决了收敛停滞问题。

### 2.6 双动量机制

```python
# 客户端本地动量
class LocalMomentum:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.velocity = None
    
    def update(self, direction):
        if self.velocity is None:
            self.velocity = direction.clone()
        else:
            self.velocity = self.beta * self.velocity + (1 - self.beta) * direction
        return self.velocity.clone()

# 服务器全局动量
class GlobalMomentum:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.velocity = None
    
    def update(self, aggregated_delta):
        if self.velocity is None:
            self.velocity = aggregated_delta.clone()
        else:
            self.velocity = self.beta * self.velocity + (1 - self.beta) * aggregated_delta
        return self.velocity.clone()
```

### 2.7 单次联合反向传播

将m次独立backward优化为1次联合backward：

```python
def compute_jacobian_and_loss(model, objective_fn, inputs, targets, prev_lambda):
    predictions = model(inputs)
    losses = objective_fn(predictions, targets, inputs)
    m = len(losses)
    
    # 使用上一步λ*构建联合损失
    if prev_lambda is None:
        lambda_weights = torch.ones(m) / m
    else:
        lambda_weights = prev_lambda
    
    L_total = sum(lambda_weights[i] * losses[i] for i in range(m))
    
    # 单次backward计算联合梯度
    model.zero_grad(set_to_none=True)
    L_total.backward(retain_graph=True)
    joint_grad = flatten_gradients(model.parameters())
    
    # 提取独立梯度用于MinNorm求解
    independent_grads = []
    for i in range(m):
        model.zero_grad(set_to_none=True)
        losses[i].backward(retain_graph=(i < m - 1))
        independent_grads.append(flatten_gradients(model.parameters()))
    
    jacobian = torch.stack(independent_grads)
    model.zero_grad(set_to_none=True)
    
    return jacobian, L_total, losses
```

**优化说明**：
- 联合梯度用于参数更新（1次backward）
- 独立梯度用于MinNorm权重求解（需要retain_graph，但仅在需要时计算）
- 当使用StochasticGramianSolver时，只需提取k个目标的梯度，进一步减少开销

---

## 3. 与FedJD的对比

| 特性 | FedJD (旧) | NFJD (新) |
|------|-----------|-----------|
| 上传内容 | Jacobian (m×d) | **Δθ (d)** |
| 通信量/客户端 | m×d×4 B | **d×4 B** |
| 服务器聚合 | MinNorm方向寻找 | **FedAvg加权平均** |
| 方向范数 | 保守（范数小） | **AdaptiveRescaling恢复** |
| 动量 | 无 | **双动量（本地+全局）** |
| Backward次数 | m次/轮 | **1次/轮** |
| 高m处理 | 直接计算（O(m²)） | **StochasticGramian (O(k²))** |
| 隐私保护 | 仅数据不出客户端 | **Δθ天然遮蔽梯度信息** |

---

## 5. 性能优化措施

### 5.1 GPU加速优化

**问题**：在GPU上运行时，`.item()`调用会导致GPU-CPU同步，造成性能瓶颈。

**解决方案**：使用纯GPU tensor操作替代`.item()`调用：

```python
# 优化前（有GPU-CPU同步）
rho = int(support_indices[-1].item())
theta = float((cumsum[rho] - 1.0) / (rho + 1))

# 优化后（纯GPU操作）
rho = support_indices[-1, 0]
theta = (cumsum[rho] - 1.0) / (rho + 1.0)
```

### 5.2 AdaptiveRescaling优化

**问题**：每次缩放计算都需要进行GPU-CPU同步来获取Python标量。

**解决方案**：使用向量化操作和tensor比较：

```python
# 优化前
scale = min(raw_norm / direction_norm, self.max_scale)

# 优化后（保持GPU在设备上）
scale = torch.minimum(raw_norm / direction_norm,
                      torch.tensor(self.max_scale, device=direction.device, dtype=direction.dtype))
```

### 5.3 梯度缓存策略

**实现机制**：通过`recompute_interval`参数控制Jacobian计算的频率：

```python
need_recompute = (step_idx % self.recompute_interval == 0) or (self.prev_lambda is None)

if need_recompute:
    # 完整路径：Jacobian → MinNorm → Rescaling → Momentum → Update
    # 计算完整的Jacobian和MinNorm方向
else:
    # Cheap step：用上一次lambda构造加权损失，单次反向传播
    # 跳过Jacobian计算和MinNorm QP求解，大幅减少计算开销
```

**性能收益**：
- 当`recompute_interval > 1`时，非重算步只需1次backward
- 跳过m个独立梯度计算和O(m²)的MinNorm QP求解
- 约90%+的计算时间节省

### 5.4 StochasticGramianSolver

**适用场景**：当目标数m较大时（m > subset_size），随机采样子集构建Gram矩阵。

**性能收益**：
- 计算复杂度从O(m²)降至O(k²)，其中k为子集大小
- 内存占用从O(m×d)降至O(k×d)

### 5.5 动态迭代次数调整

**策略**：根据目标数m动态调整MinNorm QP求解的迭代次数：

```python
@staticmethod
def _compute_dynamic_iters(m: int) -> int:
    if m <= 2:
        return 50
    if m <= 5:
        return 100
    if m <= 8:
        return 200
    return 250
```

---

## 6. 实验计划

### Stage 1：基线验证
- 验证NFJD核心链路正确性
- 对比NFJD vs FedJD vs WeightedSum vs DirectionAvg
- 在MultiMNIST和CelebA数据集上进行验证

### Stage 2：消融实验
- AdaptiveRescaling消融
- 动量消融（无/仅全局/双动量）
- StochasticGramian消融
- 本地轮次E消融

### Stage 3：高冲突数据验证
- 使用高冲突合成数据
- 验证NFJD在高冲突场景下的优势

### Stage 4：完整基准
- 多任务、多m值、Non-IID完整对比
- 在CelebA数据集上测试不同属性数量（2-6个）的性能
- 验证NFJD在图像多属性预测任务中的表现
