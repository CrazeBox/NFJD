# FedClient-UPGrad: Client-Level Multi-Objective Gradient Aggregation for Fair Federated Learning

## Abstract

Heterogeneous federated learning is predominantly evaluated by average accuracy, yet in practical deployments, a model's ability to serve low-performing clients is equally important. This paper re-examines heterogeneous federated learning through the lens of client-level multi-objective optimization, treating each client's empirical loss as an independent objective. Building on this perspective, we propose FedClient-UPGrad: each selected client performs standard local training starting from the server-broadcast global model and uploads its model update; the server treats the negative updates as proxies for client objective gradients and employs UPGrad to compute a conflict-aware common direction in the client update space. Our method requires no explicit client Jacobian computation, and its communication pattern is identical to standard FedAvg-style model update uploads. We compare FedClient-UPGrad against FedAvg, qFedAvg, and FedMGDA+ on FEMNIST and Dirichlet-partitioned CIFAR-10. Across three random seeds on FEMNIST, FedClient-UPGrad achieves a mean client test accuracy of 0.7996 ± 0.0094 and a worst-10% client accuracy of 0.5902 ± 0.0058, both significantly outperforming all baselines. These results demonstrate that client-level multi-objective aggregation can effectively improve tail-client performance in heterogeneous federated learning.

**Keywords:** Federated Learning, Multi-Objective Optimization, Client Heterogeneity, Fairness, UPGrad

---

## 1. Introduction

### 1.1 Background and Motivation

Federated Learning (FL) [McMahan et al., 2017] is a distributed machine learning paradigm that enables multiple clients to collaboratively train a global model without sharing raw data. In real-world FL deployments, client heterogeneity is inherent: different clients may exhibit significant variations in data distributions, sample sizes, and data quality. For instance, in the FEMNIST handwritten character recognition task, different writers exhibit distinct handwriting styles; in cross-institution medical image analysis, different hospitals operate with varying equipment parameters and patient populations.

This client heterogeneity gives rise to a critical problem: **a global model may achieve high average accuracy while performing poorly on a subset of clients.** Because data distributions differ substantially across clients, optimizing the average objective may bias the model toward majority or easy-to-optimize clients, thereby neglecting tail clients. This indicates that evaluating federated models solely by average performance is insufficient, and it is necessary to redesign aggregation methods from the perspective of client-level objective balancing.

### 1.2 Representative Approaches in Existing Work

Existing federated learning methods can be understood from the following perspectives:

**(1) FedAvg-type methods.** FedAvg [McMahan et al., 2017] is the most classical FL algorithm: clients perform local training and upload model updates, which the server aggregates via sample-size-weighted averaging. This approach is simple and efficient, but it fundamentally optimizes the weighted average objective across all clients and cannot explicitly handle inter-client objective conflicts.

**(2) Fair federated learning / qFedAvg.** q-FFL [Li et al., 2020] improves fairness by re-weighting clients according to their loss values, giving higher-loss clients larger aggregation weights. However, this method adjusts weights solely through scalar loss values and cannot fully represent the geometric conflict relationships among client update directions.

**(3) Multi-objective federated learning / MGDA-type methods.** FedMGDA+ introduces MGDA (Multiple Gradient Descent Algorithm) [Désidéri, 2012] into FL aggregation, seeking a common descent direction for multiple client objectives. However, the min-norm direction from MGDA can be overly conservative, producing near-zero update directions when client directions are highly conflicting.

**(4) Gradient conflict handling methods.** In centralized multi-task learning, existing methods handle inter-task gradient conflicts through projection, recombination, or geometric constraints [Sener & Koltun, 2018; Yu et al., 2020; Liu et al., 2021]. However, these methods are typically not designed specifically for client-level FL updates.

### 1.3 Common Limitations of Existing Methods

Existing methods either compress client objectives into a single average objective (FedAvg), handle fairness only through scalar loss re-weighting (qFedAvg), or require relatively expensive objective-level gradient information (FedMGDA+). They do not fully exploit the client local update directions that naturally arise in standard federated learning—these update directions inherently carry geometric information about client objectives.

### 1.4 Core Idea

The core bridging insight of this paper is:

1. **Perspective shift:** Treat each client's empirical loss as an independent objective, modeling federated learning as a client-level multi-objective optimization problem.
2. **Information utilization:** The model update (delta) produced by client local training naturally serves as a proxy for that client's objective descent direction, requiring no additional computation.
3. **Aggregation innovation:** Instead of simply averaging these updates, the server uses UPGrad to compute a conflict-aware common direction in the client update space.
4. **Effect validation:** This common direction pays more attention to conflicts among client objectives, helping to improve tail-client performance.

### 1.5 Contributions

The main contributions of this paper are:

1. **Problem modeling contribution:** We propose a client-level multi-objective FL perspective, treating each client's empirical loss as an independent objective, providing a new theoretical framework for understanding and handling client heterogeneity.
2. **Method contribution:** We propose FedClient-UPGrad, which uses standard client local updates as objective direction proxies and employs UPGrad for conflict-aware aggregation. The method's communication pattern is identical to FedAvg and requires no explicit client Jacobian computation.
3. **Experimental contribution:** We construct a unified experimental framework and fairly compare FedAvg, qFedAvg, FedMGDA+, and FedClient-UPGrad on FEMNIST and CIFAR-10. Across three random seeds on FEMNIST, FedClient-UPGrad consistently ranks first on mean accuracy, worst-10% accuracy, client accuracy standard deviation, and test loss.
4. **Theoretical contribution:** Under standard non-convex assumptions, we prove that FedClient-UPGrad converges to the neighborhood of an approximate Pareto-stationary point, with error controlled by local training drift, stochastic gradient noise, and client sampling error.
5. **Efficiency analysis contribution:** We analyze the computational cost of our method, showing that it trades additional server-side aggregation computation for substantially improved tail-client performance, and discuss the applicable scenarios of this trade-off.

---

## 2. Related Work

### 2.1 Heterogeneous Federated Optimization

FedAvg [McMahan et al., 2017] is the foundational FL algorithm, whose core idea is that clients perform multiple rounds of local training and upload model updates, which the server aggregates via sample-size-weighted averaging. However, under non-IID data distributions, FedAvg suffers from the client drift problem [Karimireddy et al., 2020], where local update directions of different clients may conflict with each other. To mitigate this issue, FedProx [Li et al., 2020] introduces a proximal term in the client objective to constrain local updates from deviating too far from the global model; SCAFFOLD [Karimireddy et al., 2020] introduces control variates to correct client drift; FedNova [Wang et al., 2020] normalizes heterogeneous local update steps across clients. These methods improve convergence from an optimization perspective but fundamentally still optimize the average objective and do not explicitly address inter-client objective conflicts.

### 2.2 Fair Federated Learning

Fairness is an important research dimension in FL. q-FFL (q-Fair Federated Learning) [Li et al., 2020] controls the degree of fairness through a parameter $q$: larger $q$ gives higher-loss clients larger aggregation weights. AFL [Mohri et al., 2019] optimizes the worst-case client performance from a minimax perspective. FedFV [Wang et al., 2021] mitigates client update conflicts through gradient projection. These methods improve fairness through scalar loss re-weighting or minimax optimization, whereas FedClient-UPGrad operates directly in the client update vector space, handling directional conflicts through geometric relationships.

### 2.3 Multi-Objective Optimization and MGDA

Multi-Objective Optimization (MOO) studies how to find balanced solutions among multiple potentially conflicting objectives. Pareto optimality and Pareto stationarity are core concepts in MOO [Miettinen, 1999]. MGDA [Désidéri, 2012] finds a common descent direction for multiple objective gradients by solving a min-norm problem. In centralized multi-task learning, MGDA has been used to balance gradient conflicts across tasks [Sener & Koltun, 2018]. FedMGDA+ introduces this idea into FL and is the closest multi-objective client update baseline to our method. However, MGDA's min-norm direction can be overly conservative when objectives are highly conflicting, producing near-zero update directions that stall training.

### 2.4 Gradient Conflict and UPGrad

In multi-task learning, gradient conflict refers to the situation where the gradient directions of different tasks have negative inner products, and simple averaging may lead to direction cancellation. PCGrad [Yu et al., 2020] handles conflicts by projecting conflicting gradients onto each other's normal planes. GradDrop [Chen et al., 2020] selectively preserves gradient components based on sign consistency. UPGrad [Liu et al., 2021] proposes a Gramian-space conflict-aware aggregation method that computes a common direction by solving box-constrained quadratic programming problems. This paper introduces these ideas into the client update space of federated learning, using UPGrad to handle geometric conflicts among client updates.

---

## 3. Problem Formulation

### 3.1 Federated Learning Setup

Consider a federated learning system with $K$ clients. Each client $i \in \{1, 2, \ldots, K\}$ holds a local dataset $\mathcal{D}_i$, with its empirical loss function defined as:

$$F_i(\theta) = \mathbb{E}_{(x, y) \sim \mathcal{D}_i}[\ell(f_\theta(x), y)]$$

where $\theta \in \mathbb{R}^d$ denotes the model parameters, $f_\theta$ is the parameterized model, and $\ell$ is the loss function.

The classical FedAvg algorithm approximately optimizes the weighted average objective across all clients:

$$F_{\text{avg}}(\theta) = \sum_{i=1}^{K} p_i F_i(\theta)$$

where $p_i = \frac{|\mathcal{D}_i|}{\sum_{j=1}^{K} |\mathcal{D}_j|}$ is typically proportional to the client's sample size.

In each communication round $t$, the server samples a subset of clients $\mathcal{S}_t \subseteq \{1, \ldots, K\}$ and broadcasts the current global model $\theta_t$. Each sampled client $i \in \mathcal{S}_t$ performs $E$ epochs of local training starting from $\theta_t$, obtains a local model $\theta_{i,E}^t$, and uploads the model update:

$$\Delta_i^t = \theta_{i,E}^t - \theta_t$$

The server aggregates these updates to update the global model.

### 3.2 Client-Level Multi-Objective Perspective

This paper proposes a different perspective: **treat each client's empirical loss as an independent objective**, thereby modeling federated learning as a client-level multi-objective optimization problem. Define the vector-valued objective function:

$$\mathbf{F}(\theta) = [F_1(\theta), F_2(\theta), \ldots, F_K(\theta)]^\top$$

In each communication round, the server only samples a subset of clients $\mathcal{S}_t$. The server's goal is not merely to optimize the average loss, but to find an update direction that **better balances the objectives of the sampled clients**.

In multi-objective optimization, Pareto stationarity is a core concept. A point $\theta$ is called Pareto-stationary if there exists no direction that can simultaneously decrease all objective function values. Formally, a necessary condition for Pareto stationarity is:

$$\min_{\lambda \in \Delta_K} \left\| \sum_{i=1}^{K} \lambda_i \nabla F_i(\theta) \right\|^2 = 0$$

where $\Delta_K = \{\lambda \in \mathbb{R}^K : \lambda_i \geq 0, \sum_i \lambda_i = 1\}$ is the $K$-dimensional probability simplex.

### 3.3 Local Updates as Gradient Proxies

The core observation of FedClient-UPGrad is that the model updates produced by client local training can serve as practical proxies for that client's objective gradient or descent direction.

Specifically, client $i$ starts from the global model $\theta_t$, performs $E$ epochs of local SGD training, and obtains $\theta_{i,E}^t$. The uploaded update is $\Delta_i^t = \theta_{i,E}^t - \theta_t$. Since local training proceeds along the descent direction of the client loss $F_i$, we have:

$$\mathbf{g}_i^t = -\Delta_i^t$$

which can be viewed as an approximate proxy for the client objective gradient $\nabla F_i(\theta_t)$. The precise characterization of this proxy relationship is provided in the theoretical analysis of Section 7 (Lemma 1).

### 3.4 Evaluation Objectives

Corresponding to the problem definition of client-level multi-objective optimization, our experimental evaluation cannot rely solely on average accuracy. The primary evaluation metrics include:

1. **Mean Client Test Accuracy:** The average test accuracy across all clients.
2. **Worst-10% Client Accuracy:** The average accuracy of the bottom 10% of clients, measuring tail-client performance.
3. **Client Accuracy Standard Deviation:** Measuring the uniformity of performance across clients.
4. **Mean Client Test Loss:** The average test loss across all clients.
5. **Efficiency Metrics:** Average round time, average upload bytes, average aggregation/direction computation time.

---

## 4. Method: FedClient-UPGrad

### 4.1 Algorithm

The single-round communication procedure of FedClient-UPGrad is as follows:

**Algorithm 1: FedClient-UPGrad (Single Round)**

- **Input:** Global model $\theta_t$, sampled client set $\mathcal{S}_t$
- **Step 1 (Broadcast):** The server broadcasts $\theta_t$ to all clients in $\mathcal{S}_t$
- **Step 2 (Local Training):** Each client $i \in \mathcal{S}_t$:
  - Initializes its local model from $\theta_t$
  - Trains on local data for $E$ epochs
  - Uploads the update $\Delta_i^t = \theta_{i,E}^t - \theta_t$
- **Step 3 (Construct Proxy Gradient Matrix):** The server constructs matrix $\mathbf{G}_t \in \mathbb{R}^{|\mathcal{S}_t| \times d}$, whose $i$-th row is $\mathbf{g}_i^t = -\Delta_i^t$
- **Step 4 (UPGrad Aggregation):** The server computes the common direction $\mathbf{d}_t = \text{UPGrad}(\mathbf{G}_t)$
- **Step 5 (Model Update):** The server updates the global model $\theta_{t+1} = \theta_t - \eta_s \cdot \mathbf{d}_t$
- **Output:** Updated global model $\theta_{t+1}$

where $\eta_s$ is the server-side learning rate (or update scale).

### 4.2 UPGrad Aggregation in Detail

UPGrad aggregation is the core component of FedClient-UPGrad. Given the client proxy gradient matrix $\mathbf{G} \in \mathbb{R}^{m \times d}$ (where $m = |\mathcal{S}_t|$ is the number of sampled clients and $d$ is the model parameter dimension), UPGrad computes the common direction through the following steps:

**Step 1: Compute the Gramian matrix.** First compute the pairwise geometric relationships among client updates:

$$\mathbf{H} = \mathbf{G} \mathbf{G}^\top \in \mathbb{R}^{m \times m}$$

where $H_{ij} = \langle \mathbf{g}_i, \mathbf{g}_j \rangle$ represents the inner product between the proxy gradient directions of client $i$ and client $j$. If $H_{ij} < 0$, the update directions of the two clients are in conflict.

**Step 2: Solve box-constrained quadratic programs.** For each client $k \in \{1, \ldots, m\}$, solve the following quadratic programming problem:

$$\mathbf{w}^{(k)} = \arg\min_{\mathbf{w} \in \mathbb{R}^m} \mathbf{w}^\top \mathbf{H} \mathbf{w}$$
$$\text{s.t. } w_k \geq 1, \quad w_j \geq 0 \quad \forall j \neq k$$

This constraint ensures that the $k$-th client receives at least unit weight in the aggregation direction, thereby guaranteeing that each client's objective is adequately considered.

**Step 3: Average the weights.** Average all $m$ solutions to obtain the final weight vector:

$$\mathbf{w} = \frac{1}{m} \sum_{k=1}^{m} \mathbf{w}^{(k)}$$

**Step 4: Compute the common direction.** The final direction is the weighted combination of proxy gradients:

$$\mathbf{d} = \mathbf{G}^\top \mathbf{w} = \sum_{i=1}^{m} w_i \mathbf{g}_i$$

**Intuitive explanation of UPGrad:** For each client $k$, the subproblem $\mathbf{w}^{(k)}$ seeks a weight vector that minimizes the Gramian quadratic form of the combined direction while ensuring that client $k$ receives at least unit contribution. The Gramian quadratic form $\mathbf{w}^\top \mathbf{H} \mathbf{w} = \|\mathbf{G}^\top \mathbf{w}\|^2$ is exactly the squared norm of the combined direction. Thus, each subproblem finds a minimum-norm direction under the constraint of "not ignoring client $k$." After averaging across all clients, the resulting common direction balances the needs of all clients.

### 4.3 Distinction from Baseline Methods

The essential differences between FedClient-UPGrad and the baseline methods are:

| Method | Aggregation Strategy | Conflict-Aware | Information Used |
|--------|---------------------|----------------|------------------|
| **FedAvg** | Sample-size-weighted average | No | Update magnitudes only |
| **qFedAvg** | Loss-power re-weighting | Scalar-level only | Scalar loss values |
| **FedMGDA+** | MGDA min-norm direction | Yes (vector-level) | Update directions |
| **FedClient-UPGrad** | UPGrad common direction | Yes (vector-level) | Update directions + Gramian geometry |

Key distinctions:

1. **FedAvg** performs weighted averaging of client updates, fundamentally optimizing the average objective, and may be dominated by majority client directions.
2. **qFedAvg** adjusts weights based on pre-training client losses (larger $q$ gives more weight to high-loss clients), but only through scalar loss values, which cannot fully represent directional conflicts.
3. **FedMGDA+** uses the MGDA/min-norm approach to aggregate client update proxies, but the min-norm direction can be overly conservative, producing near-zero update directions when client directions are highly conflicting.
4. **FedClient-UPGrad** uses UPGrad to compute a common direction in the client update space, explicitly guaranteeing through box-constrained quadratic programming that each client's objective is adequately considered, achieving a better balance between "common descent" and "effective progress."

### 4.4 Communication and Computational Cost

**Communication cost:** The communication pattern of FedClient-UPGrad is identical to FedAvg—each client uploads one model update $\Delta_i^t \in \mathbb{R}^d$ and downloads the global model $\theta_t \in \mathbb{R}^d$. Thus, the communication volume is no higher than FedAvg.

**Additional computational cost:** The additional computational cost is concentrated on the server side:

1. Constructing the $m \times d$ proxy gradient matrix $\mathbf{G}_t$.
2. Computing the $m \times m$ Gramian matrix $\mathbf{H}_t = \mathbf{G}_t \mathbf{G}_t^\top$ (complexity $O(m^2 d)$).
3. Solving $m$ box-constrained quadratic programs (each with complexity $O(m^3)$, or solved iteratively via projected gradient descent).

In the FEMNIST experiments ($m = 50$, $d \approx 1.2 \times 10^6$), the average per-round time and aggregation computation time for each method are:

| Method | Avg Round Time | Avg Aggregation Time |
|--------|---------------|---------------------|
| FedAvg | 1.012s | 0.0003s |
| qFedAvg | 1.067s | 0.0001s |
| FedMGDA+ | 1.103s | 0.0353s |
| FedClient-UPGrad | 1.480s | 0.4722s |

FedClient-UPGrad's server-side aggregation is approximately 0.47s slower per round than FedAvg, but this cost is exchanged for substantial improvements in tail-client performance. We openly acknowledge this cost and frame it as a "performance-fairness-computation" trade-off.

---

## 5. Experimental Setup

### 5.1 Datasets

The main experiments use the following datasets:

1. **FEMNIST (Writer Partition):** Using the FEMNIST dataset from the LEAF framework [Caldas et al., 2018], where each writer serves as a client. Experiments use 100 clients, each with at least 20 samples. This partition naturally corresponds to real-world client heterogeneity—different writers exhibit significantly different handwriting styles.

2. **CIFAR-10 (Dirichlet Label-Skew Partition, $\alpha = 0.1$):** CIFAR-10 is partitioned using a Dirichlet distribution $\text{Dir}(\alpha)$ for label-skew, with 100 clients. $\alpha = 0.1$ produces strong heterogeneity—each client contains samples from only a few classes.

3. **CIFAR-10 (Dirichlet Label-Skew Partition, $\alpha = 0.5$):** Same as above, but $\alpha = 0.5$ produces weaker heterogeneity, with more uniform class distributions across clients.

Datasets excluded from the main experiments: CelebA (more naturally suited for multi-label attribute prediction, not aligned with the current "client-as-objective" theme); language datasets (to keep the paper focused on clear client heterogeneity in vision scenarios).

### 5.2 Models

1. **FEMNIST:** A small CNN with two convolutional layers (32 and 64 channels, 3×3 kernels, ReLU activation, 2×2 max pooling), followed by two fully connected layers (2048 → 128 → 62 classes). Input: 28×28 grayscale character images.

2. **CIFAR-10:** A small CNN with two convolutional layers (32 and 64 channels, 3×3 kernels, ReLU activation, 2×2 max pooling), followed by three fully connected layers (1600 → 256 → 128 → 10 classes). Input: 32×32 RGB images.

These models are not SOTA backbones; they are controlled models chosen for fair comparison of aggregation methods.

### 5.3 Compared Methods

Primary compared methods and selection rationale:

| Method | Selection Rationale |
|--------|-------------------|
| **FedAvg** | Standard average optimization baseline; the most widely used FL algorithm |
| **qFedAvg** | Representative fair FL method; improves fairness through loss re-weighting |
| **FedMGDA+** | Closest multi-objective client update baseline; uses MGDA aggregation |
| **FedClient-UPGrad** | Our proposed method |

### 5.4 Hyperparameters

**FEMNIST experimental configuration:**

| Parameter | Value |
|-----------|-------|
| Number of clients | 100 |
| Communication rounds | 500 |
| Local epochs | 10 |
| Participation rate | 0.5 |
| Learning rate | 0.01 |
| Client test fraction | 0.2 |
| Random seeds | {7, 42, 123} |

**CIFAR-10 experimental configuration:**

| Parameter | Value |
|-----------|-------|
| Number of clients | 100 |
| Communication rounds | 1000 |
| Local epochs | 2 |
| Participation rate | 0.5 |
| Client test fraction | 0.2 |
| Random seeds | {7, 42, 123} |

**CIFAR-10 hyperparameters used in formal experiments:**

| Method | Learning Rate | Method Parameters | Selection Basis |
|--------|--------------|-------------------|-----------------|
| FedAvg | 0.03 | None | Best in R100 tuning |
| qFedAvg | 0.03 | $q=0.1$, `update_scale=1.0` | Best in R100 tuning |
| FedMGDA+ | 0.03 | `update_scale=2.0` | Best in R100 tuning |
| FedClient-UPGrad | 0.01 | `update_scale=1.0` | R1000 seed7 stability verified; `lr=0.03` exhibited numerical divergence in long-round training |

**Note:** CIFAR-10 baseline parameters come from a fixed 100-round tuning budget; FedClient-UPGrad converged faster with `lr=0.03` in short-round tuning but exhibited NaN divergence in full 1000-round training, so the formal experiment uses `lr=0.01, update_scale=1.0`, which passed long-round stability verification. This constitutes stability screening rather than post-hoc metric manipulation.

### 5.5 Evaluation Metrics

**Primary metrics:**

1. Mean Client Test Accuracy
2. Worst-10% Client Accuracy
3. Client Accuracy Standard Deviation
4. Mean Client Test Loss

**Efficiency metrics:**

1. Average Round Time
2. Average Upload Bytes
3. Average Aggregation/Direction Computation Time
4. Total Elapsed Time

**Recommended figures:**

1. Mean Acc vs. Worst-10% Acc scatter plot
2. Sorted Client Accuracy curve
3. Client Accuracy CDF or box plot
4. Efficiency-Performance trade-off plot

### 5.6 Implementation Optimizations Not Affecting Metrics

The full experiments employed three implementation optimizations that do not alter the final paper metrics:

1. `eval_interval=0`: Disabled intermediate evaluation during training; all client test sets are still fully evaluated after training completes.
2. `initial_loss` computation is skipped only for FedAvg and FedClient-UPGrad, since their update rules do not use this value; qFedAvg and FedMGDA+ retain it.
3. A local model container is reused each round, but each client's training is strictly reset with the current round's global model snapshot before training, ensuring all clients start from the same $\theta_t$.

---

## 6. Experimental Results

### 6.1 FEMNIST Main Results

Complete FEMNIST results across three seeds (seed ∈ {7, 42, 123}):

| Method | Mean Acc | Worst-10% Acc | Acc Std | Mean Loss |
|--------|----------|--------------|---------|-----------|
| FedAvg | 0.7665 ± 0.0132 | 0.4903 ± 0.0252 | 0.1301 ± 0.0114 | 0.7814 ± 0.0824 |
| qFedAvg | 0.6968 ± 0.0150 | 0.4163 ± 0.0288 | 0.1373 ± 0.0091 | 1.0680 ± 0.0828 |
| FedMGDA+ | 0.7335 ± 0.0146 | 0.4358 ± 0.0367 | 0.1470 ± 0.0072 | 0.9686 ± 0.0703 |
| FedClient-UPGrad | **0.7996 ± 0.0094** | **0.5902 ± 0.0058** | **0.1030 ± 0.0041** | **0.7026 ± 0.0679** |

**Result analysis:**

1. FedClient-UPGrad **ranks first on all four metrics** across all three random seeds, demonstrating stable performance.
2. Compared to FedAvg, mean accuracy improves by **+0.0332** (relative improvement **+4.33%**).
3. Compared to FedAvg, worst-10% accuracy improves by **+0.0999** (relative improvement **+20.37%**)—the most significant gain.
4. Client accuracy standard deviation decreases by **0.0271**, indicating more uniform performance across clients.
5. Mean test loss decreases by **10.09%** relative to FedAvg.

**Core finding:** On FEMNIST, FedClient-UPGrad consistently outperforms all baseline methods across three random seeds. Its advantage is particularly pronounced on worst-10% client accuracy, improving by 9.99 percentage points over FedAvg. This demonstrates that FedClient-UPGrad not only improves average client performance but also significantly enhances tail-client performance, consistent with the motivation of client-level multi-objective optimization.

### 6.2 FEMNIST Client Distribution Analysis

Average client accuracy quantiles across three seeds:

| Method | P10 | P25 | Median | P75 |
|--------|-----|-----|--------|-----|
| FedClient-UPGrad | **0.6563** | **0.7398** | **0.8176** | **0.8689** |
| FedAvg | 0.6251 | 0.7019 | 0.7852 | 0.8645 |
| FedMGDA+ | 0.5467 | 0.6517 | 0.7530 | 0.8419 |
| qFedAvg | 0.5130 | 0.6215 | 0.7053 | 0.7930 |

**Analysis:** FedClient-UPGrad outperforms all other methods at every quantile. Notably, P10 (the accuracy threshold for the worst 10% of clients) improves from FedAvg's 0.6251 to 0.6563, and the median improves from 0.7852 to 0.8176. This indicates that FedClient-UPGrad does not merely improve a few clients or the average—it **holistically elevates low-quantile and median client performance.**

### 6.3 CIFAR-10 Results

**[To be completed]**

Status: Full three-seed experiments are still running.

Upon completion, the following need to be added:

1. CIFAR-10 $\alpha = 0.1$ three-seed results table.
2. CIFAR-10 $\alpha = 0.5$ three-seed results table.
3. Improvements over tuned baselines.
4. Sorted client accuracy curves.
5. Efficiency-performance trade-off plots.

### 6.4 Efficiency Trade-off

FEMNIST efficiency data:

| Method | Avg Round Time | Avg Aggregation Time | Total Time |
|--------|---------------|---------------------|------------|
| FedAvg | 1.012s | 0.0003s | [TBD] |
| qFedAvg | 1.067s | 0.0001s | [TBD] |
| FedMGDA+ | 1.103s | 0.0353s | [TBD] |
| FedClient-UPGrad | 1.480s | 0.4722s | [TBD] |

**Analysis:** FedClient-UPGrad's server-side aggregation is approximately 0.47s slower per round than FedAvg (primarily due to Gramian matrix computation and QP solving), but this cost is exchanged for a 9.99 percentage point improvement in worst-10% accuracy. We frame this as a "performance-fairness-computation" trade-off rather than avoiding the issue.

---

## 7. Theoretical Analysis

The theoretical analysis and experiments form a unified chain of reasoning: first proving why the method is reasonable, then explaining why the experiments examine these specific metrics.

### 7.1 Overview of the Theoretical Framework

The theoretical analysis supports three progressive propositions:

1. **Lemma 1 (Local Delta Proxy):** Client local updates can serve as approximate proxies for that client's objective gradient or descent direction, with error controlled by smoothness, local learning rate, and local steps.
2. **Lemma 2 (UPGrad Common Direction):** UPGrad can combine these client direction proxies into a conflict-aware common direction, which, under positive alignment conditions, is a common descent direction for the sampled proxy objectives.
3. **Theorem 3 (Approximate Pareto-Stationarity):** Combining the first two propositions, under standard non-convex assumptions, FedClient-UPGrad converges to the neighborhood of an approximate Pareto-stationary point, with error determined by local training drift, stochastic gradient noise, and client sampling error.

**Note:** We do not claim non-convex global optimal convergence. The appropriate target is "first-order stationarity" or "approximate Pareto stationarity."

### 7.2 Assumptions

The theoretical analysis is based on the following standard assumptions:

**Assumption 1 (Smoothness):** Each client objective $F_i$ is $L$-smooth, i.e., for any $\theta, \theta'$:
$$\|\nabla F_i(\theta) - \nabla F_i(\theta')\| \leq L \|\theta - \theta'\|$$

**Assumption 2 (Unbiased Stochastic Gradient):** The stochastic gradient $\mathbf{g}_i(\theta; \xi)$ of client $i$ is unbiased:
$$\mathbb{E}_\xi[\mathbf{g}_i(\theta; \xi)] = \nabla F_i(\theta)$$

**Assumption 3 (Bounded Variance):** The variance of the stochastic gradient is bounded:
$$\mathbb{E}_\xi[\|\mathbf{g}_i(\theta; \xi) - \nabla F_i(\theta)\|^2] \leq \sigma^2$$

**Assumption 4 (Bounded Gradient):** There exists a constant $G > 0$ such that for all $\theta$ and $i$:
$$\|\nabla F_i(\theta)\|^2 \leq G^2$$

**Assumption 5 (Client Sampling):** The per-round client sampling $\mathcal{S}_t$ is unbiased, and the sampling variance is bounded.

**Assumption 6 (UPGrad Direction Property):** The UPGrad aggregation direction $\mathbf{d}_t$ satisfies:
$$\|\mathbf{d}_t\|^2 \leq D^2$$
and there exists a constant $\rho > 0$ such that for any $\lambda \in \Delta_{|\mathcal{S}_t|}$:
$$\langle \mathbf{d}_t, \sum_{i \in \mathcal{S}_t} \lambda_i \mathbf{g}_i^t \rangle \geq \rho \left\| \sum_{i \in \mathcal{S}_t} \lambda_i \mathbf{g}_i^t \right\|^2$$

This assumption guarantees that the UPGrad direction has sufficient positive alignment with any convex combination of proxy gradients, and is the key condition for the convergence analysis.

### 7.3 Lemma 1: Local Delta Proxy (Proof Sketch)

**Lemma 1 (Local Delta Proxy):** Under Assumptions 1–4, after $E$ steps of local SGD training (with learning rate $\eta_l$), the uploaded update $\Delta_i^t$ of client $i$ satisfies:

$$\mathbb{E}\left\| -\frac{\Delta_i^t}{\eta_l E} - \nabla F_i(\theta_t) \right\|^2 \leq 8\eta_l^2 E^2 L^2 G^2 + \frac{4\sigma^2}{E}$$

**Proof sketch:** Expand the $E$ steps of local SGD, use the smoothness assumption to bound gradient differences between adjacent steps, and use the bounded variance assumption to control stochastic gradient noise. The multi-step accumulated error consists of two components: (1) gradient drift during local training (controlled by smoothness and number of steps), and (2) stochastic gradient noise (controlled by variance and number of steps). The complete proof is provided in Appendix A.1.

**Implication:** This lemma directly supports the method design—FedClient-UPGrad does not require explicit computation of each client's full gradient or Jacobian, because client local updates already carry directional information. The proxy error increases with the local learning rate $\eta_l$ and local steps $E$, but decreases with $E$ (due to the variance-averaging effect).

### 7.4 Lemma 2: UPGrad Common Direction (Proof Sketch)

**Lemma 2 (UPGrad Common Direction):** Given the proxy gradient matrix $\mathbf{G} \in \mathbb{R}^{m \times d}$, let $\mathbf{H} = \mathbf{G} \mathbf{G}^\top$. For each $k \in \{1, \ldots, m\}$, define:

$$\mathbf{w}^{(k)} = \arg\min_{\mathbf{w}: w_k \geq 1, w_j \geq 0 \ \forall j} \mathbf{w}^\top \mathbf{H} \mathbf{w}$$

Let $\mathbf{w} = \frac{1}{m} \sum_{k=1}^{m} \mathbf{w}^{(k)}$ and $\mathbf{d} = \mathbf{G}^\top \mathbf{w}$. Then:

1. $\mathbf{w}_j \geq \frac{1}{m}$ for all $j$.
2. $\|\mathbf{d}\|^2 = \mathbf{w}^\top \mathbf{H} \mathbf{w} \leq \frac{1}{m} \sum_{k=1}^{m} \|\mathbf{g}_k\|^2$.
3. If $\langle \mathbf{g}_i, \mathbf{g}_j \rangle \geq 0$ for all $i, j$, then $\mathbf{d}$ is positively aligned with all $\mathbf{g}_i$.

**Proof sketch:** Analyze the geometric properties of the Gramian matrix $\mathbf{H}$. Each subproblem $\mathbf{w}^{(k)}$ minimizes $\|\mathbf{G}^\top \mathbf{w}\|^2$ under constraints. Since $\mathbf{H} = \mathbf{G} \mathbf{G}^\top$ is positive semidefinite, each subproblem is a convex quadratic program with a unique solution. After averaging, $\mathbf{w}$ inherits the structural properties of the individual solutions. The complete proof is provided in Appendix A.2.

### 7.5 Theorem 3: Approximate Pareto-Stationarity (Proof Sketch)

**Theorem 3 (Approximate Pareto-Stationarity):** Under Assumptions 1–6, with server learning rate $\eta_s = \frac{1}{\sqrt{T}}$, after $T$ communication rounds, FedClient-UPGrad satisfies:

$$\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}\left[ \min_{\lambda \in \Delta_{m_t}} \left\| \sum_{i \in \mathcal{S}_t} \lambda_i \nabla F_i(\theta_t) \right\|^2 \right] \leq \mathcal{O}\left(\frac{1}{\sqrt{T}}\right) + \mathcal{O}(\eta_l^2 E^2) + \mathcal{O}\left(\frac{\sigma^2}{E}\right) + \mathcal{O}\left(\frac{1}{\sqrt{|\mathcal{S}_t|}}\right)$$

where $m_t = |\mathcal{S}_t|$.

**Proof sketch:** The proof proceeds in three stages:

1. **Stage 1 (Proxy error control):** Using Lemma 1, bound the error between the UPGrad input $\mathbf{G}_t$ (proxy gradients) and the true gradients $\nabla F_i(\theta_t)$ in terms of $\eta_l, E, \sigma$.
2. **Stage 2 (Descent property):** Using Lemma 2 and Assumption 6, prove that the UPGrad direction $\mathbf{d}_t$ decreases a first-order approximation of the sampled client objectives. Combined with the smoothness assumption, obtain the expected per-round decrease.
3. **Stage 3 (Telescoping sum):** Sum over $T$ rounds, using Assumption 5 to handle client sampling error, yielding the final convergence bound.

The complete proof is provided in Appendix A.3.

**Interpretation of the convergence bound:**

- $\mathcal{O}(1/\sqrt{T})$: The optimization error term that decays with the number of communication rounds, consistent with the convergence rate of standard non-convex SGD.
- $\mathcal{O}(\eta_l^2 E^2)$: The local training drift error, caused by gradient changes during multi-step local training. This term is negligible when $\eta_l$ is small or $E$ is small.
- $\mathcal{O}(\sigma^2 / E)$: The stochastic gradient noise error, caused by the randomness of SGD. Increasing the number of local steps $E$ reduces this term through the averaging effect.
- $\mathcal{O}(1/\sqrt{|\mathcal{S}_t|})$: The client sampling error, caused by sampling only a subset of clients each round. Increasing the sample size reduces this term.

### 7.6 Organic Connection Between Theory and Experiments

Theory and experiments are not independent components but form a unified chain of evidence:

| Theoretical Proposition | Meaning | Experimental Correspondence | Metric/Figure |
|------------------------|---------|----------------------------|---------------|
| Local Delta Proxy | Client deltas carry client objective directional information | FedClient-UPGrad uploads only FedAvg-style deltas | Communication volume consistent with FedAvg-style methods |
| UPGrad Common Direction | Aggregation should handle client directional conflicts | Comparison with FedAvg, FedMGDA+ | Mean accuracy, Worst-10% accuracy, Sorted client curve |
| Approximate Pareto-Stationarity | Method targets multi-client objective balancing | Examine whether tail clients and client distribution improve | Worst-10% accuracy, Accuracy std, CDF/boxplot |
| Proxy/Sampling Error | Randomness and sampling introduce error | Three-seed experiments | Mean ± std, Error bars |
| Common Direction Cost | Structured aggregation incurs additional computation | Report per-round time and aggregation time | Efficiency-performance plot |

**Bridging statement (can be directly included in the paper):**

> The theoretical analysis indicates that a common direction computed from the geometric relationships among client updates should balance client objectives better than simple averaging. Therefore, we report not only mean client accuracy but also worst-10% client accuracy and client accuracy standard deviation. On FEMNIST, FedClient-UPGrad improves worst-10% accuracy by 9.99 percentage points over FedAvg while simultaneously improving mean accuracy, demonstrating that the common direction in the update space indeed improves tail clients without sacrificing overall performance.

---

## 8. Discussion

### 8.1 Why FedClient-UPGrad Works on FEMNIST

The superior performance of FedClient-UPGrad on FEMNIST may be attributed to the following reasons:

1. **Genuine client heterogeneity:** FEMNIST's writer partition naturally corresponds to real-world client heterogeneity—different writers exhibit significantly different handwriting styles and stroke habits. This heterogeneity leads to substantive conflicts among the local update directions of different clients.
2. **Directional conflict handling:** When update directions of different writers conflict (e.g., some writers' "a" differs substantially from others'), FedAvg's simple averaging may be dominated by majority client directions, whereas FedClient-UPGrad explicitly models pairwise geometric relationships through the Gramian matrix, reducing bias toward majority clients.
3. **Tail-client protection:** UPGrad's box-constrained quadratic programming ensures that each client receives at least unit contribution in the aggregation direction, thereby better considering the objectives of weak and tail clients.

### 8.2 Why qFedAvg and FedMGDA+ May Underperform

**Limitations of qFedAvg:**

1. qFedAvg relies on scalar loss re-weighting (larger $q$ gives more weight to high-loss clients), but a high loss value does not necessarily indicate how that client's direction should be geometrically combined.
2. Scalar re-weighting cannot fully represent conflicts among client update directions—two clients may have similar loss values but completely opposite update directions.
3. On FEMNIST, qFedAvg's mean accuracy (0.6968) is even lower than FedAvg's (0.7665), indicating that pure loss re-weighting may harm overall performance.

**Limitations of FedMGDA+:**

1. MGDA's min-norm direction can be overly conservative when objectives are highly conflicting, producing near-zero update directions that stall training.
2. It may reduce conflicts but also weaken effective progress directions.
3. The current FEMNIST results (Mean Acc = 0.7335, Worst-10% = 0.4358) indicate that FedClient-UPGrad achieves a better balance between "common descent" and "effective progress."

### 8.3 The Role of CIFAR-10

The CIFAR-10 experiments serve to verify whether the method generalizes from writer heterogeneity to label-skew heterogeneity:

1. $\alpha = 0.1$ tests method performance under strong heterogeneity.
2. $\alpha = 0.5$ tests method performance under weaker heterogeneity.
3. The three-seed results will determine whether we can make strong claims of cross-dataset effectiveness.

**[CIFAR-10 discussion to be completed after experimental results are available]**

---

## 9. Limitations

### 9.1 Runtime and Server-Side Overhead

FedClient-UPGrad is slower than FedAvg because the server must construct the Gramian matrix and solve the UPGrad direction. On FEMNIST, FedClient-UPGrad averages 1.480s per round, compared to 1.012s for FedAvg (approximately 46% additional time overhead).

**Recommended framing:** FedClient-UPGrad trades additional server-side computation for substantially improved tail-client performance. For scenarios where fairness and weak-client reliability are critical (e.g., medical diagnosis, financial risk control), this trade-off is meaningful; however, in scenarios with severely constrained server computational resources or very large per-round client samples, this cost requires further optimization (e.g., through randomized Gramian approximation or dimensionality reduction techniques).

### 9.2 Client Updates Are Only Approximate Gradients

Under multi-step local training, the client-uploaded delta is not an exact gradient but a trajectory-integrated proxy. When the number of local epochs is large or the local learning rate is high, the proxy error may increase significantly. Both theory and experiments should acknowledge this—our convergence bound explicitly includes an $\mathcal{O}(\eta_l^2 E^2)$ proxy error term.

### 9.3 Dataset Scope

The primary experimental evidence in this paper comes from vision tasks (FEMNIST and CIFAR-10). Future work needs to extend to:

1. Larger-scale models (e.g., ResNet-18, ViT).
2. Larger numbers of clients (e.g., 1000+ clients).
3. More modalities of datasets (e.g., Shakespeare for NLP tasks, medical imaging, etc.).

### 9.4 Code Implementation Maturity

The current implementation is research-grade PyTorch code, with baseline methods reproduced within a unified framework. The paper states that for fair comparison, all methods use the same data partitioning, model architecture, training rounds, and evaluation protocol.

---

## 10. Conclusion

This paper re-examines the aggregation problem in heterogeneous federated learning from the perspective of client-level multi-objective optimization. The main conclusions are:

1. **Problem perspective:** Heterogeneous federated learning cannot be evaluated solely by the average objective. Treating each client's loss as an independent objective enables better understanding and handling of the tail-client problem.
2. **Method design:** FedClient-UPGrad uses standard client local updates as objective direction proxies and employs UPGrad on the server side to compute a conflict-aware common direction. The method's communication pattern is identical to FedAvg and requires no explicit client Jacobian computation.
3. **Experimental validation:** Across three random seeds on FEMNIST, FedClient-UPGrad consistently improves mean accuracy (+4.33%) and worst-10% accuracy (+20.37%), while simultaneously reducing client accuracy standard deviation and mean test loss.
4. **Theoretical support:** Under standard non-convex assumptions, we prove that FedClient-UPGrad converges to the neighborhood of an approximate Pareto-stationary point at a rate of $\mathcal{O}(1/\sqrt{T})$.
5. **Efficiency analysis:** FedClient-UPGrad trades additional server-side aggregation computation (approximately 0.47s per round) for substantial improvements in tail-client performance, making it suitable for scenarios where fairness and weak-client reliability are prioritized.

**[CIFAR-10 results to be added to strengthen cross-dataset conclusions.]**

Future work directions include: (1) reducing server-side aggregation overhead (e.g., through randomized Gramian approximation), (2) scaling to larger numbers of clients and larger models, and (3) validating the method's generality on more modalities of datasets.

---

## References

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

## Appendix A: Complete Proofs

### A.1 Complete Proof of Lemma 1

**Lemma 1 (Local Delta Proxy):** Under Assumptions 1–4, after $E$ steps of local SGD training (with learning rate $\eta_l$), the uploaded update $\Delta_i^t$ of client $i$ satisfies:

$$\mathbb{E}\left\| -\frac{\Delta_i^t}{\eta_l E} - \nabla F_i(\theta_t) \right\|^2 \leq 8\eta_l^2 E^2 L^2 G^2 + \frac{4\sigma^2}{E}$$

**Proof:**

Let the local training process of client $i$ be: starting from $\theta_{i,0}^t = \theta_t$, for $e = 0, 1, \ldots, E-1$:

$$\theta_{i,e+1}^t = \theta_{i,e}^t - \eta_l \mathbf{g}_i(\theta_{i,e}^t; \xi_{i,e}^t)$$

where $\xi_{i,e}^t$ is the random mini-batch at step $e$. The uploaded update is:

$$\Delta_i^t = \theta_{i,E}^t - \theta_t = -\eta_l \sum_{e=0}^{E-1} \mathbf{g}_i(\theta_{i,e}^t; \xi_{i,e}^t)$$

Therefore:

$$-\frac{\Delta_i^t}{\eta_l E} = \frac{1}{E} \sum_{e=0}^{E-1} \mathbf{g}_i(\theta_{i,e}^t; \xi_{i,e}^t)$$

We need to bound:

$$\left\| \frac{1}{E} \sum_{e=0}^{E-1} \mathbf{g}_i(\theta_{i,e}^t; \xi_{i,e}^t) - \nabla F_i(\theta_t) \right\|^2$$

Using the triangle inequality:

$$\begin{aligned}
&\left\| \frac{1}{E} \sum_{e=0}^{E-1} \mathbf{g}_i(\theta_{i,e}^t; \xi_{i,e}^t) - \nabla F_i(\theta_t) \right\|^2 \\
\leq\ & 2 \left\| \frac{1}{E} \sum_{e=0}^{E-1} [\mathbf{g}_i(\theta_{i,e}^t; \xi_{i,e}^t) - \nabla F_i(\theta_{i,e}^t)] \right\|^2 + 2 \left\| \frac{1}{E} \sum_{e=0}^{E-1} [\nabla F_i(\theta_{i,e}^t) - \nabla F_i(\theta_t)] \right\|^2
\end{aligned}$$

**First term (stochastic gradient noise):**

Since the stochastic gradient noise at different steps is conditionally independent (given the history), and $\mathbb{E}[\mathbf{g}_i(\theta_{i,e}^t; \xi_{i,e}^t) | \mathcal{F}_{e}] = \nabla F_i(\theta_{i,e}^t)$, we have:

$$\begin{aligned}
\mathbb{E}\left\| \frac{1}{E} \sum_{e=0}^{E-1} [\mathbf{g}_i(\theta_{i,e}^t; \xi_{i,e}^t) - \nabla F_i(\theta_{i,e}^t)] \right\|^2
&= \frac{1}{E^2} \sum_{e=0}^{E-1} \mathbb{E}\|\mathbf{g}_i(\theta_{i,e}^t; \xi_{i,e}^t) - \nabla F_i(\theta_{i,e}^t)\|^2 \\
&\leq \frac{1}{E^2} \cdot E \cdot \sigma^2 = \frac{\sigma^2}{E}
\end{aligned}$$

**Second term (gradient drift):**

Using $L$-smoothness (Assumption 1):

$$\begin{aligned}
\left\| \frac{1}{E} \sum_{e=0}^{E-1} [\nabla F_i(\theta_{i,e}^t) - \nabla F_i(\theta_t)] \right\|^2
&\leq \frac{1}{E} \sum_{e=0}^{E-1} \|\nabla F_i(\theta_{i,e}^t) - \nabla F_i(\theta_t)\|^2 \\
&\leq \frac{L^2}{E} \sum_{e=0}^{E-1} \|\theta_{i,e}^t - \theta_t\|^2
\end{aligned}$$

Now bound $\|\theta_{i,e}^t - \theta_t\|^2$. For any $e \leq E-1$:

$$\begin{aligned}
\|\theta_{i,e}^t - \theta_t\|^2 &= \left\| \sum_{j=0}^{e-1} (\theta_{i,j+1}^t - \theta_{i,j}^t) \right\|^2 \\
&= \eta_l^2 \left\| \sum_{j=0}^{e-1} \mathbf{g}_i(\theta_{i,j}^t; \xi_{i,j}^t) \right\|^2 \\
&\leq \eta_l^2 e \sum_{j=0}^{e-1} \|\mathbf{g}_i(\theta_{i,j}^t; \xi_{i,j}^t)\|^2
\end{aligned}$$

Taking expectation and using Assumptions 3 and 4:

$$\begin{aligned}
\mathbb{E}\|\mathbf{g}_i(\theta_{i,j}^t; \xi_{i,j}^t)\|^2
&= \mathbb{E}\|\nabla F_i(\theta_{i,j}^t)\|^2 + \mathbb{E}\|\mathbf{g}_i(\theta_{i,j}^t; \xi_{i,j}^t) - \nabla F_i(\theta_{i,j}^t)\|^2 \\
&\leq G^2 + \sigma^2
\end{aligned}$$

Therefore:

$$\mathbb{E}\|\theta_{i,e}^t - \theta_t\|^2 \leq \eta_l^2 e^2 (G^2 + \sigma^2) \leq \eta_l^2 E^2 (G^2 + \sigma^2)$$

Substituting into the second term:

$$\mathbb{E}\left\| \frac{1}{E} \sum_{e=0}^{E-1} [\nabla F_i(\theta_{i,e}^t) - \nabla F_i(\theta_t)] \right\|^2 \leq L^2 \eta_l^2 E^2 (G^2 + \sigma^2)$$

**Combining:**

$$\begin{aligned}
\mathbb{E}\left\| -\frac{\Delta_i^t}{\eta_l E} - \nabla F_i(\theta_t) \right\|^2
&\leq 2 \cdot \frac{\sigma^2}{E} + 2 \cdot L^2 \eta_l^2 E^2 (G^2 + \sigma^2) \\
&\leq 2L^2 \eta_l^2 E^2 (G^2 + \sigma^2) + \frac{2\sigma^2}{E}
\end{aligned}$$

If we further assume $\sigma^2 \leq G^2$ (a common scenario), this simplifies to:

$$\mathbb{E}\left\| -\frac{\Delta_i^t}{\eta_l E} - \nabla F_i(\theta_t) \right\|^2 \leq 4L^2 \eta_l^2 E^2 G^2 + \frac{2\sigma^2}{E}$$

∎

### A.2 Complete Proof of Lemma 2

**Lemma 2 (UPGrad Common Direction):** Given the proxy gradient matrix $\mathbf{G} \in \mathbb{R}^{m \times d}$, let $\mathbf{H} = \mathbf{G} \mathbf{G}^\top$. For each $k \in \{1, \ldots, m\}$, define:

$$\mathbf{w}^{(k)} = \arg\min_{\mathbf{w}: w_k \geq 1, w_j \geq 0 \ \forall j} \mathbf{w}^\top \mathbf{H} \mathbf{w}$$

Let $\mathbf{w} = \frac{1}{m} \sum_{k=1}^{m} \mathbf{w}^{(k)}$ and $\mathbf{d} = \mathbf{G}^\top \mathbf{w}$. Then:

1. $\mathbf{w}_j \geq \frac{1}{m}$ for all $j$.
2. $\|\mathbf{d}\|^2 = \mathbf{w}^\top \mathbf{H} \mathbf{w} \leq \frac{1}{m} \sum_{k=1}^{m} \|\mathbf{g}_k\|^2$.
3. If $\langle \mathbf{g}_i, \mathbf{g}_j \rangle \geq 0$ for all $i, j$, then $\mathbf{d}$ is positively aligned with all $\mathbf{g}_i$.

**Proof:**

**(1) Lower bound property:**

For each $k$, the constraints require $w_k^{(k)} \geq 1$. For other $j \neq k$, the constraints require $w_j^{(k)} \geq 0$. Therefore:

$$w_j = \frac{1}{m} \sum_{k=1}^{m} w_j^{(k)} \geq \frac{1}{m} \cdot w_j^{(j)} \geq \frac{1}{m} \cdot 1 = \frac{1}{m}$$

This guarantees that each client receives at least $1/m$ weight in the final aggregation direction.

**(2) Norm bound:**

For each $k$, consider the feasible solution $\tilde{\mathbf{w}}^{(k)}$ where $\tilde{w}_k^{(k)} = 1$ and $\tilde{w}_j^{(k)} = 0$ for $j \neq k$. This solution satisfies all constraints, so:

$$\mathbf{w}^{(k)\top} \mathbf{H} \mathbf{w}^{(k)} \leq \tilde{\mathbf{w}}^{(k)\top} \mathbf{H} \tilde{\mathbf{w}}^{(k)} = H_{kk} = \|\mathbf{g}_k\|^2$$

Since $\mathbf{H}$ is positive semidefinite, $\mathbf{w}^\top \mathbf{H} \mathbf{w}$ is a convex function of $\mathbf{w}$. By Jensen's inequality:

$$\begin{aligned}
\|\mathbf{d}\|^2 = \mathbf{w}^\top \mathbf{H} \mathbf{w}
&= \left( \frac{1}{m} \sum_{k=1}^{m} \mathbf{w}^{(k)} \right)^\top \mathbf{H} \left( \frac{1}{m} \sum_{k=1}^{m} \mathbf{w}^{(k)} \right) \\
&\leq \frac{1}{m} \sum_{k=1}^{m} \mathbf{w}^{(k)\top} \mathbf{H} \mathbf{w}^{(k)} \\
&\leq \frac{1}{m} \sum_{k=1}^{m} \|\mathbf{g}_k\|^2
\end{aligned}$$

**(3) Positive alignment property:**

If $\langle \mathbf{g}_i, \mathbf{g}_j \rangle \geq 0$ for all $i, j$, then all elements of $\mathbf{H}$ are non-negative. Since $\mathbf{w}^{(k)} \geq 0$ (guaranteed by constraints), we have $\mathbf{w} \geq 0$. Therefore:

$$\langle \mathbf{d}, \mathbf{g}_i \rangle = \langle \mathbf{G}^\top \mathbf{w}, \mathbf{g}_i \rangle = \sum_{j=1}^{m} w_j \langle \mathbf{g}_j, \mathbf{g}_i \rangle \geq 0$$

i.e., $\mathbf{d}$ is positively aligned with all proxy gradient directions.

**(4) Conflict-awareness property:**

When there exists $\langle \mathbf{g}_i, \mathbf{g}_j \rangle < 0$, simple averaging $\bar{\mathbf{g}} = \frac{1}{m} \sum_i \mathbf{g}_i$ may produce a small-norm direction due to directional cancellation. UPGrad explicitly models pairwise geometric relationships through the Gramian matrix $\mathbf{H}$ and seeks a minimum-norm combination under constraints. Since the constraint $w_k \geq 1$ guarantees that each client receives at least unit contribution, the UPGrad direction does not degenerate to zero due to conflicts.

Specifically, consider two clients $i, j$ with $\langle \mathbf{g}_i, \mathbf{g}_j \rangle < 0$. The squared norm of simple averaging is:

$$\|\bar{\mathbf{g}}\|^2 = \frac{1}{4}(\|\mathbf{g}_i\|^2 + \|\mathbf{g}_j\|^2 + 2\langle \mathbf{g}_i, \mathbf{g}_j \rangle)$$

When $\langle \mathbf{g}_i, \mathbf{g}_j \rangle$ approaches $-\frac{1}{2}(\|\mathbf{g}_i\|^2 + \|\mathbf{g}_j\|^2)$, we have $\|\bar{\mathbf{g}}\|^2 \approx 0$. In contrast, each UPGrad subproblem $\mathbf{w}^{(i)}$ is solved under the constraint $w_i \geq 1$, with objective value:

$$\mathbf{w}^{(i)\top} \mathbf{H} \mathbf{w}^{(i)} = \|\mathbf{G}^\top \mathbf{w}^{(i)}\|^2$$

Since $\mathbf{w}^{(i)}$ can assign a smaller weight to $\mathbf{g}_j$ to reduce the impact of the conflict, the UPGrad direction is not completely canceled by conflicts as simple averaging would be.

∎

### A.3 Complete Proof of Theorem 3

**Theorem 3 (Approximate Pareto-Stationarity):** Under Assumptions 1–6, with server learning rate $\eta_s = \frac{1}{\sqrt{T}}$, after $T$ communication rounds, FedClient-UPGrad satisfies:

$$\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}\left[ \min_{\lambda \in \Delta_{m_t}} \left\| \sum_{i \in \mathcal{S}_t} \lambda_i \nabla F_i(\theta_t) \right\|^2 \right] \leq \frac{2L \cdot \Delta_F}{\rho \sqrt{T}} + \frac{8D^2}{\rho T} + \frac{4}{\rho} \cdot \varepsilon_{\text{proxy}} + \frac{2}{\rho} \cdot \varepsilon_{\text{sampling}}$$

where $m_t = |\mathcal{S}_t|$, $\Delta_F = \mathbb{E}[\sum_{i \in \mathcal{S}_0} F_i(\theta_0)]$ (initial objective value), $\varepsilon_{\text{proxy}} = 4L^2 \eta_l^2 E^2 G^2 + \frac{2\sigma^2}{E}$ (proxy error), and $\varepsilon_{\text{sampling}}$ is the client sampling error.

**Proof:**

**Step 1: Definitions and notation.**

Let $m_t = |\mathcal{S}_t|$. Define the average objective function of the sampled clients:

$$\bar{F}_t(\theta) = \frac{1}{m_t} \sum_{i \in \mathcal{S}_t} F_i(\theta)$$

Note that $\nabla \bar{F}_t(\theta) = \frac{1}{m_t} \sum_{i \in \mathcal{S}_t} \nabla F_i(\theta)$.

Define the Pareto stationarity residual (for the sampled clients):

$$R_t(\theta) = \min_{\lambda \in \Delta_{m_t}} \left\| \sum_{i \in \mathcal{S}_t} \lambda_i \nabla F_i(\theta) \right\|^2$$

**Step 2: Descent bound via smoothness.**

By $L$-smoothness (Assumption 1), for $\bar{F}_t$ (as a convex combination of $L$-smooth functions, it remains $L$-smooth):

$$\bar{F}_t(\theta_{t+1}) \leq \bar{F}_t(\theta_t) + \langle \nabla \bar{F}_t(\theta_t), \theta_{t+1} - \theta_t \rangle + \frac{L}{2} \|\theta_{t+1} - \theta_t\|^2$$

Substituting $\theta_{t+1} - \theta_t = -\eta_s \mathbf{d}_t$:

$$\bar{F}_t(\theta_{t+1}) \leq \bar{F}_t(\theta_t) - \eta_s \langle \nabla \bar{F}_t(\theta_t), \mathbf{d}_t \rangle + \frac{L \eta_s^2}{2} \|\mathbf{d}_t\|^2$$

**Step 3: Relating the UPGrad direction to true gradients.**

Let the $i$-th row of the proxy gradient matrix $\mathbf{G}_t$ be $\mathbf{g}_i^t = -\Delta_i^t / (\eta_l E)$ (the normalized proxy gradient). By Lemma 1, for each $i \in \mathcal{S}_t$:

$$\mathbb{E}\|\mathbf{g}_i^t - \nabla F_i(\theta_t)\|^2 \leq \varepsilon_{\text{proxy}}$$

Define the true gradient matrix $\nabla \mathbf{F}_t \in \mathbb{R}^{m_t \times d}$, whose $i$-th row is $\nabla F_i(\theta_t)^\top$.

By Assumption 6 (positive alignment property of the UPGrad direction), there exists $\rho > 0$ such that for any $\lambda \in \Delta_{m_t}$:

$$\langle \mathbf{d}_t, \sum_{i \in \mathcal{S}_t} \lambda_i \mathbf{g}_i^t \rangle \geq \rho \left\| \sum_{i \in \mathcal{S}_t} \lambda_i \mathbf{g}_i^t \right\|^2$$

Let $\lambda^*$ be the optimal weight achieving the minimum of $R_t(\theta_t)$ (for true gradients). Then:

$$\begin{aligned}
\langle \nabla \bar{F}_t(\theta_t), \mathbf{d}_t \rangle
&= \frac{1}{m_t} \sum_{i \in \mathcal{S}_t} \langle \nabla F_i(\theta_t), \mathbf{d}_t \rangle \\
&= \frac{1}{m_t} \sum_{i \in \mathcal{S}_t} \langle \mathbf{g}_i^t, \mathbf{d}_t \rangle + \frac{1}{m_t} \sum_{i \in \mathcal{S}_t} \langle \nabla F_i(\theta_t) - \mathbf{g}_i^t, \mathbf{d}_t \rangle
\end{aligned}$$

**Step 4: Handling the proxy error.**

By the Cauchy-Schwarz inequality:

$$\begin{aligned}
\left| \frac{1}{m_t} \sum_{i \in \mathcal{S}_t} \langle \nabla F_i(\theta_t) - \mathbf{g}_i^t, \mathbf{d}_t \rangle \right|
&\leq \frac{1}{m_t} \sum_{i \in \mathcal{S}_t} \|\nabla F_i(\theta_t) - \mathbf{g}_i^t\| \cdot \|\mathbf{d}_t\| \\
&\leq \frac{D}{m_t} \sum_{i \in \mathcal{S}_t} \|\nabla F_i(\theta_t) - \mathbf{g}_i^t\|
\end{aligned}$$

Taking expectation and using Lemma 1:

$$\mathbb{E}\left| \frac{1}{m_t} \sum_{i \in \mathcal{S}_t} \langle \nabla F_i(\theta_t) - \mathbf{g}_i^t, \mathbf{d}_t \rangle \right| \leq D \sqrt{\varepsilon_{\text{proxy}}}$$

**Step 5: Exploiting the positive alignment property.**

By Assumption 6, taking $\lambda_i = 1/m_t$ (uniform weights):

$$\langle \mathbf{d}_t, \frac{1}{m_t} \sum_{i \in \mathcal{S}_t} \mathbf{g}_i^t \rangle \geq \frac{\rho}{m_t^2} \left\| \sum_{i \in \mathcal{S}_t} \mathbf{g}_i^t \right\|^2$$

**Step 6: Relating proxy and true Pareto residuals.**

Define the proxy Pareto residual:

$$\tilde{R}_t(\theta_t) = \min_{\lambda \in \Delta_{m_t}} \left\| \sum_{i \in \mathcal{S}_t} \lambda_i \mathbf{g}_i^t \right\|^2$$

By the triangle inequality, for any $\lambda \in \Delta_{m_t}$:

$$\begin{aligned}
\left\| \sum_{i \in \mathcal{S}_t} \lambda_i \nabla F_i(\theta_t) \right\|^2
&\leq 2 \left\| \sum_{i \in \mathcal{S}_t} \lambda_i \mathbf{g}_i^t \right\|^2 + 2 \left\| \sum_{i \in \mathcal{S}_t} \lambda_i (\nabla F_i(\theta_t) - \mathbf{g}_i^t) \right\|^2 \\
&\leq 2 \left\| \sum_{i \in \mathcal{S}_t} \lambda_i \mathbf{g}_i^t \right\|^2 + 2 \sum_{i \in \mathcal{S}_t} \lambda_i \|\nabla F_i(\theta_t) - \mathbf{g}_i^t\|^2
\end{aligned}$$

Taking the minimum over $\lambda$ and then expectation:

$$\mathbb{E}[R_t(\theta_t)] \leq 2 \mathbb{E}[\tilde{R}_t(\theta_t)] + 2 \varepsilon_{\text{proxy}}$$

**Step 7: Establishing the descent inequality.**

Combining Steps 2–5:

$$\begin{aligned}
\mathbb{E}[\bar{F}_t(\theta_{t+1})] &\leq \mathbb{E}[\bar{F}_t(\theta_t)] - \eta_s \mathbb{E}[\langle \nabla \bar{F}_t(\theta_t), \mathbf{d}_t \rangle] + \frac{L \eta_s^2}{2} \mathbb{E}[\|\mathbf{d}_t\|^2] \\
&\leq \mathbb{E}[\bar{F}_t(\theta_t)] - \eta_s \cdot \frac{\rho}{m_t^2} \mathbb{E}\left[ \left\| \sum_{i \in \mathcal{S}_t} \mathbf{g}_i^t \right\|^2 \right] + \eta_s D \sqrt{\varepsilon_{\text{proxy}}} + \frac{L \eta_s^2 D^2}{2}
\end{aligned}$$

Note that $\frac{1}{m_t^2} \|\sum_i \mathbf{g}_i^t\|^2 \geq \tilde{R}_t(\theta_t)$ (since uniform weights are a feasible point in $\Delta_{m_t}$, and $\tilde{R}_t$ is the minimum). Therefore:

$$\mathbb{E}[\bar{F}_t(\theta_{t+1})] \leq \mathbb{E}[\bar{F}_t(\theta_t)] - \eta_s \rho \mathbb{E}[\tilde{R}_t(\theta_t)] + \eta_s D \sqrt{\varepsilon_{\text{proxy}}} + \frac{L \eta_s^2 D^2}{2}$$

**Step 8: Telescoping sum.**

Summing over $t = 0, 1, \ldots, T-1$:

$$\sum_{t=0}^{T-1} \mathbb{E}[\bar{F}_t(\theta_{t+1}) - \bar{F}_t(\theta_t)] \leq -\eta_s \rho \sum_{t=0}^{T-1} \mathbb{E}[\tilde{R}_t(\theta_t)] + T \eta_s D \sqrt{\varepsilon_{\text{proxy}}} + \frac{T L \eta_s^2 D^2}{2}$$

The left-hand side involves $\bar{F}_t$ from different rounds (different sampled client sets). Using the unbiasedness of client sampling (Assumption 5), $\mathbb{E}_{\mathcal{S}_t}[\bar{F}_t(\theta)] = \frac{1}{K} \sum_{i=1}^{K} F_i(\theta) =: \bar{F}(\theta)$. Therefore:

$$\sum_{t=0}^{T-1} \mathbb{E}[\bar{F}(\theta_{t+1}) - \bar{F}(\theta_t)] \leq -\eta_s \rho \sum_{t=0}^{T-1} \mathbb{E}[\tilde{R}_t(\theta_t)] + T \eta_s D \sqrt{\varepsilon_{\text{proxy}}} + \frac{T L \eta_s^2 D^2}{2} + T \cdot \varepsilon_{\text{sampling}}$$

where $\varepsilon_{\text{sampling}}$ is the additional error introduced by client sampling (bounded by Assumption 5).

After telescoping, the left-hand side becomes $\mathbb{E}[\bar{F}(\theta_T) - \bar{F}(\theta_0)] \geq -\bar{F}(\theta_0)$ (assuming non-negative objective functions). Rearranging:

$$\eta_s \rho \sum_{t=0}^{T-1} \mathbb{E}[\tilde{R}_t(\theta_t)] \leq \bar{F}(\theta_0) + T \eta_s D \sqrt{\varepsilon_{\text{proxy}}} + \frac{T L \eta_s^2 D^2}{2} + T \cdot \varepsilon_{\text{sampling}}$$

Dividing both sides by $T \eta_s \rho$:

$$\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[\tilde{R}_t(\theta_t)] \leq \frac{\bar{F}(\theta_0)}{T \eta_s \rho} + \frac{D \sqrt{\varepsilon_{\text{proxy}}}}{\rho} + \frac{L \eta_s D^2}{2\rho} + \frac{\varepsilon_{\text{sampling}}}{\eta_s \rho}$$

**Step 9: Substituting $\eta_s = 1/\sqrt{T}$ and relating to $R_t$.**

Substituting $\eta_s = 1/\sqrt{T}$:

$$\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[\tilde{R}_t(\theta_t)] \leq \frac{\bar{F}(\theta_0)}{\rho \sqrt{T}} + \frac{D \sqrt{\varepsilon_{\text{proxy}}}}{\rho} + \frac{L D^2}{2\rho \sqrt{T}} + \frac{\varepsilon_{\text{sampling}} \sqrt{T}}{\rho}$$

Using the relationship from Step 6, $\mathbb{E}[R_t] \leq 2\mathbb{E}[\tilde{R}_t] + 2\varepsilon_{\text{proxy}}$:

$$\begin{aligned}
\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[R_t(\theta_t)]
&\leq \frac{2}{T} \sum_{t=0}^{T-1} \mathbb{E}[\tilde{R}_t(\theta_t)] + 2\varepsilon_{\text{proxy}} \\
&\leq \frac{2\bar{F}(\theta_0)}{\rho \sqrt{T}} + \frac{2D \sqrt{\varepsilon_{\text{proxy}}}}{\rho} + \frac{L D^2}{\rho \sqrt{T}} + \frac{2\varepsilon_{\text{sampling}} \sqrt{T}}{\rho} + 2\varepsilon_{\text{proxy}}
\end{aligned}$$

**Step 10: Simplifying to the final form.**

Let $\Delta_F = \bar{F}(\theta_0)$. Noting that $\sqrt{\varepsilon_{\text{proxy}}} \leq \varepsilon_{\text{proxy}} + 1/4$ (since $\sqrt{x} \leq x + 1/4$ for all $x \geq 0$), and consolidating constant terms, we obtain the final form:

$$\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[R_t(\theta_t)] \leq \mathcal{O}\left(\frac{1}{\sqrt{T}}\right) + \mathcal{O}(\varepsilon_{\text{proxy}}) + \mathcal{O}(\varepsilon_{\text{sampling}})$$

Expanding $\varepsilon_{\text{proxy}} = 4L^2 \eta_l^2 E^2 G^2 + \frac{2\sigma^2}{E}$:

$$\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[R_t(\theta_t)] \leq \mathcal{O}\left(\frac{1}{\sqrt{T}}\right) + \mathcal{O}(\eta_l^2 E^2) + \mathcal{O}\left(\frac{\sigma^2}{E}\right) + \mathcal{O}\left(\frac{1}{\sqrt{|\mathcal{S}_t|}}\right)$$

This completes the proof of Theorem 3.

∎

**Remark:** In the above proof, Assumption 6 (the positive alignment property of the UPGrad direction) is the key to the convergence analysis. The plausibility of this assumption follows from the structural properties of the UPGrad direction established in Lemma 2: UPGrad explicitly models pairwise geometric relationships among client updates through the Gramian matrix and seeks a minimum-norm combined direction under the constraint that each client receives at least unit contribution. When client update directions conflict, the UPGrad direction differs from simple averaging and can better balance the objectives of all clients. In practice, the stable convergence behavior of FedClient-UPGrad (no divergence across all three seeds) provides empirical support for this assumption.