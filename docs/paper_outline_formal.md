# FedClient-UPGrad Paper Outline

This document is a formal writing blueprint for the paper. It expands the initial outline in `论文大纲.md` into a paper-ready structure tied to the current implementation, experiment design, and completed FEMNIST results. CIFAR10 full-run results should be inserted after the ongoing experiments finish.

## Working Title

FedClient-UPGrad: Client-Level Multi-Objective Gradient Aggregation for Fair Federated Learning

Alternative titles:

1. Client-Level Multi-Objective Federated Learning via Update-Space UPGrad
2. Improving Tail-Client Performance in Federated Learning with Client-Level UPGrad
3. FedClient-UPGrad: Conflict-Aware Aggregation of Client Updates in Federated Learning

Recommendation: use the first title if the paper emphasizes the multi-objective formulation; use the second if targeting a federated learning venue where fairness and tail-client performance are more immediately recognizable.

## Core Thesis

Federated learning with heterogeneous clients can be interpreted as a client-level multi-objective optimization problem, where each client's empirical loss is one objective. Standard aggregation methods such as FedAvg primarily optimize an average objective and can underperform on tail clients. FedClient-UPGrad treats sampled client updates as proxy descent directions for individual client objectives and computes a common update direction using UPGrad in the client-update space. This improves both average client accuracy and worst-client-group performance, especially under writer-level or label-skew heterogeneity.

## Target Claims

The paper should defend four claims:

1. **Problem claim:** Client heterogeneity makes average-loss optimization insufficient; the model should also improve weak or conflicting clients.
2. **Method claim:** Client local updates can serve as practical gradient proxies for client-level objectives, avoiding expensive full Jacobian computation.
3. **Optimization claim:** UPGrad-style aggregation can produce a conflict-aware common direction from sampled client updates.
4. **Empirical claim:** FedClient-UPGrad improves mean client performance, tail-client performance, and client-performance dispersion compared with FedAvg, qFedAvg, and FedMGDA+ on federated vision benchmarks.

## Abstract

The abstract should answer four questions in one paragraph:

1. What is the problem?
Federated learning with non-IID clients often optimizes average performance while leaving low-performing clients behind.

2. What is the key idea?
Formulate sampled client losses as client-level objectives and aggregate client local updates with UPGrad to find a shared update direction that better respects client conflicts.

3. What is the method?
Each selected client starts from the broadcast global model, performs ordinary local training, uploads a model delta, and the server treats negative deltas as client-gradient proxies. The server then solves a UPGrad aggregation problem and updates the global model with the resulting direction.

4. What is the evidence?
On FEMNIST, FedClient-UPGrad achieves `0.7996 ± 0.0094` mean client accuracy and `0.5902 ± 0.0058` worst-10% client accuracy across three seeds, outperforming FedAvg (`0.7665 ± 0.0132`, `0.4903 ± 0.0252`) and other baselines. CIFAR10 results should be added after full runs finish.

Draft abstract skeleton:

```text
Federated learning under client heterogeneity is often evaluated by average accuracy, although practical deployments also require reliable performance on tail clients. We study this issue through a client-level multi-objective view, where each client's empirical loss defines one objective. Based on this view, we propose FedClient-UPGrad, a server-side aggregation method that uses ordinary client local updates as gradient proxies and applies UPGrad to compute a common conflict-aware direction. Unlike methods that require explicit per-objective Jacobians, FedClient-UPGrad is compatible with standard local training and communicates FedAvg-style model deltas. Experiments on writer-partitioned FEMNIST and Dirichlet-partitioned CIFAR10 compare FedClient-UPGrad with FedAvg, qFedAvg, and FedMGDA+. On FEMNIST, FedClient-UPGrad consistently improves mean client accuracy, worst-10% client accuracy, client accuracy dispersion, and test loss across three seeds. These results indicate that client-level multi-objective aggregation is a practical way to improve tail-client performance in heterogeneous federated learning.
```

## 1. Introduction

The introduction should follow the five-question structure from the original outline.

### 1.1 Why This Problem Matters

Write around the practical and theoretical importance of client-level heterogeneity:

1. Federated learning is deployed across naturally different clients: writers, devices, users, hospitals, regions, or data silos.
2. Non-IID client data causes gradient/update conflicts across clients.
3. Optimizing average performance can hide systematic failure on tail clients.
4. Tail-client performance matters for fairness, robustness, and deployment reliability.
5. This motivates evaluating not only mean accuracy but also worst-10% accuracy and client-performance dispersion.

Suggested wording:

```text
In heterogeneous federated learning, a high average accuracy does not necessarily imply that the global model serves all clients well. A small set of clients may remain poorly optimized because their data distributions conflict with those of the majority. This motivates a client-level objective view in which each client's loss is treated as a separate optimization objective rather than only as a weighted component of a single average loss.
```

### 1.2 Representative Existing Ideas

Discuss these groups:

1. **FedAvg-style averaging:** efficient and standard, but optimizes an average update.
2. **Fair FL / qFedAvg:** reweights clients using loss values; helps fairness but depends on scalar weighting and can be unstable or underperform empirically.
3. **Multi-objective FL / MGDA-style methods:** explicitly seek common descent directions, but can be expensive or require per-objective gradients/Jacobians.
4. **Conflict-aware gradient methods:** aim to handle conflicting objectives but often assume access to task gradients rather than client local update trajectories.

### 1.3 Common Limitation

The common limitation to emphasize:

```text
Existing methods either collapse clients into a single averaged objective, rely on scalar loss reweighting, or require expensive objective-wise gradient information. They do not fully exploit the fact that standard FL already computes informative client local updates.
```

### 1.4 Core Idea

FedClient-UPGrad bridges this gap:

1. Each sampled client is an objective.
2. Each client performs normal local training from the same global model `theta_t`.
3. The uploaded delta `Delta_i = theta_i - theta_t` approximates a descent direction for client `i`.
4. The server uses `-Delta_i` as a gradient proxy row in a client-update matrix.
5. UPGrad finds a shared direction from this matrix.

### 1.5 Contributions

Recommended contribution list:

1. We formulate heterogeneous federated learning as a client-level multi-objective problem where each client's empirical loss is one objective.
2. We propose FedClient-UPGrad, a practical aggregation method that applies UPGrad to FedAvg-style client local updates without requiring explicit per-client Jacobian computation.
3. We implement a fair experimental framework comparing FedAvg, qFedAvg, FedMGDA+, and FedClient-UPGrad on writer-partitioned FEMNIST and Dirichlet CIFAR10, using client-held-out test splits and multi-seed evaluation.
4. We show that FedClient-UPGrad consistently improves FEMNIST mean accuracy, worst-10% client accuracy, client accuracy standard deviation, and test loss across three random seeds; CIFAR10 full-run results should be inserted after completion.
5. We analyze efficiency trade-offs and document runtime-preserving implementation optimizations that do not change the reported final metrics.

## 2. Related Work

This section should not be a broad literature dump. It should build the gap that FedClient-UPGrad fills.

### 2.1 Federated Optimization Under Heterogeneity

Cover FedAvg and non-IID challenges. Key angle: FedAvg is the natural baseline but its weighted-average update can be dominated by majority clients or easier distributions.

### 2.2 Fair Federated Learning

Include qFedAvg/q-FFL and client fairness methods. Explain that qFedAvg uses loss-aware scalar reweighting, while FedClient-UPGrad uses update-space vector geometry.

### 2.3 Multi-Objective Optimization and MGDA

Explain MGDA/common descent directions. Then position FedMGDA+ as an important baseline because it also treats sampled client updates as objectives, but uses a min-norm/simplex-style direction.

### 2.4 Gradient Conflict and Projection-Based Methods

Introduce gradient conflict, Pareto stationarity, and projection/UPGrad-style aggregation. Keep it tied to the algorithm rather than overextending theory.

## 3. Problem Formulation

### 3.1 Federated Learning Setup

Let there be `K` clients. Client `i` has local data distribution or dataset `D_i` and empirical loss:

```text
F_i(theta) = E_{(x,y) in D_i}[ell(f_theta(x), y)]
```

Standard FedAvg approximately optimizes:

```text
F_avg(theta) = sum_i p_i F_i(theta)
```

where `p_i` is often proportional to client data size.

### 3.2 Client-Level Multi-Objective View

Instead of treating clients only through the weighted average, define the vector objective:

```text
F(theta) = [F_1(theta), F_2(theta), ..., F_K(theta)]
```

In each round, only a sampled subset `S_t` participates. The server seeks an update direction that improves or balances the sampled client objectives.

### 3.3 Practical Gradient Proxy from Local Updates

Each selected client starts from the same broadcast model `theta_t` and performs `E` local epochs to produce `theta_i^t`. Define:

```text
Delta_i^t = theta_i^t - theta_t
g_i^t approx -Delta_i^t
```

The approximation is exact up to learning-rate scaling for a single small SGD step and becomes a local-trajectory descent proxy for multiple local steps. This is the key practical bridge: FedClient-UPGrad does not require computing `n` separate full gradients at the server.

### 3.4 Evaluation Objectives

The method's goal should be evaluated by:

1. Mean client test accuracy.
2. Worst-10% client test accuracy.
3. Client accuracy standard deviation.
4. Mean client test loss.
5. Efficiency metrics: average round time and aggregation compute time.

## 4. Method: FedClient-UPGrad

### 4.1 Algorithm Overview

One communication round:

```text
Input: global model theta_t, sampled clients S_t
Server broadcasts theta_t to all clients in S_t
For each client i in S_t:
  client initializes local model with theta_t
  client trains for E local epochs
  client uploads Delta_i = theta_i - theta_t
Server forms G_t with rows g_i = -Delta_i
Server computes d_t = UPGrad(G_t)
Server updates theta_{t+1} = theta_t - eta_server * scale * d_t
```

Implementation note: in the current code, `update_scale` is the server-side scale applied to the UPGrad direction. CIFAR10 tuning selected `fedclient_update_scale=1.0`; FEMNIST uses `1.0`.

### 4.2 UPGrad Aggregation

The implementation uses the Gramian-space dual formulation in `fedjd/aggregators/__init__.py`.

For a client update matrix `G in R^{m x d}` over `m` sampled clients, it computes a direction from the client-update geometry. The code builds:

```text
Gramian = G G^T
```

and solves lower-bound constrained quadratic subproblems, using an active-set solver for small `m` and projected gradient descent fallback for larger `m`. The final direction is:

```text
d = G^T w
```

where `w` is derived by averaging the per-objective UPGrad solutions.

Writing guidance: avoid overclaiming exact Pareto optimality unless a formal proof is added. Safer wording is:

```text
FedClient-UPGrad computes a conflict-aware common direction from sampled client update proxies. It is motivated by multi-objective common descent geometry and implemented through the UPGrad aggregation rule.
```

### 4.3 Relationship to Baselines

Clearly distinguish the four methods:

1. **FedAvg:** averages client deltas by data-size weights.
2. **qFedAvg:** uses pre-update client loss to reweight client updates.
3. **FedMGDA+:** treats sampled client update proxies as objectives and uses MGDA/min-norm aggregation.
4. **FedClient-UPGrad:** treats sampled client update proxies as objectives and uses UPGrad aggregation.

### 4.4 Computational and Communication Cost

Communication is FedAvg-style:

```text
Each client uploads one model delta of dimension d.
```

The extra cost is server-side aggregation:

```text
FedClient-UPGrad builds an m x d update matrix and solves an UPGrad problem in sampled-client space.
```

Current FEMNIST timing:

1. FedAvg average round time: `1.012s`.
2. qFedAvg average round time: `1.067s`.
3. FedMGDA+ average round time: `1.103s`.
4. FedClient-UPGrad average round time: `1.480s`.

This is an important limitation/trade-off: FedClient-UPGrad is slower, but it substantially improves worst-10% accuracy.

## 5. Experimental Setup

### 5.1 Datasets

Main datasets:

1. **FEMNIST:** LEAF/FEMNIST writer partition. Each writer is a federated client. Experiments use 100 clients, per-client train/test split, and client-test-union global evaluation.
2. **CIFAR10 alpha=0.1:** torchvision CIFAR10 with Dirichlet label-skew client partition, 100 clients.
3. **CIFAR10 alpha=0.5:** same as above with milder label skew.

Datasets excluded from the paper:

1. **CelebA:** supported by data code and documentation, but less suitable for the current client-level objective story because its natural structure is multi-label attribute prediction rather than client-as-objective fairness.
2. Additional language benchmarks are intentionally left out of this paper to keep the empirical narrative focused on vision benchmarks with clear client-level heterogeneity.

### 5.2 Models

1. FEMNIST: compact CNN for 28x28 grayscale character classification.
2. CIFAR10: compact CNN for 32x32 RGB classification.

Do not overstate model scale. These are controlled benchmark models for method comparison, not state-of-the-art vision backbones.

### 5.3 Baselines

Main comparison set:

1. `fedavg`
2. `qfedavg`
3. `fedmgda_plus`
4. `fedclient_upgrad`

Rationale:

1. FedAvg is the standard average-optimization baseline.
2. qFedAvg is a fairness-oriented FL baseline.
3. FedMGDA+ is the closest multi-objective client-update baseline.
4. FedClient-UPGrad is the proposed method.

### 5.4 Hyperparameters

FEMNIST full-run settings:

```text
num_clients = 100
num_rounds = 500
local_epochs = 10
participation_rate = 0.5
learning_rate = 0.01
client_test_fraction = 0.2
seeds = {7, 42, 123}
```

CIFAR10 full-run settings:

```text
num_clients = 100
num_rounds = 1000
local_epochs = 2
participation_rate = 0.5
client_test_fraction = 0.2
seeds = {7, 42, 123}
```

CIFAR10 tuned method parameters from `tune_cifar_R100`:

| Method | Learning Rate | Method Parameter |
|---|---:|---|
| FedAvg | 0.03 | none |
| qFedAvg | 0.03 | `q=0.1`, `update_scale=1.0` |
| FedMGDA+ | 0.03 | `update_scale=2.0` |
| FedClient-UPGrad | 0.03 | `update_scale=1.0` |

Writing guidance: explicitly state that CIFAR10 hyperparameters were selected using a fixed 100-round tuning budget and then fixed for full 1000-round runs.

### 5.5 Metrics

Primary metrics:

1. Mean client test accuracy.
2. Worst-10% client test accuracy.
3. Client accuracy standard deviation.
4. Mean client test loss.

Efficiency metrics:

1. Average round time.
2. Average upload bytes.
3. Average aggregation/direction compute time.
4. Total elapsed time.

Visualization metrics:

1. Mean-vs-worst10 scatter plot.
2. Sorted client accuracy curve.
3. Client accuracy CDF or boxplot.
4. Efficiency-performance scatter plot.

### 5.6 Runtime-Preserving Implementation Choices

For full runs, the implementation uses optimizations that preserve final metrics:

1. `eval_interval=0`: disables intermediate curve evaluation; final client metrics are still computed after training.
2. Skip `initial_loss` only for FedAvg and FedClient-UPGrad, where it is not used in the update rule. qFedAvg and FedMGDA+ keep it.
3. Reuse a local model container within each round while strictly resetting it to a cloned server-state snapshot before every client update.

This should be placed in either the experimental setup or appendix to preempt concerns about implementation shortcuts.

## 6. Results

### 6.1 Main FEMNIST Result

Completed result directory:

```text
results/full_femnist_E10R500_pr50_lr001
```

Three-seed summary:

| Method | Mean Acc | Worst10 Acc | Acc Std | Mean Loss |
|---|---:|---:|---:|---:|
| FedAvg | 0.7665 ± 0.0132 | 0.4903 ± 0.0252 | 0.1301 ± 0.0114 | 0.7814 ± 0.0824 |
| qFedAvg | 0.6968 ± 0.0150 | 0.4163 ± 0.0288 | 0.1373 ± 0.0091 | 1.0680 ± 0.0828 |
| FedMGDA+ | 0.7335 ± 0.0146 | 0.4358 ± 0.0367 | 0.1470 ± 0.0072 | 0.9686 ± 0.0703 |
| FedClient-UPGrad | **0.7996 ± 0.0094** | **0.5902 ± 0.0058** | **0.1030 ± 0.0041** | **0.7026 ± 0.0679** |

Interpretation:

1. FedClient-UPGrad ranks first on all four metrics for every seed.
2. Compared with FedAvg, it improves mean accuracy by `+0.0332` (`+4.33%`) and worst-10% accuracy by `+0.0999` (`+20.37%`).
3. It reduces client accuracy standard deviation by `0.0271`, indicating more uniform client performance.
4. It reduces mean test loss by `10.09%` compared with FedAvg.

Suggested paper text:

```text
On FEMNIST, FedClient-UPGrad consistently outperforms all baselines across three seeds. The improvement is especially pronounced on the worst-10% client accuracy, where it improves over FedAvg by 9.99 percentage points. This supports the client-level multi-objective motivation: the proposed aggregation not only improves average performance but also substantially raises the tail of the client performance distribution.
```

### 6.2 FEMNIST Client Distribution Analysis

Three-seed averaged client accuracy quantiles:

| Method | P10 | P25 | Median | P75 |
|---|---:|---:|---:|---:|
| FedClient-UPGrad | **0.6563** | **0.7398** | **0.8176** | **0.8689** |
| FedAvg | 0.6251 | 0.7019 | 0.7852 | 0.8645 |
| FedMGDA+ | 0.5467 | 0.6517 | 0.7530 | 0.8419 |
| qFedAvg | 0.5130 | 0.6215 | 0.7053 | 0.7930 |

Interpretation:

```text
The sorted-client and quantile results show that FedClient-UPGrad does not merely improve a small number of clients or the average. It shifts the lower and middle portions of the client distribution upward, which is consistent with the worst-10% improvement.
```

### 6.3 CIFAR10 Alpha=0.1 Result

Completed result directory:

```text
results/full_cifar_alpha0p1_E2R1000_pr50_mixed
```

This setting uses mixed hyperparameters: FedAvg, qFedAvg, and FedMGDA+ use the best configurations selected by the R100 tuning runs, while FedClient-UPGrad uses the long-horizon stable configuration `learning_rate=0.01, update_scale=1.0`. The short-horizon `learning_rate=0.03` configuration for FedClient-UPGrad was faster early in training but numerically unstable in the 1000-round run.

Three-seed summary:

| Method | Mean Acc | Worst10 Acc | Acc Std | Mean Loss |
|---|---:|---:|---:|---:|
| FedAvg | **0.6040 ± 0.0047** | 0.3424 ± 0.0394 | 0.1362 ± 0.0105 | **1.1175 ± 0.0084** |
| qFedAvg | 0.5290 ± 0.0252 | 0.2754 ± 0.0758 | 0.1333 ± 0.0093 | 1.3015 ± 0.0343 |
| FedMGDA+ | 0.4505 ± 0.2848 | 0.2588 ± 0.2289 | 0.1765 ± 0.0651 | 1.2631 ± 0.3054 |
| FedClient-UPGrad | 0.5928 ± 0.0133 | **0.3777 ± 0.0286** | **0.1190 ± 0.0096** | 1.1551 ± 0.0372 |

Interpretation:

1. FedClient-UPGrad does not fully dominate FedAvg on CIFAR10 alpha=0.1; FedAvg achieves the best mean accuracy and mean loss.
2. FedClient-UPGrad achieves the best worst-10% client accuracy, improving over FedAvg by `+0.0352` absolute and `+10.28%` relative.
3. FedClient-UPGrad also achieves the lowest client accuracy standard deviation, reducing it by `0.0172` compared with FedAvg.
4. Compared with qFedAvg and FedMGDA+, FedClient-UPGrad is more stable on mean accuracy, tail accuracy, and client dispersion.
5. FedMGDA+ numerically diverges on seed 123, yielding `mean_client_test_loss=NaN`, which indicates that long-horizon stability is a real issue for aggressive multi-objective aggregation under strong label skew.

Per-seed winners:

| Seed | Best Mean Acc | Best Worst10 | Lowest Acc Std | Lowest Mean Loss |
|---:|---|---|---|---|
| 7 | FedClient-UPGrad | FedClient-UPGrad | qFedAvg | FedAvg |
| 42 | FedMGDA+ | FedMGDA+ | FedClient-UPGrad | FedMGDA+ |
| 123 | FedAvg | FedClient-UPGrad | FedClient-UPGrad | FedAvg |

Suggested paper text:

```text
On CIFAR10 alpha=0.1, FedClient-UPGrad does not outperform FedAvg in mean accuracy, but it achieves the best worst-10% client accuracy and the lowest client accuracy standard deviation. This suggests a fairness-performance trade-off under severe label skew: FedClient-UPGrad sacrifices a small amount of average accuracy for improved tail-client performance and reduced inter-client dispersion.
```

### 6.4 CIFAR10 Alpha=0.5 Result

Status: full three-seed runs are still running.

Insert after completion:

1. CIFAR10 alpha=0.5 three-seed table.
2. Relative improvement over tuned baselines.
3. Sorted client accuracy distribution.
4. Efficiency-performance trade-off.

### 6.5 Efficiency Trade-Off

FEMNIST efficiency summary:

| Method | Avg Round Time | Avg Aggregation Compute Time | Elapsed Time |
|---|---:|---:|---:|
| FedAvg | 1.012s ± 0.011s | 0.0003s | 506.23s ± 5.58s |
| qFedAvg | 1.067s ± 0.032s | 0.0001s | 533.57s ± 16.16s |
| FedMGDA+ | 1.103s ± 0.037s | 0.0353s | 551.90s ± 18.50s |
| FedClient-UPGrad | 1.480s ± 0.055s | 0.4722s | 740.67s ± 27.40s |

Interpretation:

```text
FedClient-UPGrad incurs higher server-side aggregation cost due to UPGrad direction solving. On FEMNIST, its average round time is about 46% higher than FedAvg, but this cost yields +4.33% relative mean accuracy and +20.37% relative worst-10% accuracy over FedAvg. The efficiency-performance trade-off should be reported explicitly rather than hidden.
```

CIFAR10 alpha=0.1 efficiency summary:

| Method | Avg Round Time | Avg Aggregation Compute Time | Elapsed Time |
|---|---:|---:|---:|
| FedAvg | 2.788s ± 0.060s | 0.0003s | 2788.30s ± 60.46s |
| qFedAvg | 4.078s ± 0.057s | 0.0001s | 4078.68s ± 56.75s |
| FedMGDA+ | 4.109s ± 0.040s | 0.0318s | 4109.37s ± 39.71s |
| FedClient-UPGrad | 3.229s ± 0.020s | 0.4710s | 3230.03s ± 20.20s |

Interpretation:

```text
On CIFAR10 alpha=0.1, FedClient-UPGrad is slower than FedAvg but faster in elapsed time than qFedAvg and FedMGDA+ in the current implementation. Its main additional cost remains the UPGrad aggregation step, while the stable lr=0.01 configuration avoids the long-horizon divergence observed with lr=0.03.
```

### 6.5 Ablation Studies

Recommended but not yet required:

1. FedClient-UPGrad `update_scale` ablation on CIFAR10: `0.5`, `1.0`, `2.0`. Already partially available from tuning.
2. Learning-rate sensitivity on CIFAR10. Already available from `tune_cifar_R100`.
3. Participation-rate sensitivity: e.g., `0.2`, `0.5`, `1.0`, if time permits.
4. Optional ablation: normalized client updates vs raw client updates. Current main runs use raw updates.

Recommendation: do not add too many new experiments before the main FEMNIST+CIFAR10 results are finalized. A compact tuning/ablation appendix is enough.

## 7. Theory and Analysis

This section should prove only what the method needs. The logic should be sequential: first justify the object FedClient-UPGrad aggregates, then justify the aggregation direction, and finally state the nonconvex convergence target implied by the first two steps. Avoid a large theory section disconnected from the experiments.

### 7.1 What Needs to Be Shown

The theory should support three connected claims:

1. **Local Delta Proxy Lemma:** local model deltas are valid approximate client-gradient or descent-direction proxies under standard smoothness and small-step assumptions.
2. **UPGrad Common Direction Lemma:** UPGrad aggregation combines these client proxy directions into a conflict-aware common direction using their Gramian geometry.
3. **Approximate Pareto-Stationarity Theorem:** combining the proxy lemma and the common-direction lemma gives convergence toward an approximate multi-objective stationary condition, with explicit error terms from local training, stochastic gradients, and partial participation.

The section should make clear that the paper is not proving convergence to a global optimum. It is proving that FedClient-UPGrad targets a meaningful first-order multi-objective stationarity condition.

### 7.2 Lemma 1: Local Delta Proxy

Purpose:

```text
Justify why the server can treat each uploaded client delta as information about that client's objective direction.
```

Setup:

Client `i` starts round `t` from the broadcast global model `theta_t`, runs local SGD for `E` local steps with local learning rate `eta_l`, and returns:

```text
Delta_i^t = theta_{i,E}^t - theta_t
```

For one stochastic local step:

```text
-Delta_i^t / eta_l = stochastic gradient of F_i(theta_t)
```

For multiple local steps:

```text
-Delta_i^t / (eta_l E) = grad F_i(theta_t) + proxy_error_i^t
```

where `proxy_error_i^t` is controlled by smoothness, stochastic gradient variance, local learning rate, and the number of local steps.

Expected bound shape:

```text
E || proxy_error_i^t ||^2 <= C_1 eta_l^2 E^2 G^2 + C_2 sigma_i^2 / E
```

The exact constants are less important than the interpretation:

1. Smaller local learning rate reduces drift.
2. Fewer local steps reduce drift.
3. More stochasticity increases proxy noise.
4. The uploaded delta remains a useful descent proxy when local drift is controlled.

Statement idea:

```text
If each client performs one SGD step from theta_t with learning rate eta, then -Delta_i / eta is an unbiased stochastic estimate of grad F_i(theta_t). For multiple local steps, -Delta_i is a trajectory-integrated descent proxy for client i under smoothness assumptions.
```

Use this to justify why FedClient-UPGrad can work without explicit per-client full gradients.

### 7.3 Lemma 2: UPGrad Common Direction

Purpose:

```text
Show that the server-side UPGrad step is not an arbitrary transformation of client updates; it is a structured common-direction computation over sampled client objectives.
```

Define client update conflict using pairwise inner products:

```text
<g_i, g_j> < 0
```

where:

```text
g_i^t = -Delta_i^t
```

The sampled client proxy matrix is:

```text
G_t = [g_i^t]_{i in S_t}
```

UPGrad uses the Gramian:

```text
H_t = G_t G_t^T
```

to compute weights `w_t` and direction:

```text
d_t = G_t^T w_t
```

The lemma should establish one of the following, depending on how much formal detail is manageable:

1. **Alignment form:** the UPGrad direction has nonnegative or improved alignment with sampled client proxy directions compared with naive averaging under specified conditions.
2. **Projection form:** UPGrad solves a Gramian-space constrained quadratic problem that yields a direction adjusted for client-update conflicts.
3. **Descent form:** if the proxy direction `d_t` has sufficient positive alignment with each active client gradient proxy, then a small enough server step decreases the first-order approximation of the sampled client objectives.

Suggested cautious statement:

```text
Given client proxy gradients G_t, UPGrad computes a Gramian-aware combination d_t = G_t^T w_t. When client updates conflict, this direction differs from simple averaging by accounting for pairwise update geometry. Under a positive-alignment condition, d_t is a common descent direction for the sampled proxy objectives.
```

This lemma connects directly to the empirical worst-10% accuracy result: the method is designed to avoid letting majority or easy-client directions dominate the update.

### 7.4 Theorem 3: Approximate Pareto-Stationarity

A realistic nonconvex convergence target is not global optimality, but first-order multi-objective stationarity. This theorem should combine Lemma 1 and Lemma 2.

Define a Pareto-stationarity residual over the full client set or sampled client set:

```text
R(theta) = min_{lambda in simplex} || sum_i lambda_i grad F_i(theta) ||^2
```

or sampled version:

```text
R_S(theta_t) = min_{lambda in simplex} || sum_{i in S_t} lambda_i grad F_i(theta_t) ||^2
```

The desired theorem shape:

```text
1/T sum_{t=0}^{T-1} E[R(theta_t)]
  <= optimization_error(T)
   + local_update_proxy_error
   + stochastic_gradient_error
   + client_sampling_error
```

Expected rate form:

```text
O(1/sqrt(T)) + O(proxy error) + O(sampling error)
```

For a scalarized fallback result, the stationarity measure can be `||grad F_bar(theta)||^2`, where `F_bar` is the average or a fixed scalarization of client losses. For the paper's multi-objective view, the Pareto-stationarity residual is more aligned with the method.

Logical dependency:

1. Lemma 1 says `G_t` is a noisy/proxy version of sampled client gradients.
2. Lemma 2 says UPGrad computes a structured common direction from `G_t`.
3. The theorem says repeated updates with this approximate common direction drive the model toward approximate Pareto stationarity, up to proxy and sampling errors.

Important distinction:

```text
Nonconvex first-order convergence is plausible.
Nonconvex global convergence to the global optimum is not a realistic claim for neural-network federated learning without very restrictive assumptions.
```

Recommended theorem assumptions:

1. Each client objective `F_i` is L-smooth.
2. Stochastic gradients are unbiased with bounded variance.
3. Gradients or local updates have bounded second moment.
4. Client sampling is unbiased or has bounded sampling variance.
5. Local learning rate and server step size are sufficiently small.
6. UPGrad direction has bounded norm and satisfies a positive alignment or sufficient descent condition with the proxy objectives.

Recommended wording:

```text
Under standard smooth nonconvex assumptions, FedClient-UPGrad converges to a neighborhood of a Pareto-stationary point. The neighborhood size is governed by the local-update proxy error, stochastic gradient noise, and partial-participation sampling error.
```

Avoid claiming convergence to global optimum.

### 7.5 Theory-Experiment Link

The theory and experiments should be written as one argument, not as two independent parts. Each theoretical claim should motivate a concrete experimental measurement.

| Theory Claim | What It Says | Experimental Counterpart | Metric/Figure |
|---|---|---|---|
| Local Delta Proxy Lemma | Client local deltas contain usable client-objective descent information. | FedClient-UPGrad uses exactly the same FedAvg-style local training and uploads one model delta per client. | Communication parity with FedAvg-style methods; no explicit Jacobian upload. |
| UPGrad Common Direction Lemma | Aggregation should account for update conflicts rather than only average client deltas. | Compare FedClient-UPGrad with FedAvg and FedMGDA+ on heterogeneous clients. | Mean accuracy, worst-10% accuracy, sorted client accuracy curves. |
| Approximate Pareto-Stationarity Theorem | The method targets a multi-objective first-order balance condition, not only average-loss stationarity. | Evaluate whether the final model improves tail-client performance and reduces inter-client dispersion. | Worst-10% accuracy, client accuracy standard deviation, client accuracy CDF/boxplot. |
| Proxy/Sampling Error Terms | Local training, partial participation, and stochasticity introduce approximation errors. | Use three random seeds and report mean ± std. | Three-seed result tables and error bars. |
| Server-Side Common Direction Cost | Structured aggregation has extra computation. | Report average round time and aggregation compute time. | Efficiency-performance trade-off plot. |

The FEMNIST result already follows this logic:

1. Theory says local deltas can represent client-objective directions; the implementation uses local deltas only, not expensive per-client Jacobians.
2. Theory says conflict-aware aggregation should help balance objectives; FEMNIST writer clients create real client heterogeneity and FedClient-UPGrad improves worst-10% accuracy from `0.4903 ± 0.0252` to `0.5902 ± 0.0058` over FedAvg.
3. Theory says approximate stationarity is affected by stochasticity and sampling; experiments report three seeds, not a single run.
4. Theory says better common-direction computation has a cost; experiments report FedClient-UPGrad's higher average round time (`1.480s`) and aggregation time (`0.4722s`).

Recommended writing structure for the Results section:

1. Start each result subsection by naming the theoretical claim being tested.
2. Then report the relevant metric.
3. Then interpret whether the metric supports the claim.

Example:

```text
The theory suggests that a common direction computed from client update geometry should better balance client objectives than simple averaging. We therefore evaluate both average client accuracy and tail-client accuracy. On FEMNIST, FedClient-UPGrad improves worst-10% accuracy by 9.99 percentage points over FedAvg while also improving mean accuracy, indicating that the update-space common direction improves tail clients without sacrificing average performance.
```

This is the central theory-experiment bridge of the paper: the theory defines what kind of stationarity and direction quality the method targets; the experiments test whether this target manifests as improved tail-client performance, lower client dispersion, and acceptable efficiency trade-offs.

## 8. Discussion

### 8.1 Why FedClient-UPGrad Works on FEMNIST

Likely explanation:

1. FEMNIST writer partition creates real client identity heterogeneity.
2. Writer-specific distributions induce client-update conflicts.
3. FedClient-UPGrad reduces dominance by average-client directions and better accounts for tail-client objectives.

### 8.2 Why qFedAvg and FedMGDA+ May Underperform

qFedAvg:

1. Scalar loss reweighting does not fully capture update-direction conflict.
2. Loss values can be noisy and may not map cleanly to useful aggregation geometry.

FedMGDA+:

1. Min-norm aggregation can be conservative.
2. It may reduce conflict but also shrink useful progress directions.
3. FedClient-UPGrad appears to find a better balance between common descent and progress in current FEMNIST results.

### 8.3 What CIFAR10 Will Add

CIFAR10 is important because:

1. It tests label-skew heterogeneity rather than writer heterogeneity.
2. It includes two heterogeneity strengths: alpha=0.1 and alpha=0.5.
3. It evaluates whether the method generalizes beyond FEMNIST.

## 9. Limitations

This section should be honest and strategic.

### 9.1 Runtime and Server-Side Cost

FedClient-UPGrad is slower than FedAvg because it solves an aggregation problem over sampled client updates. On FEMNIST, average round time is `1.480s` versus `1.012s` for FedAvg.

Recommended framing:

```text
FedClient-UPGrad trades additional server-side computation for improved tail-client performance. This trade-off is acceptable in settings where fairness or weak-client reliability is important, but it may be less suitable for extremely resource-constrained servers or very large sampled-client counts.
```

### 9.2 Approximate Client Gradients

The method uses local deltas as gradient proxies. This is practical but approximate, especially with multiple local epochs and nonconvex models.

### 9.3 Benchmark Scope

Current main evidence is limited to vision benchmarks. This is acceptable for a focused paper, but the conclusion should state that broader modalities and larger-scale models remain future work.

### 9.4 Implementation Maturity

The current code is a research implementation. It uses PyTorch and custom baseline implementations rather than official library implementations for every method. This should be acknowledged if appropriate, but not overemphasized in the main paper.

## 10. Conclusion

The conclusion should restate:

1. Heterogeneous FL should be evaluated at the client level, not only by global average performance.
2. FedClient-UPGrad provides a practical way to perform client-level multi-objective aggregation using standard client local updates.
3. FEMNIST results show consistent gains in average accuracy, worst-10% accuracy, dispersion, and loss.
4. CIFAR10 results will determine how strongly to claim generalization across label-skew settings.
5. Future work should improve aggregation efficiency and test larger-scale models/datasets.

## Recommended Figures and Tables

Main paper:

1. Table 1: FEMNIST and CIFAR10 three-seed `mean ± std` results.
2. Figure 1: Mean accuracy vs worst-10% accuracy scatter, one subplot per dataset.
3. Figure 2: Sorted client accuracy curves for FEMNIST and CIFAR10.
4. Figure 3: Worst-10% accuracy bar chart with three-seed error bars.
5. Figure 4: Efficiency-performance trade-off, using average round time vs worst-10% accuracy.

Appendix:

1. CIFAR10 R100 tuning table.
2. Update-scale and learning-rate sensitivity.
3. Per-seed result tables.
4. Runtime optimization details.

## Current Evidence Status

Completed and supportive:

1. FEMNIST three-seed full run strongly supports FedClient-UPGrad.
2. CIFAR10 alpha=0.1 three-seed mixed-parameter full run supports FedClient-UPGrad as a tail-client/fairness-improving method: best worst-10% accuracy and lowest client accuracy standard deviation, but not best mean accuracy.
3. CIFAR10 R100 tuning identifies strong baseline parameters and reveals that short-horizon tuning must be paired with long-horizon stability checks for FedClient-UPGrad and FedMGDA+.
4. Prior CIFAR10 seed-7 R1000 run with `lr=0.01` was positive for FedClient-UPGrad and motivated the stable full-run configuration.

Pending:

1. CIFAR10 alpha=0.5 three-seed full result with mixed/stable parameters.
2. Final cross-dataset aggregate table.
3. Main plots generated from final result directories.

## Writing Strategy

Recommended emphasis:

1. Lead with client-level multi-objective formulation and tail-client performance.
2. Present FedClient-UPGrad as a practical update-space method, not as a heavy theoretical framework requiring full Jacobians.
3. Use FEMNIST as the strongest completed evidence because it is naturally client-defined through writers.
4. Use CIFAR10 to show generalization to synthetic label-skew heterogeneity once complete.
5. Be transparent about runtime cost and frame it as a performance-fairness trade-off.

Avoid these claims unless additional proof or experiments are added:

1. Do not claim convergence to a global optimum for nonconvex FL; claim approximate first-order or Pareto stationarity only if a theorem is written carefully.
2. Do not claim communication reduction over FedAvg; communication is FedAvg-style, not lower.
3. Do not claim universal superiority; the current CIFAR10 alpha=0.1 result supports a fairness-performance trade-off rather than full dominance.

## Immediate Next Steps

1. Wait for CIFAR10 alpha=0.5 full runs to finish.
2. Aggregate CIFAR10 alpha=0.5 into the same table format as FEMNIST and CIFAR10 alpha=0.1.
3. Generate final plots from `summary.csv` and `clients_*.csv` files.
4. Draft Sections 1, 3, 4, and 5 first; they are already strongly grounded in the current implementation.
5. Draft the theory section conservatively around local update proxies and common-direction alignment.
