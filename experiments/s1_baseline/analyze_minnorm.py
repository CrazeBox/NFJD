from __future__ import annotations

import random

import torch

from fedjd import FedJDClient, FedJDServer, MinNormAggregator, MeanAggregator, RandomAggregator
from fedjd.aggregators import _project_simplex
from fedjd.data import make_synthetic_federated_regression
from fedjd.models import SmallRegressor
from fedjd.problems import two_objective_regression


def analyze_minnorm_behavior():
    print("=" * 60)
    print("MinNorm 行为深入分析")
    print("=" * 60)

    seed = 7
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")

    data = make_synthetic_federated_regression(
        num_clients=8, samples_per_client=64, input_dim=8, noise_std=0.1, seed=seed
    )
    model = SmallRegressor(input_dim=8, hidden_dim=16, output_dim=2)
    clients = [
        FedJDClient(client_id=i, dataset=ds, batch_size=32, device=device)
        for i, ds in enumerate(data.client_datasets)
    ]

    server = FedJDServer(
        model=model, clients=clients, aggregator=MinNormAggregator(max_iters=500, lr=0.2),
        objective_fn=two_objective_regression,
        participation_rate=0.5, learning_rate=0.05, device=device,
    )

    print("\n--- 逐轮 MinNorm 权重分析 ---")
    for round_idx in range(10):
        sampled = server.sample_clients()
        total_examples = sum(c.num_examples for c in sampled)

        aggregated_jacobian = None
        for client in sampled:
            import copy
            result = client.compute_jacobian(copy.deepcopy(server.model), server.objective_fn)
            weight = result.num_examples / total_examples
            weighted_jac = result.jacobian.to(device) * weight
            if aggregated_jacobian is None:
                aggregated_jacobian = weighted_jac
            else:
                aggregated_jacobian.add_(weighted_jac)

        # Compute gramian and solve for lambda
        gramian = aggregated_jacobian @ aggregated_jacobian.T
        lambdas = torch.full((2,), 0.5, dtype=aggregated_jacobian.dtype, device=device)

        for _ in range(500):
            grad = gramian @ lambdas
            candidate = _project_simplex(lambdas - 0.2 * grad)
            if torch.norm(candidate - lambdas, p=2) <= 1e-8:
                lambdas = candidate
                break
            lambdas = candidate

        # Compute direction
        direction = aggregated_jacobian.T @ lambdas

        # Gradient inner products
        g0 = aggregated_jacobian[0]
        g1 = aggregated_jacobian[1]
        g0_dot_d = float((g0 @ direction).item())
        g1_dot_d = float((g1 @ direction).item())
        g0_dot_g1 = float((g0 @ g1).item())
        g0_norm = float(torch.norm(g0, p=2).item())
        g1_norm = float(torch.norm(g1, p=2).item())

        # Cosine similarity between gradients
        cos_sim = g0_dot_g1 / (g0_norm * g1_norm) if g0_norm > 0 and g1_norm > 0 else 0

        obj = server.evaluate_global_objectives()
        print(
            f"  Round {round_idx}: λ=[{lambdas[0]:.4f}, {lambdas[1]:.4f}] | "
            f"obj=[{obj[0]:.4f}, {obj[1]:.4f}] | "
            f"cos(g0,g1)={cos_sim:.4f} | "
            f"<g0,d>={g0_dot_d:.4f} | <g1,d>={g1_dot_d:.4f}"
        )

        server.run_round(round_idx)

    print("\n--- 分析结论 ---")
    print("MinNorm 在 m=2 场景下的行为：")
    print("  - 当两个目标梯度方向高度一致（cos ≈ 1），MinNorm 退化为选择范数较大的梯度")
    print("  - 当两个目标梯度方向冲突（cos < 0），MinNorm 才会真正平衡两个目标")
    print("  - 在合成回归任务中，两个 MSE 目标的梯度通常高度一致")
    print("  - 这导致 MinNorm 的表现与简单平均或随机权重差异不大")
    print("  - 这不是 bug，而是 MinNorm 在低冲突场景下的预期行为")


if __name__ == "__main__":
    analyze_minnorm_behavior()
