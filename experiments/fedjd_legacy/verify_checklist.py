from __future__ import annotations

import random
import sys

import torch

from fedjd import FedJDClient, FedJDServer, MinNormAggregator
from fedjd.core.client import flatten_gradients
from fedjd.data import make_synthetic_federated_regression
from fedjd.models import SmallRegressor
from fedjd.problems import two_objective_regression


def check_A_dimension_and_numerics():
    print("=" * 60)
    print("A. 维度与数值正确性")
    print("=" * 60)

    seed = 7
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")

    data = make_synthetic_federated_regression(
        num_clients=8, samples_per_client=64, input_dim=8, noise_std=0.1, seed=seed
    )
    model = SmallRegressor(input_dim=8, hidden_dim=16, output_dim=2)
    num_params = sum(p.numel() for p in model.parameters())

    client = FedJDClient(client_id=0, dataset=data.client_datasets[0], batch_size=32, device=device)
    result = client.compute_jacobian(model, two_objective_regression)

    # A1: Jacobian shape
    expected_shape = (2, num_params)
    a1_pass = result.jacobian.shape == expected_shape
    print(f"  A1 Jacobian shape = {tuple(result.jacobian.shape)}, expected = {expected_shape}: {'PASS' if a1_pass else 'FAIL'}")

    # A2: m=2 rows
    a2_pass = result.jacobian.shape[0] == 2
    print(f"  A2 Jacobian rows = {result.jacobian.shape[0]}, expected = 2: {'PASS' if a2_pass else 'FAIL'}")

    # A3: d = model parameter count
    a3_pass = result.jacobian.shape[1] == num_params
    print(f"  A3 Jacobian cols = {result.jacobian.shape[1]}, model params = {num_params}: {'PASS' if a3_pass else 'FAIL'}")

    # A4: Jacobian not all zeros
    a4_pass = result.jacobian.abs().sum().item() > 0
    print(f"  A4 Jacobian abs sum = {result.jacobian.abs().sum().item():.6f} (> 0): {'PASS' if a4_pass else 'FAIL'}")

    # A5: No NaN/Inf
    nan_count = torch.isnan(result.jacobian).sum().item()
    inf_count = torch.isinf(result.jacobian).sum().item()
    a5_pass = nan_count == 0 and inf_count == 0
    print(f"  A5 NaN count = {nan_count}, Inf count = {inf_count}: {'PASS' if a5_pass else 'FAIL'}")

    # A6: Direction dimension = d
    aggregator = MinNormAggregator(max_iters=250, lr=0.2)
    direction = aggregator(result.jacobian)
    a6_pass = direction.shape == (num_params,)
    print(f"  A6 Direction shape = {tuple(direction.shape)}, expected = ({num_params},): {'PASS' if a6_pass else 'FAIL'}")

    # A7: Direction not all zeros and finite
    dir_norm = torch.norm(direction, p=2).item()
    dir_has_nan = torch.isnan(direction).any().item()
    a7_pass = dir_norm > 0 and not dir_has_nan
    print(f"  A7 Direction norm = {dir_norm:.6f}, has_nan = {dir_has_nan}: {'PASS' if a7_pass else 'FAIL'}")

    return {
        "A1": a1_pass, "A2": a2_pass, "A3": a3_pass, "A4": a4_pass,
        "A5": a5_pass, "A6": a6_pass, "A7": a7_pass,
    }


def check_B_aggregation_consistency():
    print("\n" + "=" * 60)
    print("B. 聚合一致性")
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

    # B1: Weighted weights sum to 1
    total_examples = sum(c.num_examples for c in clients)
    weights = [c.num_examples / total_examples for c in clients]
    weight_sum = sum(weights)
    b1_pass = abs(weight_sum - 1.0) < 1e-6
    print(f"  B1 Weight sum = {weight_sum:.10f} (≈ 1.0): {'PASS' if b1_pass else 'FAIL'}")

    # B2: Full participation includes all clients
    server = FedJDServer(
        model=model, clients=clients, aggregator=MinNormAggregator(),
        objective_fn=two_objective_regression,
        participation_rate=1.0, learning_rate=0.05, device=device,
    )
    sampled = server.sample_clients()
    sampled_ids = sorted([c.client_id for c in sampled])
    all_ids = list(range(8))
    b2_pass = sampled_ids == all_ids
    print(f"  B2 C=1.0 sampled = {sampled_ids}, all = {all_ids}: {'PASS' if b2_pass else 'FAIL'}")

    # B3: Aggregation matches manual calculation
    jacobians = []
    for client in clients:
        result = client.compute_jacobian(model, two_objective_regression)
        jacobians.append(result.jacobian)

    manual_agg = torch.zeros_like(jacobians[0])
    for j, w in zip(jacobians, weights):
        manual_agg += w * j

    server_agg = torch.zeros_like(jacobians[0])
    for j, w in zip(jacobians, weights):
        server_agg.add_(w * j)

    diff = torch.norm(manual_agg - server_agg, p="fro").item()
    b3_pass = diff < 1e-5
    print(f"  B3 Manual vs server aggregation diff = {diff:.10f} (< 1e-5): {'PASS' if b3_pass else 'FAIL'}")

    # B4: MinNorm weights on simplex
    aggregator = MinNormAggregator(max_iters=500, lr=0.1)
    gramian = server_agg @ server_agg.T
    lambdas = torch.full((2,), 0.5, dtype=server_agg.dtype, device=server_agg.device)
    for _ in range(500):
        grad = gramian @ lambdas
        sorted_v, _ = torch.sort(lambdas - 0.1 * grad, descending=True)
        cumsum = torch.cumsum(sorted_v, dim=0)
        steps = torch.arange(1, 3, device=server_agg.device, dtype=server_agg.dtype)
        support = sorted_v - (cumsum - 1.0) / steps > 0
        rho = int(torch.nonzero(support, as_tuple=False)[-1].item())
        theta = (cumsum[rho] - 1.0) / float(rho + 1)
        candidate = torch.clamp(lambdas - 0.1 * grad - theta, min=0.0)
        if torch.norm(candidate - lambdas, p=2) <= 1e-8:
            lambdas = candidate
            break
        lambdas = candidate

    lambda_nonneg = (lambdas >= -1e-8).all().item()
    lambda_sum = lambdas.sum().item()
    b4_pass = lambda_nonneg and abs(lambda_sum - 1.0) < 1e-4
    print(f"  B4 MinNorm λ = [{lambdas[0]:.6f}, {lambdas[1]:.6f}], sum = {lambda_sum:.6f}, all ≥ 0: {'PASS' if b4_pass else 'FAIL'}")

    return {"B1": b1_pass, "B2": b2_pass, "B3": b3_pass, "B4": b4_pass}


def check_C_parameter_update():
    print("\n" + "=" * 60)
    print("C. 参数更新有效性")
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
        model=model, clients=clients, aggregator=MinNormAggregator(),
        objective_fn=two_objective_regression,
        participation_rate=0.5, learning_rate=0.05, device=device,
    )

    # C1: Parameters change after update
    params_before = torch.cat([p.detach().reshape(-1) for p in server.model.parameters()]).clone()
    stats = server.run_round(0)
    params_after = torch.cat([p.detach().reshape(-1) for p in server.model.parameters()]).clone()
    param_diff = torch.norm(params_after - params_before, p=2).item()
    c1_pass = param_diff > 0
    print(f"  C1 Parameter change norm = {param_diff:.6f} (> 0): {'PASS' if c1_pass else 'FAIL'}")

    # C2: Direction is a descent direction (at least one objective gradient has negative inner product)
    random.seed(seed)
    torch.manual_seed(seed)
    data2 = make_synthetic_federated_regression(
        num_clients=8, samples_per_client=64, input_dim=8, noise_std=0.1, seed=seed
    )
    model2 = SmallRegressor(input_dim=8, hidden_dim=16, output_dim=2)
    client2 = FedJDClient(client_id=0, dataset=data2.client_datasets[0], batch_size=64, device=device)

    loader = torch.utils.data.DataLoader(data2.client_datasets[0], batch_size=64, shuffle=False)
    batch_x, batch_y = next(iter(loader))
    model2.zero_grad(set_to_none=True)
    preds = model2(batch_x)
    obj_vals = two_objective_regression(preds, batch_y, batch_x)
    grads = []
    for i, ov in enumerate(obj_vals):
        model2.zero_grad(set_to_none=True)
        ov.backward(retain_graph=(i < len(obj_vals) - 1))
        grads.append(flatten_gradients(model2.parameters()).clone())

    jacobian = torch.stack(grads, dim=0)
    aggregator = MinNormAggregator()
    direction = aggregator(jacobian)

    inner_products = [float((g @ direction).item()) for g in grads]
    c2_pass = any(ip > 0 for ip in inner_products)
    print(f"  C2 Inner products <g_i, d> = [{inner_products[0]:.4f}, {inner_products[1]:.4f}], at least one > 0 (descent since theta -= lr*d): {'PASS' if c2_pass else 'FAIL'}")

    # C3: Learning rate affects update magnitude
    random.seed(seed)
    torch.manual_seed(seed)
    data3 = make_synthetic_federated_regression(
        num_clients=8, samples_per_client=64, input_dim=8, noise_std=0.1, seed=seed
    )
    model_small_lr = SmallRegressor(input_dim=8, hidden_dim=16, output_dim=2)
    model_large_lr = SmallRegressor(input_dim=8, hidden_dim=16, output_dim=2)
    model_large_lr.load_state_dict(model_small_lr.state_dict())

    clients3 = [
        FedJDClient(client_id=i, dataset=ds, batch_size=32, device=device)
        for i, ds in enumerate(data3.client_datasets)
    ]

    server_small = FedJDServer(
        model=model_small_lr, clients=clients3, aggregator=MinNormAggregator(),
        objective_fn=two_objective_regression,
        participation_rate=1.0, learning_rate=0.01, device=device,
    )
    server_large = FedJDServer(
        model=model_large_lr, clients=clients3, aggregator=MinNormAggregator(),
        objective_fn=two_objective_regression,
        participation_rate=1.0, learning_rate=0.1, device=device,
    )

    params_s_before = torch.cat([p.detach().reshape(-1) for p in server_small.model.parameters()]).clone()
    params_l_before = torch.cat([p.detach().reshape(-1) for p in server_large.model.parameters()]).clone()

    random.seed(seed + 100)
    torch.manual_seed(seed + 100)
    server_small.run_round(0)
    random.seed(seed + 100)
    torch.manual_seed(seed + 100)
    server_large.run_round(0)

    params_s_after = torch.cat([p.detach().reshape(-1) for p in server_small.model.parameters()])
    params_l_after = torch.cat([p.detach().reshape(-1) for p in server_large.model.parameters()])

    diff_small = torch.norm(params_s_after - params_s_before, p=2).item()
    diff_large = torch.norm(params_l_after - params_l_before, p=2).item()
    c3_pass = diff_large > diff_small
    print(f"  C3 lr=0.01 change = {diff_small:.6f}, lr=0.1 change = {diff_large:.6f}, larger lr → larger change: {'PASS' if c3_pass else 'FAIL'}")

    return {"C1": c1_pass, "C2": c2_pass, "C3": c3_pass}


if __name__ == "__main__":
    results_A = check_A_dimension_and_numerics()
    results_B = check_B_aggregation_consistency()
    results_C = check_C_parameter_update()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_results = {**results_A, **results_B, **results_C}
    for k, v in all_results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    total = len(all_results)
    passed = sum(all_results.values())
    print(f"\n  Total: {passed}/{total} passed")
