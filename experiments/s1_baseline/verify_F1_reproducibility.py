from __future__ import annotations

import random

import torch

from fedjd import ExperimentConfig, FedJDClient, FedJDServer, FedJDTrainer, MinNormAggregator
from fedjd.data import make_synthetic_federated_regression
from fedjd.models import SmallRegressor
from fedjd.problems import two_objective_regression


def run_once(seed: int) -> list[float]:
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")
    data = make_synthetic_federated_regression(
        num_clients=8, samples_per_client=64, input_dim=8, noise_std=0.1, seed=seed
    )
    clients = [
        FedJDClient(client_id=i, dataset=ds, batch_size=32, device=device)
        for i, ds in enumerate(data.client_datasets)
    ]
    model = SmallRegressor(input_dim=8, hidden_dim=16, output_dim=2)
    server = FedJDServer(
        model=model, clients=clients, aggregator=MinNormAggregator(),
        objective_fn=two_objective_regression,
        participation_rate=0.5, learning_rate=0.05, device=device,
    )

    objectives = []
    for round_idx in range(5):
        server.run_round(round_idx)
        obj = server.evaluate_global_objectives()
        objectives.extend(obj)

    return objectives


def check_F1_reproducibility():
    print("=" * 60)
    print("F1. 同种子可复现性验证")
    print("=" * 60)

    seed = 42
    results_run1 = run_once(seed)
    results_run2 = run_once(seed)

    all_match = True
    for i, (v1, v2) in enumerate(zip(results_run1, results_run2)):
        if abs(v1 - v2) > 1e-10:
            all_match = False
            print(f"  Mismatch at index {i}: run1={v1:.10f}, run2={v2:.10f}, diff={abs(v1-v2):.2e}")

    f1_pass = all_match
    print(f"  F1 Same seed produces identical results: {'PASS' if f1_pass else 'FAIL'}")
    print(f"     Run 1 final objectives: [{results_run1[-2]:.6f}, {results_run1[-1]:.6f}]")
    print(f"     Run 2 final objectives: [{results_run2[-2]:.6f}, {results_run2[-1]:.6f}]")

    return f1_pass


if __name__ == "__main__":
    check_F1_reproducibility()
