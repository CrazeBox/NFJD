from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from fedjd.aggregators import MinNormAggregator
from fedjd.core import (
    DirectionAvgServer, FedJDClient, FedJDServer, FedJDTrainer,
    FMGDAServer, NFJDClient, NFJDServer, NFJDTrainer, WeightedSumServer,
)
from fedjd.data import make_synthetic_federated_regression
from fedjd.metrics import extract_pareto_front, hypervolume
from fedjd.models import SmallRegressor
from fedjd.problems import multi_objective_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/nfjd_phase1")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_FIELDNAMES = [
    "exp_id", "method", "dataset", "m", "seed", "num_rounds", "num_clients",
    "participation_rate", "learning_rate", "conflict_strength",
    "model_size", "local_epochs", "use_adaptive_rescaling",
    "use_stochastic_gramian", "client_compute_mode", "recompute_interval",
    "elapsed_time", "all_decreased",
    "hypervolume", "pareto_gap", "avg_relative_improvement",
    "avg_upload_bytes", "avg_round_time", "upload_per_client", "avg_rescale_factor",
]
MAX_M = 10
for i in range(MAX_M):
    ALL_FIELDNAMES.extend([f"init_obj_{i}", f"final_obj_{i}", f"delta_obj_{i}"])


def _run_single(method, dataset, m, seed, num_rounds=50,
                num_clients=10, participation_rate=0.5, learning_rate=0.01,
                model_size="small", local_epochs=3, use_adaptive_rescaling=True,
                use_stochastic_gramian=True, conflict_strength=0.0,
                client_compute_mode="full_loader", recompute_interval=1):
    exp_id = f"P1-{method}-synth-m{m}-seed{seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    fed_data = make_synthetic_federated_regression(
        num_clients=num_clients, samples_per_client=100, input_dim=8,
        num_objectives=m, seed=seed)

    model_cls = SmallRegressor
    model = model_cls(input_dim=fed_data.input_dim, output_dim=m)
    objective_fn = multi_objective_regression
    baseline_client_kwargs = dict(
        batch_size=32,
        device=device,
        use_full_loader=(client_compute_mode == "full_loader"),
        local_epochs=local_epochs if client_compute_mode == "full_loader" else 1,
    )

    if method == "nfjd":
        clients = [NFJDClient(client_id=i, dataset=fed_data.client_datasets[i], batch_size=32,
                              device=device, local_epochs=local_epochs, learning_rate=learning_rate,
                              local_momentum_beta=0.9, use_adaptive_rescaling=use_adaptive_rescaling,
                              use_stochastic_gramian=use_stochastic_gramian, stochastic_subset_size=4,
                              stochastic_seed=seed, recompute_interval=recompute_interval) for i in range(num_clients)]
        server = NFJDServer(model=model, clients=clients, objective_fn=objective_fn,
                            participation_rate=participation_rate, learning_rate=learning_rate,
                            device=device, global_momentum_beta=0.9)
        trainer = NFJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "fedjd":
        clients = [FedJDClient(client_id=i, dataset=fed_data.client_datasets[i], **baseline_client_kwargs) for i in range(num_clients)]
        aggregator = MinNormAggregator(max_iters=250, lr=0.1, max_direction_norm=0.0)
        server = FedJDServer(model=model, clients=clients, aggregator=aggregator, objective_fn=objective_fn, participation_rate=participation_rate, learning_rate=learning_rate, device=device)
        trainer = FedJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "fmgda":
        clients = [FedJDClient(client_id=i, dataset=fed_data.client_datasets[i], **baseline_client_kwargs) for i in range(num_clients)]
        server = FMGDAServer(model=model, clients=clients, objective_fn=objective_fn, participation_rate=participation_rate, learning_rate=learning_rate, device=device)
        trainer = FedJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "weighted_sum":
        clients = [FedJDClient(client_id=i, dataset=fed_data.client_datasets[i], **baseline_client_kwargs) for i in range(num_clients)]
        server = WeightedSumServer(model=model, clients=clients, objective_fn=objective_fn, participation_rate=participation_rate, learning_rate=learning_rate, device=device)
        trainer = FedJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "direction_avg":
        clients = [FedJDClient(client_id=i, dataset=fed_data.client_datasets[i], **baseline_client_kwargs) for i in range(num_clients)]
        server = DirectionAvgServer(model=model, clients=clients, objective_fn=objective_fn, participation_rate=participation_rate, learning_rate=learning_rate, device=device)
        trainer = FedJDTrainer(server=server, num_rounds=num_rounds)
    else:
        raise ValueError(f"Unknown method: {method}")

    start = time.time()
    history = trainer.fit()
    elapsed = time.time() - start

    initial_obj = history[0].objective_values
    final_obj = history[-1].objective_values
    obj_history = [s.objective_values for s in history]
    all_decreased = all(final_obj[j] <= initial_obj[j] for j in range(m))

    min_vals = [min(h[j] for h in obj_history) for j in range(m)]
    max_vals = [max(h[j] for h in obj_history) for j in range(m)]
    ranges = [max_vals[j] - min_vals[j] for j in range(m)]
    for j in range(m):
        if ranges[j] < 1e-10:
            ranges[j] = 1.0

    normalized_history = [[(h[j] - min_vals[j]) / ranges[j] for j in range(m)] for h in obj_history]
    ref_point = [1.1] * m
    pareto_front = extract_pareto_front(normalized_history)
    raw_hv = hypervolume(pareto_front, ref_point)
    max_possible_hv = 1.1 ** m
    hv = raw_hv / max_possible_hv if max_possible_hv > 0 else 0.0

    normalized_final = [(final_obj[j] - min_vals[j]) / ranges[j] for j in range(m)]
    pg = sum(normalized_final) / m

    ri_sum = 0.0
    for j in range(m):
        if abs(initial_obj[j]) > 1e-10:
            ri_sum += (initial_obj[j] - final_obj[j]) / abs(initial_obj[j])
        else:
            ri_sum += 1.0 if final_obj[j] < abs(initial_obj[j]) else 0.0
    avg_ri = ri_sum / m

    avg_upload = sum(s.upload_bytes for s in history) / max(len(history), 1)
    avg_round_time = sum(s.round_time for s in history) / max(len(history), 1)
    upload_per_client = avg_upload / max(participation_rate * num_clients, 1)

    avg_rescale = 1.0
    if method == "nfjd":
        rescale_vals = [s.avg_rescale_factor for s in history]
        avg_rescale = sum(rescale_vals) / len(rescale_vals) if rescale_vals else 1.0

    effective_local_epochs = local_epochs if (method == "nfjd" or client_compute_mode == "full_loader") else 1
    row = {
        "exp_id": exp_id, "method": method, "dataset": dataset, "m": m, "seed": seed,
        "num_rounds": num_rounds, "num_clients": num_clients,
        "participation_rate": participation_rate, "learning_rate": learning_rate,
        "conflict_strength": conflict_strength,
        "model_size": model_size, "local_epochs": effective_local_epochs,
        "use_adaptive_rescaling": use_adaptive_rescaling if method == "nfjd" else False,
        "use_stochastic_gramian": use_stochastic_gramian if method == "nfjd" else False,
        "client_compute_mode": client_compute_mode,
        "recompute_interval": recompute_interval,
        "elapsed_time": round(elapsed, 2), "all_decreased": all_decreased,
        "hypervolume": round(hv, 6), "pareto_gap": round(pg, 6),
        "avg_relative_improvement": round(avg_ri, 6),
        "avg_upload_bytes": round(avg_upload, 0),
        "avg_round_time": round(avg_round_time, 4),
        "upload_per_client": round(upload_per_client, 0),
        "avg_rescale_factor": round(avg_rescale, 4),
    }
    for i in range(MAX_M):
        if i < m:
            row[f"init_obj_{i}"] = round(initial_obj[i], 6)
            row[f"final_obj_{i}"] = round(final_obj[i], 6)
            row[f"delta_obj_{i}"] = round(final_obj[i] - initial_obj[i], 6)
        else:
            row[f"init_obj_{i}"] = ""
            row[f"final_obj_{i}"] = ""
            row[f"delta_obj_{i}"] = ""

    logger.info("[%s] %s: NHV=%.4f RI=%.4f upload/client=%d rescale=%.2f mode=%s",
                exp_id, method, hv, avg_ri, upload_per_client, avg_rescale, client_compute_mode)
    return row


def _write_csv(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Run NFJD Phase 1 baseline verification.")
    parser.add_argument(
        "--client-compute-mode",
        choices=("full_loader", "single_batch"),
        default="full_loader",
        help="How non-NFJD baselines compute local Jacobians. full_loader matches NFJD's local workload more fairly.",
    )
    args = parser.parse_args()

    seeds = [7, 42, 123]
    methods = ["nfjd", "fedjd", "fmgda", "weighted_sum", "direction_avg"]
    m_values = [2, 3, 5]
    all_rows = []
    experiments = []

    for m in m_values:
        for method in methods:
            for seed in seeds:
                experiments.append(dict(
                    method=method,
                    dataset="synthetic_regression",
                    m=m,
                    seed=seed,
                    num_rounds=50,
                    model_size="small",
                    local_epochs=3,
                    client_compute_mode=args.client_compute_mode,
                ))

    total = len(experiments)
    logger.info("Starting NFJD Phase 1 Baseline Verification: %d experiments, client_compute_mode=%s", total, args.client_compute_mode)

    for idx, exp in enumerate(experiments):
        logger.info("[%d/%d] Running %s...", idx + 1, total, exp)
        try:
            row = _run_single(**exp)
            all_rows.append(row)
        except Exception as exc:
            logger.error("Experiment failed: %s", exc)
            import traceback
            traceback.print_exc()

    csv_path = RESULTS_DIR / "phase1_results.csv"
    _write_csv(csv_path, all_rows)
    logger.info("Phase 1 complete! %d/%d experiments, saved to %s", len(all_rows), total, csv_path)


if __name__ == "__main__":
    main()
