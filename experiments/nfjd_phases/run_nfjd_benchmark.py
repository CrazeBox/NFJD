from __future__ import annotations

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
    DirectionAvgServer, FMGDAClient, FedJDClient, FedJDServer, FedJDTrainer,
    FMGDAServer, NFJDClient, NFJDServer, NFJDTrainer, WeightedSumServer,
)
from fedjd.experiments.nfjd_phases.metric_utils import summarize_objective_history, summarize_round_history
from fedjd.data import make_high_conflict_federated_regression, make_synthetic_federated_regression
from fedjd.models import MediumRegressor, SmallRegressor
from fedjd.problems import multi_objective_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/nfjd_benchmark")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_FIELDNAMES = [
    "exp_id", "method", "dataset", "m", "seed", "num_rounds", "num_clients",
    "participation_rate", "learning_rate", "conflict_strength",
    "model_size", "local_epochs", "use_adaptive_rescaling",
    "use_stochastic_gramian", "elapsed_time", "all_decreased",
    "hypervolume", "pareto_gap", "task_jfi", "task_mmag", "avg_relative_improvement", "avg_ri",
    "avg_upload_bytes", "avg_round_time", "upload_per_client", "avg_rescale_factor",
]
MAX_M = 10
for i in range(MAX_M):
    ALL_FIELDNAMES.extend([f"init_obj_{i}", f"final_obj_{i}", f"delta_obj_{i}"])


def _run_single(method, dataset, m, seed, conflict_strength=0.0, num_rounds=50,
                num_clients=10, participation_rate=0.5, learning_rate=0.01,
                model_size="small", local_epochs=3, use_adaptive_rescaling=True,
                use_stochastic_gramian=True):
    exp_id = f"NFJD-{method}-{dataset}-m{m}-seed{seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    if conflict_strength > 0:
        fed_data = make_high_conflict_federated_regression(
            num_clients=num_clients, samples_per_client=100, input_dim=8,
            num_objectives=m, conflict_strength=conflict_strength, seed=seed)
    else:
        fed_data = make_synthetic_federated_regression(
            num_clients=num_clients, samples_per_client=100, input_dim=8,
            num_objectives=m, seed=seed)

    model_cls = MediumRegressor if model_size == "medium" else SmallRegressor
    model = model_cls(input_dim=fed_data.input_dim, output_dim=m)
    objective_fn = multi_objective_regression

    if method == "nfjd":
        clients = [NFJDClient(client_id=i, dataset=fed_data.client_datasets[i], batch_size=32,
                              device=device, local_epochs=local_epochs, learning_rate=learning_rate,
                              local_momentum_beta=0.0, use_adaptive_rescaling=False,
                              use_stochastic_gramian=False, stochastic_subset_size=4,
                              stochastic_seed=seed + i, recompute_interval=1,
                              exact_upgrad=True, use_objective_normalization=True,
                              upload_align_scores=False) for i in range(num_clients)]
        server = NFJDServer(model=model, clients=clients, objective_fn=objective_fn,
                            participation_rate=participation_rate, learning_rate=learning_rate,
                            device=device, global_momentum_beta=0.0,
                            parallel_clients=False, eval_dataset=fed_data.val_dataset)
        trainer = NFJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "fedjd":
        clients = [FedJDClient(client_id=i, dataset=fed_data.client_datasets[i], batch_size=32, device=device, use_full_loader=True, local_epochs=local_epochs) for i in range(num_clients)]
        aggregator = MinNormAggregator(max_iters=250, lr=0.1, max_direction_norm=0.0)
        server = FedJDServer(model=model, clients=clients, aggregator=aggregator, objective_fn=objective_fn, participation_rate=participation_rate, learning_rate=learning_rate, device=device, eval_dataset=fed_data.val_dataset)
        trainer = FedJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "fmgda":
        clients = [FMGDAClient(client_id=i, dataset=fed_data.client_datasets[i], batch_size=32, device=device, learning_rate=learning_rate, local_epochs=local_epochs) for i in range(num_clients)]
        server = FMGDAServer(model=model, clients=clients, objective_fn=objective_fn, participation_rate=participation_rate, learning_rate=learning_rate, device=device, eval_dataset=fed_data.val_dataset, num_objectives=m)
        trainer = FedJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "weighted_sum":
        clients = [FedJDClient(client_id=i, dataset=fed_data.client_datasets[i], batch_size=32, device=device, use_full_loader=True, local_epochs=local_epochs) for i in range(num_clients)]
        server = WeightedSumServer(model=model, clients=clients, objective_fn=objective_fn, participation_rate=participation_rate, learning_rate=learning_rate, device=device, eval_dataset=fed_data.val_dataset)
        trainer = FedJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "direction_avg":
        clients = [FedJDClient(client_id=i, dataset=fed_data.client_datasets[i], batch_size=32, device=device, use_full_loader=True, local_epochs=local_epochs) for i in range(num_clients)]
        server = DirectionAvgServer(model=model, clients=clients, objective_fn=objective_fn, participation_rate=participation_rate, learning_rate=learning_rate, device=device, eval_dataset=fed_data.val_dataset)
        trainer = FedJDTrainer(server=server, num_rounds=num_rounds)
    else:
        raise ValueError(f"Unknown method: {method}")

    start = time.time()
    initial_obj = trainer.server.evaluate_global_objectives()
    history = trainer.fit()
    elapsed = time.time() - start

    objective_summary = summarize_objective_history(initial_obj, [s.objective_values for s in history])
    final_obj = objective_summary["final_obj"]
    avg_ri = float(objective_summary["avg_ri"])
    all_decreased = bool(objective_summary["all_decreased"])
    round_summary = summarize_round_history(history)
    avg_upload = round_summary["avg_upload_bytes"]
    avg_round_time = round_summary["avg_round_time"]
    upload_per_client = round_summary["upload_per_client"]

    avg_rescale = 1.0
    if method == "nfjd":
        rescale_vals = [s.avg_rescale_factor for s in history]
        avg_rescale = sum(rescale_vals) / len(rescale_vals) if rescale_vals else 1.0

    row = {
        "exp_id": exp_id, "method": method, "dataset": dataset, "m": m, "seed": seed,
        "num_rounds": num_rounds, "num_clients": num_clients,
        "participation_rate": participation_rate, "learning_rate": learning_rate,
        "conflict_strength": conflict_strength,
        "model_size": model_size, "local_epochs": local_epochs if method in ("nfjd", "fmgda") else 1,
        "use_adaptive_rescaling": False,
        "use_stochastic_gramian": False,
        "elapsed_time": round(elapsed, 2), "all_decreased": all_decreased,
        "hypervolume": round(float(objective_summary["hypervolume"]), 6),
        "pareto_gap": round(float(objective_summary["pareto_gap"]), 6),
        "task_jfi": round(float(objective_summary["task_jfi"]), 6),
        "task_mmag": round(float(objective_summary["task_mmag"]), 6),
        "avg_relative_improvement": round(avg_ri, 6),
        "avg_ri": round(avg_ri, 6),
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

    logger.info("[%s] %s: RI=%.4f JFI=%.4f upload/client=%d rescale=%.2f",
                exp_id, method, avg_ri, float(objective_summary["task_jfi"]), upload_per_client, avg_rescale)
    return row


def _write_csv(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    SEEDS = [7, 42, 123]
    METHODS = ["nfjd", "fedjd", "fmgda", "weighted_sum", "direction_avg"]
    all_rows = []
    experiments = []

    for m in [2, 3, 5]:
        for method in METHODS:
            for seed in SEEDS:
                experiments.append(dict(method=method, dataset="synthetic_regression",
                    m=m, seed=seed, conflict_strength=0.0, num_rounds=50, model_size="small", local_epochs=3))

    for m in [2, 3, 5]:
        for method in METHODS:
            for seed in SEEDS:
                experiments.append(dict(method=method, dataset="highconflict_regression",
                    m=m, seed=seed, conflict_strength=1.0, num_rounds=50, model_size="small", local_epochs=3))

    for m in [2, 5]:
        for seed in SEEDS:
            experiments.append(dict(method="nfjd", dataset="highconflict_regression",
                m=m, seed=seed, conflict_strength=1.0, num_rounds=50, model_size="small",
                local_epochs=3, use_adaptive_rescaling=False))

    for m in [2, 5]:
        for seed in SEEDS:
            experiments.append(dict(method="nfjd", dataset="highconflict_regression",
                m=m, seed=seed, conflict_strength=1.0, num_rounds=50, model_size="small",
                local_epochs=3, use_stochastic_gramian=False))

    total = len(experiments)
    logger.info(f"Starting NFJD benchmark: {total} experiments")

    for idx, exp in enumerate(experiments):
        logger.info(f"[{idx+1}/{total}] Running {exp}...")
        try:
            row = _run_single(**exp)
            all_rows.append(row)
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()

    csv_path = RESULTS_DIR / "nfjd_results.csv"
    _write_csv(csv_path, all_rows)
    logger.info(f"NFJD benchmark complete! {len(all_rows)}/{total} experiments, saved to {csv_path}")


if __name__ == "__main__":
    main()
