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
from fedjd.experiments.nfjd_phases.metric_utils import summarize_round_history
from fedjd.experiments.nfjd_phases.phase5_utils import evaluate_model, fill_regression_metrics
from fedjd.data import make_high_conflict_federated_regression
from fedjd.models import SmallRegressor
from fedjd.problems import multi_objective_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/nfjd_phase3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_FIELDNAMES = [
    "exp_id", "method", "dataset", "m", "seed", "num_rounds", "num_clients",
    "participation_rate", "learning_rate", "conflict_strength",
    "model_size", "local_epochs", "use_adaptive_rescaling",
    "use_stochastic_gramian", "conflict_aware_momentum",
    "elapsed_time", "avg_upload_bytes", "avg_round_time",
    "upload_per_client", "avg_mse", "max_mse", "mse_std", "avg_r2",
]


def _run_single(method, m, seed, conflict_strength, num_rounds=50,
                num_clients=10, participation_rate=0.5, learning_rate=0.01,
                model_size="small", local_epochs=3, use_adaptive_rescaling=True,
                use_stochastic_gramian=True, conflict_aware_momentum=False):
    exp_id = f"P3-{method}-m{m}-cs{conflict_strength}-seed{seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    fed_data = make_high_conflict_federated_regression(
        num_clients=num_clients, samples_per_client=100, input_dim=8,
        num_objectives=m, conflict_strength=conflict_strength, seed=seed)

    model = SmallRegressor(input_dim=fed_data.input_dim, output_dim=m)
    objective_fn = multi_objective_regression

    if method == "nfjd":
        clients = [NFJDClient(client_id=i, dataset=fed_data.client_datasets[i], batch_size=32,
                              device=device, local_epochs=local_epochs, learning_rate=learning_rate,
                              local_momentum_beta=0.0, use_adaptive_rescaling=False,
                              use_stochastic_gramian=False, stochastic_subset_size=4,
                              stochastic_seed=seed + i, conflict_aware_momentum=False,
                              momentum_min_beta=0.1, recompute_interval=1,
                              exact_upgrad=True, use_objective_normalization=True,
                              upload_align_scores=False) for i in range(num_clients)]
        server = NFJDServer(model=model, clients=clients, objective_fn=objective_fn,
                            participation_rate=participation_rate, learning_rate=learning_rate,
                            device=device, global_momentum_beta=0.0,
                            conflict_aware_momentum=False,
                            momentum_min_beta=0.1, parallel_clients=False,
                            eval_dataset=fed_data.val_dataset,
                            use_global_progress_weights=True,
                            progress_beta=2.0, progress_min_weight=0.5,
                            progress_max_weight=2.0)
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
    history = trainer.fit()
    elapsed = time.time() - start
    round_summary = summarize_round_history(history)
    avg_upload = round_summary["avg_upload_bytes"]
    avg_round_time = round_summary["avg_round_time"]
    upload_per_client = round_summary["upload_per_client"]
    predictions, targets = evaluate_model(trainer.server.model, fed_data.test_dataset, device, batch_size=256)

    row = {
        "exp_id": exp_id, "method": method, "dataset": f"highconflict_cs{conflict_strength}",
        "m": m, "seed": seed, "num_rounds": num_rounds, "num_clients": num_clients,
        "participation_rate": participation_rate, "learning_rate": learning_rate,
        "conflict_strength": conflict_strength,
        "model_size": model_size, "local_epochs": local_epochs if method in ("nfjd", "fmgda") else 1,
        "use_adaptive_rescaling": False,
        "use_stochastic_gramian": False,
        "conflict_aware_momentum": False,
        "elapsed_time": round(elapsed, 2),
        "avg_upload_bytes": round(avg_upload, 0),
        "avg_round_time": round(avg_round_time, 4),
        "upload_per_client": round(upload_per_client, 0),
        "avg_mse": "", "max_mse": "", "mse_std": "", "avg_r2": "",
    }
    row = fill_regression_metrics(row, predictions, targets, m)

    logger.info("[%s] %s: MSE=%.6f maxMSE=%.6f R2=%.6f cs=%.2f time=%.1fs",
                exp_id, method, float(row["avg_mse"]), float(row["max_mse"]), float(row["avg_r2"]), conflict_strength, elapsed)
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
    M_VALUES = [2, 3, 5]
    CONFLICT_STRENGTHS = [0.5, 1.0, 2.0]
    all_rows = []
    experiments = []

    for cs in CONFLICT_STRENGTHS:
        for m in M_VALUES:
            for method in METHODS:
                for seed in SEEDS:
                    experiments.append(dict(method=method, m=m, seed=seed,
                        conflict_strength=cs, conflict_aware_momentum=False))

    for cs in CONFLICT_STRENGTHS:
        for m in M_VALUES:
            for seed in SEEDS:
                experiments.append(dict(method="nfjd", m=m, seed=seed,
                    conflict_strength=cs, conflict_aware_momentum=True))

    total = len(experiments)
    logger.info(f"Starting NFJD Phase 3 High-Conflict: {total} experiments")

    for idx, exp in enumerate(experiments):
        logger.info(f"[{idx+1}/{total}] Running {exp}...")
        try:
            row = _run_single(**exp)
            all_rows.append(row)
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()

    csv_path = RESULTS_DIR / "phase3_results.csv"
    _write_csv(csv_path, all_rows)
    logger.info(f"Phase 3 complete! {len(all_rows)}/{total} experiments, saved to {csv_path}")


if __name__ == "__main__":
    main()
