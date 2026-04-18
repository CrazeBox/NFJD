from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from fedjd.aggregators import MinNormAggregator
from fedjd.compressors import Float16Compressor, NoCompressor
from fedjd.core.baselines import DirectionAvgServer, FMGDAServer, WeightedSumServer
from fedjd.core.client import FedJDClient
from fedjd.core.server import FedJDServer
from fedjd.core.trainer import FedJDTrainer
from fedjd.data.classification import make_federated_classification
from fedjd.data.synthetic import make_synthetic_federated_regression
from fedjd.metrics import extract_pareto_front, hypervolume
from fedjd.models.classifier import MultiTaskClassifier
from fedjd.models.small_regressor import MediumRegressor, SmallRegressor
from fedjd.problems.classification import multi_task_classification
from fedjd.problems.regression import multi_objective_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("e:/AIProject/results/s4_benchmark")

ALL_FIELDNAMES = [
    "exp_id", "method", "task", "m", "seed", "num_rounds", "num_clients",
    "participation_rate", "learning_rate", "noniid_strength", "use_float16",
    "model_size", "elapsed_time", "all_decreased", "any_nan", "total_nan_inf",
    "hypervolume", "pareto_gap", "num_pareto_points", "avg_upload_bytes",
    "avg_round_time", "upload_per_client", "avg_relative_improvement",
]
MAX_M = 10
for i in range(MAX_M):
    ALL_FIELDNAMES.extend([f"init_obj_{i}", f"final_obj_{i}", f"delta_obj_{i}"])


def _make_key(method, task, m, seed, **kwargs):
    parts = [f"S4-{method}-{task}-m{m}"]
    for k, v in sorted(kwargs.items()):
        if isinstance(v, float):
            parts.append(f"{k}{v:.1f}")
        else:
            parts.append(f"{k}{v}")
    parts.append(f"seed{seed}")
    return "-".join(parts)


def _run_single(
    method: str,
    task: str,
    m: int,
    seed: int,
    num_rounds: int = 50,
    num_clients: int = 10,
    participation_rate: float = 0.5,
    learning_rate: float = 0.01,
    noniid_strength: float = 0.0,
    use_float16: bool = False,
    model_size: str = "small",
    save_folder: bool = False,
) -> dict:
    exp_id = _make_key(method, task, m, seed, noniid=noniid_strength, model=model_size, float16=use_float16)
    device = torch.device("cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    if task == "regression":
        data = make_synthetic_federated_regression(
            num_clients=num_clients, samples_per_client=64,
            input_dim=8, num_objectives=m, noise_std=0.1, seed=seed,
        )
        if model_size == "medium":
            model = MediumRegressor(input_dim=8, hidden_dim=64, output_dim=m)
        else:
            model = SmallRegressor(input_dim=8, hidden_dim=16, output_dim=m)
        objective_fn = multi_objective_regression
    elif task == "classification":
        data = make_federated_classification(
            num_clients=num_clients, samples_per_client=128,
            input_dim=64, num_classes=10, num_tasks=m,
            noniid_strength=noniid_strength, seed=seed,
        )
        hidden = 128 if model_size == "medium" else 64
        model = MultiTaskClassifier(input_dim=64, hidden_dim=hidden, num_classes=10, num_tasks=m)
        objective_fn = multi_task_classification
    else:
        raise ValueError(f"Unknown task: {task}")

    clients = [
        FedJDClient(client_id=i, dataset=ds, batch_size=32, device=device)
        for i, ds in enumerate(data.client_datasets)
    ]

    compressor = Float16Compressor() if use_float16 else NoCompressor()
    aggregator = MinNormAggregator(max_iters=250, lr=0.1, max_direction_norm=0.0)

    if method == "fedjd":
        server = FedJDServer(
            model=model, clients=clients, aggregator=aggregator,
            objective_fn=objective_fn, participation_rate=participation_rate,
            learning_rate=learning_rate, device=device, compressor=compressor,
        )
    elif method == "fmgda":
        server = FMGDAServer(
            model=model, clients=clients, objective_fn=objective_fn,
            participation_rate=participation_rate, learning_rate=learning_rate,
            device=device,
        )
    elif method == "weighted_sum":
        server = WeightedSumServer(
            model=model, clients=clients, objective_fn=objective_fn,
            participation_rate=participation_rate, learning_rate=learning_rate,
            device=device,
        )
    elif method == "direction_avg":
        server = DirectionAvgServer(
            model=model, clients=clients, objective_fn=objective_fn,
            participation_rate=participation_rate, learning_rate=learning_rate,
            device=device,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    output_dir = None
    if save_folder:
        output_dir = str(RESULTS_DIR / exp_id)

    trainer = FedJDTrainer(
        server=server, num_rounds=num_rounds, output_dir=output_dir,
        save_checkpoints=False,
    )

    start = time.time()
    history = trainer.fit()
    elapsed = time.time() - start

    initial_obj = history[0].objective_values
    final_obj = history[-1].objective_values
    obj_history = [s.objective_values for s in history]

    all_decreased = all(final_obj[j] <= initial_obj[j] for j in range(m))
    any_nan = any(s.nan_inf_count > 0 for s in history)
    total_nan_inf = sum(s.nan_inf_count for s in history)

    min_vals = [min(h[j] for h in obj_history) for j in range(m)]
    max_vals = [max(h[j] for h in obj_history) for j in range(m)]
    ranges = [max_vals[j] - min_vals[j] for j in range(m)]
    for j in range(m):
        if ranges[j] < 1e-10:
            ranges[j] = 1.0

    normalized_history = []
    for h in obj_history:
        normalized_history.append([(h[j] - min_vals[j]) / ranges[j] for j in range(m)])

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

    avg_upload = sum(s.upload_bytes for s in history if s.is_full_sync_round) / max(sum(1 for s in history if s.is_full_sync_round), 1)
    avg_round_time = sum(s.round_time for s in history) / max(len(history), 1)

    upload_per_client = 0
    for s in history:
        if s.is_full_sync_round and s.jacobian_upload_per_client > 0:
            upload_per_client = s.jacobian_upload_per_client
            break
        elif s.is_full_sync_round and s.gradient_upload_per_client > 0:
            upload_per_client = s.gradient_upload_per_client
            break

    row = {
        "exp_id": exp_id,
        "method": method,
        "task": task,
        "m": m,
        "seed": seed,
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "participation_rate": participation_rate,
        "learning_rate": learning_rate,
        "noniid_strength": noniid_strength,
        "use_float16": use_float16,
        "model_size": model_size,
        "elapsed_time": round(elapsed, 2),
        "all_decreased": all_decreased,
        "any_nan": any_nan,
        "total_nan_inf": total_nan_inf,
        "hypervolume": round(hv, 6),
        "pareto_gap": round(pg, 6),
        "num_pareto_points": len(pareto_front),
        "avg_upload_bytes": round(avg_upload, 0),
        "avg_round_time": round(avg_round_time, 4),
        "upload_per_client": upload_per_client,
        "avg_relative_improvement": round(avg_ri, 6),
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

    return row


def _write_csv(csv_path: Path, rows: list[dict]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_rows = []
    count = 0

    methods = ["fedjd", "fmgda", "weighted_sum", "direction_avg"]
    seeds = [7, 42, 123]

    logger.info("=" * 60)
    logger.info("Group A: Method Comparison")
    logger.info("=" * 60)
    tasks_a = [
        ("regression", 2, "small", 0.0),
        ("regression", 3, "small", 0.0),
        ("regression", 5, "medium", 0.0),
        ("classification", 2, "small", 0.0),
        ("classification", 3, "small", 0.0),
    ]
    for method in methods:
        for task, m, model_size, noniid in tasks_a:
            for seed in seeds:
                is_key = (method in ("fedjd", "fmgda") and m == 2 and seed == 7) or \
                         (method == "fedjd" and task == "classification" and seed == 7)
                try:
                    row = _run_single(method=method, task=task, m=m, seed=seed,
                                      num_rounds=50, model_size=model_size,
                                      noniid_strength=noniid, save_folder=is_key)
                    all_rows.append(row)
                    count += 1
                    logger.info("[%d] %s: HV=%.4f PG=%.4f dec=%s",
                                count, row["exp_id"], row["hypervolume"],
                                row["pareto_gap"], row["all_decreased"])
                except Exception as e:
                    logger.error("FAILED %s-%s-m%d-seed%d: %s", method, task, m, seed, e)

    logger.info("=" * 60)
    logger.info("Group B: Non-IID Ablation")
    logger.info("=" * 60)
    for method in ["fedjd", "fmgda", "weighted_sum"]:
        for noniid in [0.0, 0.3, 0.6, 0.9]:
            for seed in seeds:
                try:
                    row = _run_single(method=method, task="classification", m=2, seed=seed,
                                      num_rounds=50, noniid_strength=noniid, save_folder=False)
                    all_rows.append(row)
                    count += 1
                    logger.info("[%d] %s: HV=%.4f noniid=%.1f",
                                count, row["exp_id"], row["hypervolume"], noniid)
                except Exception as e:
                    logger.error("FAILED %s-noniid%.1f-seed%d: %s", method, noniid, seed, e)

    logger.info("=" * 60)
    logger.info("Group C: Objective Scaling")
    logger.info("=" * 60)
    for method in ["fedjd", "fmgda", "weighted_sum"]:
        for m in [5, 10]:
            for seed in seeds:
                model_size = "medium" if m >= 5 else "small"
                try:
                    row = _run_single(method=method, task="regression", m=m, seed=seed,
                                      num_rounds=50, model_size=model_size, save_folder=False)
                    all_rows.append(row)
                    count += 1
                    logger.info("[%d] %s: HV=%.4f m=%d",
                                count, row["exp_id"], row["hypervolume"], m)
                except Exception as e:
                    logger.error("FAILED %s-m%d-seed%d: %s", method, m, seed, e)

    logger.info("=" * 60)
    logger.info("Group D: Communication Efficiency (float16)")
    logger.info("=" * 60)
    for m in [2, 3, 5]:
        for seed in seeds:
            model_size = "medium" if m >= 5 else "small"
            try:
                row = _run_single(method="fedjd", task="regression", m=m, seed=seed,
                                  num_rounds=50, model_size=model_size,
                                  use_float16=True, save_folder=False)
                all_rows.append(row)
                count += 1
                logger.info("[%d] %s: HV=%.4f upload=%.0f float16=True",
                            count, row["exp_id"], row["hypervolume"], row["avg_upload_bytes"])
            except Exception as e:
                logger.error("FAILED fedjd-f16-m%d-seed%d: %s", m, seed, e)

    csv_path = RESULTS_DIR / "s4_results.csv"
    _write_csv(csv_path, all_rows)
    logger.info("=" * 60)
    logger.info("Stage 4 complete! %d experiments, saved to %s", count, csv_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    run_all()
