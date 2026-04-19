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
    DirectionAvgServer, FedJDClient, FedJDServer, FedJDTrainer,
    FMGDAServer, NFJDClient, NFJDServer, NFJDTrainer, WeightedSumServer,
)
from fedjd.data import (
    make_federated_classification, make_high_conflict_federated_regression,
    make_synthetic_federated_regression,
)
from fedjd.metrics import extract_pareto_front, hypervolume
from fedjd.models import MODEL_REGISTRY, MultiTaskClassifier
from fedjd.problems import multi_objective_regression, multi_task_classification

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("e:/AIProject/results/nfjd_phase4")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_FIELDNAMES = [
    "exp_id", "method", "task_type", "dataset", "m", "seed", "num_rounds",
    "num_clients", "participation_rate", "learning_rate", "conflict_strength",
    "noniid_strength", "model_size", "local_epochs", "use_adaptive_rescaling",
    "use_stochastic_gramian", "conflict_aware_momentum",
    "elapsed_time", "all_decreased", "hypervolume", "pareto_gap",
    "avg_relative_improvement", "avg_upload_bytes", "avg_round_time",
    "upload_per_client", "avg_rescale_factor", "avg_cosine_sim", "avg_effective_beta",
]
MAX_M = 10
for i in range(MAX_M):
    ALL_FIELDNAMES.extend([f"init_obj_{i}", f"final_obj_{i}", f"delta_obj_{i}"])


def _run_regression(method, m, seed, model_size, num_rounds=50,
                    num_clients=10, participation_rate=0.5, learning_rate=0.01):
    exp_id = f"P4-reg-{method}-m{m}-{model_size}-seed{seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    fed_data = make_synthetic_federated_regression(
        num_clients=num_clients, samples_per_client=100, input_dim=8,
        num_objectives=m, seed=seed)

    model_cls = MODEL_REGISTRY[model_size]
    model = model_cls(input_dim=fed_data.input_dim, output_dim=m)
    objective_fn = multi_objective_regression

    return _run_common(exp_id, method, model, fed_data.client_datasets,
                       objective_fn, m, seed, device, num_rounds, num_clients,
                       participation_rate, learning_rate, model_size,
                       task_type="regression", dataset="synthetic_regression")


def _run_highconflict(method, m, seed, conflict_strength=1.0, num_rounds=50,
                      num_clients=10, participation_rate=0.5, learning_rate=0.01):
    exp_id = f"P4-hc-{method}-m{m}-cs{conflict_strength}-seed{seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    fed_data = make_high_conflict_federated_regression(
        num_clients=num_clients, samples_per_client=100, input_dim=8,
        num_objectives=m, conflict_strength=conflict_strength, seed=seed)

    model = MODEL_REGISTRY["small"](input_dim=fed_data.input_dim, output_dim=m)
    objective_fn = multi_objective_regression

    return _run_common(exp_id, method, model, fed_data.client_datasets,
                       objective_fn, m, seed, device, num_rounds, num_clients,
                       participation_rate, learning_rate, "small",
                       task_type="highconflict", dataset=f"highconflict_cs{conflict_strength}",
                       conflict_strength=conflict_strength)


def _run_classification(method, m, seed, noniid_strength=0.0, num_rounds=50,
                        num_clients=10, participation_rate=0.5, learning_rate=0.01):
    exp_id = f"P4-cls-{method}-m{m}-niid{noniid_strength}-seed{seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    fed_data = make_federated_classification(
        num_clients=num_clients, samples_per_client=128, input_dim=64,
        num_classes=10, num_tasks=m, noniid_strength=noniid_strength, seed=seed)

    model = MultiTaskClassifier(input_dim=fed_data.input_dim, hidden_dim=64,
                                num_classes=fed_data.num_classes, num_tasks=m)
    objective_fn = multi_task_classification

    return _run_common(exp_id, method, model, fed_data.client_datasets,
                       objective_fn, m, seed, device, num_rounds, num_clients,
                       participation_rate, learning_rate, "classifier",
                       task_type="classification", dataset=f"cls_niid{noniid_strength}",
                       noniid_strength=noniid_strength)


def _run_common(exp_id, method, model, client_datasets, objective_fn, m, seed,
                device, num_rounds, num_clients, participation_rate, learning_rate,
                model_size, task_type, dataset, conflict_strength=0.0,
                noniid_strength=0.0):

    if method == "nfjd":
        clients = [NFJDClient(client_id=i, dataset=client_datasets[i], batch_size=32,
                              device=device, local_epochs=3, learning_rate=learning_rate,
                              local_momentum_beta=0.9, use_adaptive_rescaling=True,
                              use_stochastic_gramian=True, stochastic_subset_size=4,
                              stochastic_seed=seed, conflict_aware_momentum=False,
                              momentum_min_beta=0.1) for i in range(num_clients)]
        server = NFJDServer(model=model, clients=clients, objective_fn=objective_fn,
                            participation_rate=participation_rate, learning_rate=learning_rate,
                            device=device, global_momentum_beta=0.9,
                            conflict_aware_momentum=False, momentum_min_beta=0.1)
        trainer = NFJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "fedjd":
        clients = [FedJDClient(client_id=i, dataset=client_datasets[i], batch_size=32, device=device) for i in range(num_clients)]
        aggregator = MinNormAggregator(max_iters=250, lr=0.1, max_direction_norm=0.0)
        server = FedJDServer(model=model, clients=clients, aggregator=aggregator, objective_fn=objective_fn, participation_rate=participation_rate, learning_rate=learning_rate, device=device)
        trainer = FedJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "fmgda":
        clients = [FedJDClient(client_id=i, dataset=client_datasets[i], batch_size=32, device=device) for i in range(num_clients)]
        server = FMGDAServer(model=model, clients=clients, objective_fn=objective_fn, participation_rate=participation_rate, learning_rate=learning_rate, device=device)
        trainer = FedJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "weighted_sum":
        clients = [FedJDClient(client_id=i, dataset=client_datasets[i], batch_size=32, device=device) for i in range(num_clients)]
        server = WeightedSumServer(model=model, clients=clients, objective_fn=objective_fn, participation_rate=participation_rate, learning_rate=learning_rate, device=device)
        trainer = FedJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "direction_avg":
        clients = [FedJDClient(client_id=i, dataset=client_datasets[i], batch_size=32, device=device) for i in range(num_clients)]
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
    avg_cosine_sim = 0.0
    avg_effective_beta = 0.9
    if method == "nfjd":
        rescale_vals = [s.avg_rescale_factor for s in history]
        avg_rescale = sum(rescale_vals) / len(rescale_vals) if rescale_vals else 1.0
        cosine_vals = [getattr(s, "avg_cosine_sim", 0.0) for s in history]
        avg_cosine_sim = sum(cosine_vals) / len(cosine_vals) if cosine_vals else 0.0
        beta_vals = [getattr(s, "effective_global_beta", 0.9) for s in history]
        avg_effective_beta = sum(beta_vals) / len(beta_vals) if beta_vals else 0.9

    local_epochs = 3 if method == "nfjd" else 1

    row = {
        "exp_id": exp_id, "method": method, "task_type": task_type,
        "dataset": dataset, "m": m, "seed": seed, "num_rounds": num_rounds,
        "num_clients": num_clients, "participation_rate": participation_rate,
        "learning_rate": learning_rate, "conflict_strength": conflict_strength,
        "noniid_strength": noniid_strength, "model_size": model_size,
        "local_epochs": local_epochs,
        "use_adaptive_rescaling": method == "nfjd",
        "use_stochastic_gramian": method == "nfjd",
        "conflict_aware_momentum": False,
        "elapsed_time": round(elapsed, 2), "all_decreased": all_decreased,
        "hypervolume": round(hv, 6), "pareto_gap": round(pg, 6),
        "avg_relative_improvement": round(avg_ri, 6),
        "avg_upload_bytes": round(avg_upload, 0),
        "avg_round_time": round(avg_round_time, 4),
        "upload_per_client": round(upload_per_client, 0),
        "avg_rescale_factor": round(avg_rescale, 4),
        "avg_cosine_sim": round(avg_cosine_sim, 4),
        "avg_effective_beta": round(avg_effective_beta, 4),
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

    logger.info("[%s] %s: RI=%.4f NHV=%.4f time=%.1fs", exp_id, method, avg_ri, hv, elapsed)
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

    for model_size in ["small", "medium"]:
        for m in [2, 3, 5, 10]:
            for method in METHODS:
                for seed in SEEDS:
                    experiments.append(dict(
                        type="regression", method=method, m=m, seed=seed,
                        model_size=model_size))

    for m in [2, 3, 5]:
        for method in METHODS:
            for seed in SEEDS:
                experiments.append(dict(
                    type="highconflict", method=method, m=m, seed=seed,
                    conflict_strength=1.0))

    for m in [2, 3]:
        for noniid in [0.0, 0.3, 0.6, 0.9]:
            for method in METHODS:
                for seed in SEEDS:
                    experiments.append(dict(
                        type="classification", method=method, m=m, seed=seed,
                        noniid_strength=noniid))

    total = len(experiments)
    logger.info(f"Starting NFJD Phase 4 Full Benchmark: {total} experiments")

    reg_count = sum(1 for e in experiments if e["type"] == "regression")
    hc_count = sum(1 for e in experiments if e["type"] == "highconflict")
    cls_count = sum(1 for e in experiments if e["type"] == "classification")
    logger.info(f"  Regression: {reg_count}, HighConflict: {hc_count}, Classification: {cls_count}")

    for idx, exp in enumerate(experiments):
        exp_type = exp.pop("type")
        logger.info(f"[{idx+1}/{total}] Running {exp_type} {exp}...")

        try:
            if exp_type == "regression":
                row = _run_regression(**exp)
            elif exp_type == "highconflict":
                row = _run_highconflict(**exp)
            elif exp_type == "classification":
                row = _run_classification(**exp)
            else:
                raise ValueError(f"Unknown type: {exp_type}")
            all_rows.append(row)
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()

    csv_path = RESULTS_DIR / "phase4_results.csv"
    _write_csv(csv_path, all_rows)
    logger.info(f"Phase 4 complete! {len(all_rows)}/{total} experiments, saved to {csv_path}")


if __name__ == "__main__":
    main()
