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
from fedjd.experiments.nfjd_phases.phase5_utils import evaluate_model, fill_regression_metrics, fill_classification_metrics
from fedjd.data import (
    make_federated_classification, make_high_conflict_federated_regression,
    make_synthetic_federated_regression,
)
from fedjd.models import MODEL_REGISTRY, MultiTaskClassifier
from fedjd.problems import multi_objective_regression, multi_task_classification

RESULTS_DIR = Path("results/nfjd_phase4")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(RESULTS_DIR / "p4_run.log", mode="w"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

ALL_FIELDNAMES = [
    "exp_id", "method", "task_type", "dataset", "m", "seed", "num_rounds",
    "num_clients", "participation_rate", "learning_rate", "conflict_strength",
    "noniid_strength", "model_size", "local_epochs", "use_adaptive_rescaling",
    "use_stochastic_gramian", "conflict_aware_momentum",
    "elapsed_time", "avg_upload_bytes", "avg_round_time",
    "upload_per_client",
    "avg_accuracy", "avg_f1", "min_task_acc", "min_task_f1",
    "avg_mse", "max_mse", "mse_std", "avg_r2",
    "total_local_steps", "total_local_epoch_budget", "fair_comparison",
]


def _run_common(exp_id, method, model, client_datasets, objective_fn, m, seed,
                device, num_rounds, num_clients, participation_rate, learning_rate,
                model_size, task_type, dataset, conflict_strength=0.0,
                noniid_strength=0.0, local_epochs_override=None,
                fair_comparison=False, eval_dataset=None):

    le = local_epochs_override if local_epochs_override else 3

    if method == "nfjd":
        clients = [NFJDClient(client_id=i, dataset=client_datasets[i], batch_size=32,
                              device=device, local_epochs=le, learning_rate=learning_rate,
                              local_momentum_beta=0.0, use_adaptive_rescaling=False,
                              use_stochastic_gramian=False, stochastic_subset_size=4,
                              stochastic_seed=seed + i, conflict_aware_momentum=False,
                              momentum_min_beta=0.1, recompute_interval=1,
                              exact_upgrad=True, use_objective_normalization=True,
                              upload_align_scores=False) for i in range(num_clients)]
        server = NFJDServer(model=model, clients=clients, objective_fn=objective_fn,
                            participation_rate=participation_rate, learning_rate=learning_rate,
                            device=device, global_momentum_beta=0.0,
                            conflict_aware_momentum=False, momentum_min_beta=0.1,
                            parallel_clients=False, eval_dataset=eval_dataset,
                            use_global_progress_weights=True,
                            progress_beta=2.0, progress_min_weight=0.5,
                            progress_max_weight=2.0)
        trainer = NFJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "fedjd":
        clients = [FedJDClient(client_id=i, dataset=client_datasets[i], batch_size=32, device=device, use_full_loader=True, local_epochs=le) for i in range(num_clients)]
        aggregator = MinNormAggregator(max_iters=250, lr=0.1, max_direction_norm=0.0)
        server = FedJDServer(model=model, clients=clients, aggregator=aggregator, objective_fn=objective_fn, participation_rate=participation_rate, learning_rate=learning_rate, device=device, eval_dataset=eval_dataset)
        trainer = FedJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "fmgda":
        clients = [FMGDAClient(client_id=i, dataset=client_datasets[i], batch_size=32, device=device, learning_rate=learning_rate, local_epochs=le) for i in range(num_clients)]
        server = FMGDAServer(model=model, clients=clients, objective_fn=objective_fn, participation_rate=participation_rate, learning_rate=learning_rate, device=device, eval_dataset=eval_dataset, num_objectives=m)
        trainer = FedJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "weighted_sum":
        clients = [FedJDClient(client_id=i, dataset=client_datasets[i], batch_size=32, device=device, use_full_loader=True, local_epochs=le) for i in range(num_clients)]
        server = WeightedSumServer(model=model, clients=clients, objective_fn=objective_fn, participation_rate=participation_rate, learning_rate=learning_rate, device=device, eval_dataset=eval_dataset)
        trainer = FedJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "direction_avg":
        clients = [FedJDClient(client_id=i, dataset=client_datasets[i], batch_size=32, device=device, use_full_loader=True, local_epochs=le) for i in range(num_clients)]
        server = DirectionAvgServer(model=model, clients=clients, objective_fn=objective_fn, participation_rate=participation_rate, learning_rate=learning_rate, device=device, eval_dataset=eval_dataset)
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
    predictions, targets = evaluate_model(trainer.server.model, eval_dataset, device, batch_size=256)

    local_epochs = le
    total_local_steps = local_epochs * sum(max(int(getattr(s, "num_sampled_clients", 0)), 0) for s in history)

    row = {
        "exp_id": exp_id, "method": method, "task_type": task_type,
        "dataset": dataset, "m": m, "seed": seed, "num_rounds": num_rounds,
        "num_clients": num_clients, "participation_rate": participation_rate,
        "learning_rate": learning_rate, "conflict_strength": conflict_strength,
        "noniid_strength": noniid_strength, "model_size": model_size,
        "local_epochs": local_epochs,
        "use_adaptive_rescaling": False,
        "use_stochastic_gramian": False,
        "conflict_aware_momentum": False,
        "elapsed_time": round(elapsed, 2),
        "avg_upload_bytes": round(avg_upload, 0),
        "avg_round_time": round(avg_round_time, 4),
        "upload_per_client": round(upload_per_client, 0),
        "avg_accuracy": "", "avg_f1": "", "min_task_acc": "", "min_task_f1": "",
        "avg_mse": "", "max_mse": "", "mse_std": "", "avg_r2": "",
        "total_local_steps": total_local_steps,
        "total_local_epoch_budget": total_local_steps,
        "fair_comparison": fair_comparison,
    }
    if task_type == "classification":
        row = fill_classification_metrics(row, predictions, targets, m)
        logger.info("[%s] %s: acc=%.4f f1=%.4f min_acc=%.4f time=%.1fs", exp_id, method, float(row["avg_accuracy"]), float(row["avg_f1"]), float(row["min_task_acc"]), elapsed)
    else:
        row = fill_regression_metrics(row, predictions, targets, m)
        logger.info("[%s] %s: MSE=%.6f maxMSE=%.6f R2=%.6f time=%.1fs", exp_id, method, float(row["avg_mse"]), float(row["max_mse"]), float(row["avg_r2"]), elapsed)
    return row


def _run_regression(method, m, seed, model_size, num_rounds=50,
                    num_clients=10, participation_rate=0.5, learning_rate=0.01,
                    local_epochs_override=None, fair_comparison=False):
    if fair_comparison:
        raise ValueError("fair_comparison is deprecated: the default setup already uses matched local epochs and rounds across methods.")

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
                        task_type="regression", dataset="synthetic_regression",
                        local_epochs_override=local_epochs_override,
                        fair_comparison=fair_comparison, eval_dataset=fed_data.val_dataset)


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
                        conflict_strength=conflict_strength, eval_dataset=fed_data.val_dataset)


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
                        noniid_strength=noniid_strength, eval_dataset=fed_data.val_dataset)


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
    fair_count = sum(1 for e in experiments if e["type"] == "fair_regression")
    logger.info(f"  Regression: {reg_count}, HighConflict: {hc_count}, Classification: {cls_count}, FairComparison: {fair_count}")

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
