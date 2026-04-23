from __future__ import annotations

import csv
import gc
import logging
import time

import numpy as np
import torch

from fedjd.core import (
    FMGDAClient, FMGDAServer,
    NFJDClient, NFJDServer, NFJDTrainer,
    PHASE5_FORMAL_BASELINES, Phase5OfficialBaselineClient,
    Phase5OfficialBaselineServer, FedJDTrainer, get_phase5_method_spec,
)
from fedjd.experiments.nfjd_phases.metric_utils import summarize_objective_history, summarize_round_history
from fedjd.metrics import (
    jain_fairness_index, min_max_gap, compute_f1_scores,
    compute_accuracy, compute_mse_per_task,
)

logger = logging.getLogger(__name__)

ALL_FIELDNAMES = [
    "exp_id", "method", "dataset", "data_split", "m", "seed", "num_rounds",
    "num_clients", "participation_rate", "learning_rate", "local_epochs",
    "use_adaptive_rescaling", "use_stochastic_gramian", "conflict_aware_momentum",
    "use_objective_normalization", "exact_upgrad",
    "model_arch", "total_local_steps",
    "elapsed_time", "all_decreased", "avg_ri",
    "avg_upload_bytes", "avg_round_time", "upload_per_client",
    "avg_rescale_factor", "avg_cosine_sim", "avg_effective_beta",
    "avg_accuracy", "avg_f1", "task_jfi", "task_mmag",
    "avg_mse", "max_mse", "mse_std",
]
MAX_M = 8
for _i in range(MAX_M):
    ALL_FIELDNAMES.extend([f"init_obj_{_i}", f"final_obj_{_i}", f"delta_obj_{_i}"])
for _i in range(MAX_M):
    ALL_FIELDNAMES.extend([f"task_{_i}_acc", f"task_{_i}_f1", f"task_{_i}_mse"])


def build_trainer(method, model, client_datasets, objective_fn, m, seed,
                  device, num_rounds, num_clients, participation_rate,
                  learning_rate, local_epochs=1, eval_dataset=None):
    if method == "nfjd":
        clients = [NFJDClient(
            client_id=i, dataset=client_datasets[i], batch_size=256,
            device=device, local_epochs=local_epochs, learning_rate=learning_rate,
            local_momentum_beta=0.0, use_adaptive_rescaling=False,
            use_stochastic_gramian=False, stochastic_subset_size=min(4, m),
            stochastic_seed=seed + i, conflict_aware_momentum=False,
            momentum_min_beta=0.1,
            recompute_interval=1,
            exact_upgrad=True,
            use_objective_normalization=True,
            upload_align_scores=False,
        ) for i in range(num_clients)]
        server = NFJDServer(
            model=model, clients=clients, objective_fn=objective_fn,
            participation_rate=participation_rate, learning_rate=learning_rate,
            device=device, global_momentum_beta=0.0,
            conflict_aware_momentum=False, momentum_min_beta=0.1,
            parallel_clients=False, eval_dataset=eval_dataset,
        )
        return NFJDTrainer(server=server, num_rounds=num_rounds)

    if method == "fmgda":
        clients = [
            FMGDAClient(
                client_id=i,
                dataset=client_datasets[i],
                batch_size=256,
                device=device,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
            )
            for i in range(num_clients)
        ]
        server = FMGDAServer(
            model=model,
            clients=clients,
            objective_fn=objective_fn,
            participation_rate=participation_rate,
            learning_rate=learning_rate,
            device=device,
            eval_dataset=eval_dataset,
            num_objectives=m,
        )
        return FedJDTrainer(server=server, num_rounds=num_rounds)

    if method in PHASE5_FORMAL_BASELINES:
        clients = [
            Phase5OfficialBaselineClient(
                client_id=i,
                dataset=client_datasets[i],
                batch_size=256,
                device=device,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
                method=method,
                seed=seed + i,
            )
            for i in range(num_clients)
        ]
        server = Phase5OfficialBaselineServer(
            model=model,
            clients=clients,
            objective_fn=objective_fn,
            participation_rate=participation_rate,
            device=device,
            method_name=method,
            eval_dataset=eval_dataset,
        )
        return FedJDTrainer(server=server, num_rounds=num_rounds)

    raise ValueError(f"Unknown Phase 5 method: {method}")


def run_experiment(exp_id, method, model, client_datasets, objective_fn, m, seed,
                   device, num_rounds, num_clients, participation_rate, learning_rate,
                   model_arch, dataset, data_split, local_epochs=1,
                   eval_dataset=None):

    trainer = build_trainer(
        method=method, model=model, client_datasets=client_datasets,
        objective_fn=objective_fn, m=m, seed=seed, device=device,
        num_rounds=num_rounds, num_clients=num_clients,
        participation_rate=participation_rate, learning_rate=learning_rate,
        local_epochs=local_epochs, eval_dataset=eval_dataset,
    )

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
    avg_cosine_sim = 0.0
    avg_effective_beta = 0.9
    if method == "nfjd":
        rescale_vals = [s.avg_rescale_factor for s in history]
        avg_rescale = sum(rescale_vals) / len(rescale_vals) if rescale_vals else 1.0
        cosine_vals = [getattr(s, "avg_cosine_sim", 0.0) for s in history]
        avg_cosine_sim = sum(cosine_vals) / len(cosine_vals) if cosine_vals else 0.0
        beta_vals = [getattr(s, "effective_global_beta", 0.9) for s in history]
        avg_effective_beta = sum(beta_vals) / len(beta_vals) if beta_vals else 0.9

    total_local_steps = local_epochs * sum(max(int(getattr(s, "num_sampled_clients", 0)), 0) for s in history)
    spec = get_phase5_method_spec(method)

    row = {
        "exp_id": exp_id, "method": method, "dataset": dataset,
        "data_split": data_split, "m": m, "seed": seed,
        "num_rounds": num_rounds, "num_clients": num_clients,
        "participation_rate": participation_rate,
        "learning_rate": learning_rate, "local_epochs": local_epochs,
        "use_adaptive_rescaling": False,
        "use_stochastic_gramian": False,
        "conflict_aware_momentum": False,
        "use_objective_normalization": method == "nfjd",
        "exact_upgrad": method == "nfjd",
        "model_arch": model_arch,
        "total_local_steps": total_local_steps,
        "elapsed_time": round(elapsed, 2), "all_decreased": all_decreased,
        "avg_ri": round(avg_ri, 6),
        "avg_upload_bytes": round(avg_upload, 0),
        "avg_round_time": round(avg_round_time, 4),
        "upload_per_client": round(upload_per_client, 0),
        "avg_rescale_factor": round(avg_rescale, 4),
        "avg_cosine_sim": round(avg_cosine_sim, 4),
        "avg_effective_beta": round(avg_effective_beta, 4),
        "avg_accuracy": "", "avg_f1": "", "task_jfi": "", "task_mmag": "",
        "avg_mse": "", "max_mse": "", "mse_std": "",
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
    for i in range(MAX_M):
        row[f"task_{i}_acc"] = ""
        row[f"task_{i}_f1"] = ""
        row[f"task_{i}_mse"] = ""

    logger.info("[%s] %s (%s): RI=%.4f steps=%d time=%.1fs", exp_id, spec.display_name, spec.family, avg_ri, total_local_steps, elapsed)
    return row


def evaluate_model(model, test_dataset, device, batch_size=256):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            by = by.to(device)
            pred = model(bx)
            all_preds.append(pred.cpu())
            all_targets.append(by.cpu())
    return torch.cat(all_preds), torch.cat(all_targets)


def fill_classification_metrics(row, predictions, targets, m):
    accs = compute_accuracy(predictions, targets, m)
    f1s = compute_f1_scores(predictions, targets, m)
    row["avg_accuracy"] = round(sum(accs) / m, 4)
    row["avg_f1"] = round(sum(f1s) / m, 4)
    row["task_jfi"] = round(jain_fairness_index(accs), 4)
    row["task_mmag"] = round(min_max_gap(accs), 4)
    for i in range(m):
        row[f"task_{i}_acc"] = round(accs[i], 4)
        row[f"task_{i}_f1"] = round(f1s[i], 4)
    return row


def fill_regression_metrics(row, predictions, targets, m):
    mses = compute_mse_per_task(predictions, targets, m)
    row["avg_mse"] = round(sum(mses) / m, 6)
    row["max_mse"] = round(max(mses), 6)
    row["mse_std"] = round(float(np.std(mses)), 6)
    row["task_jfi"] = round(jain_fairness_index([1.0 / (ms + 1e-8) for ms in mses]), 4)
    row["task_mmag"] = round(min_max_gap(mses), 6)
    for i in range(m):
        row[f"task_{i}_mse"] = round(mses[i], 6)
    return row


def write_csv(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
