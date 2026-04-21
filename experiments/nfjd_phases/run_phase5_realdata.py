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
from fedjd.data.multimnist import make_multimnist
from fedjd.data.celeba import make_celeba
from fedjd.metrics import extract_pareto_front, hypervolume
from fedjd.models.lenet_mtl import LeNetMTL
from fedjd.models.celeba_cnn import CelebaCNN
from fedjd.problems import multi_objective_regression, multi_task_classification

RESULTS_DIR = Path("results/nfjd_phase5")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(RESULTS_DIR / "p5_run.log", mode="w"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

ALL_FIELDNAMES = [
    "exp_id", "method", "dataset", "data_split", "m", "seed", "num_rounds",
    "num_clients", "participation_rate", "learning_rate", "local_epochs",
    "use_adaptive_rescaling", "use_stochastic_gramian", "conflict_aware_momentum",
    "model_arch", "total_local_steps", "fair_comparison",
    "elapsed_time", "all_decreased", "hypervolume", "pareto_gap",
    "avg_relative_improvement", "avg_upload_bytes", "avg_round_time",
    "upload_per_client", "avg_rescale_factor", "avg_cosine_sim", "avg_effective_beta",
    "task_L_acc", "task_R_acc", "avg_accuracy",
    "per_task_mse", "avg_mse", "max_mse", "mse_std",
]
MAX_M = 10
for i in range(MAX_M):
    ALL_FIELDNAMES.extend([f"init_obj_{i}", f"final_obj_{i}", f"delta_obj_{i}"])


def _compute_accuracy(predictions, targets, num_tasks):
    correct = [0] * num_tasks
    total = 0
    with torch.no_grad():
        for t in range(num_tasks):
            pred_labels = predictions[:, t].argmax(dim=-1)
            true_labels = targets[:, t].long()
            correct[t] += (pred_labels == true_labels).sum().item()
        total += predictions.shape[0]
    return [c / max(total, 1) for c in correct]


def _run_multimnist(method, seed, iid=True, num_rounds=50,
                    num_clients=10, participation_rate=0.5, learning_rate=0.001,
                    fair_comparison=False):
    split_name = "iid" if iid else "noniid"
    exp_id = f"P5-mm-{method}-{split_name}-seed{seed}"
    if fair_comparison:
        exp_id = f"P5-fair-mm-{method}-{split_name}-seed{seed}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    data = make_multimnist(num_clients=num_clients, iid=iid, seed=seed)
    model = LeNetMTL(input_channels=1, num_tasks=2, num_classes=10)
    objective_fn = multi_task_classification

    if method == "nfjd":
        le = 3
        nr = num_rounds
        if fair_comparison:
            le = 3
            nr = num_rounds
    else:
        le = 1
        nr = num_rounds * 3 if fair_comparison else num_rounds

    row = _run_common(
        exp_id=exp_id, method=method, model=model,
        client_datasets=data["client_datasets"],
        objective_fn=objective_fn, m=2, seed=seed, device=device,
        num_rounds=nr, num_clients=num_clients,
        participation_rate=participation_rate, learning_rate=learning_rate,
        model_arch="lenet_mtl", task_type="classification",
        dataset="multimnist", data_split=split_name,
        local_epochs=le, fair_comparison=fair_comparison,
    )

    test_ds = data["test_dataset"]
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            by = by.to(device)
            pred = model(bx)
            all_preds.append(pred.cpu())
            all_targets.append(by.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    accs = _compute_accuracy(all_preds, all_targets, 2)
    row["task_L_acc"] = round(accs[0], 4)
    row["task_R_acc"] = round(accs[1], 4)
    row["avg_accuracy"] = round(sum(accs) / 2, 4)

    return row


def _run_celeba(method, seed, iid=True, num_rounds=50,
                num_clients=10, participation_rate=0.5, learning_rate=0.0001,
                num_tasks=4, fair_comparison=False):
    split_name = "iid" if iid else "noniid"
    exp_id = f"P5-ca-{method}-{split_name}-m{num_tasks}-seed{seed}"
    if fair_comparison:
        exp_id = f"P5-fair-ca-{method}-{split_name}-m{num_tasks}-seed{seed}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    # Create CelebA dataset
    train_datasets, val_datasets, test_datasets = make_celeba(
        num_clients=num_clients, iid=iid, seed=seed, num_tasks=num_tasks,
        root="/root/data/celeba/celeba", download=False
    )
    
    data = {
        "client_datasets": train_datasets,
        "test_dataset": torch.utils.data.ConcatDataset(test_datasets)
    }
    
    model = CelebaCNN(num_attributes=num_tasks)
    objective_fn = multi_objective_regression  # Use MSE for binary attributes

    if method == "nfjd":
        le = 3
        nr = num_rounds
    else:
        le = 1
        nr = num_rounds * 3 if fair_comparison else num_rounds

    row = _run_common(
        exp_id=exp_id, method=method, model=model,
        client_datasets=data["client_datasets"],
        objective_fn=objective_fn, m=num_tasks, seed=seed, device=device,
        num_rounds=nr, num_clients=num_clients,
        participation_rate=participation_rate, learning_rate=learning_rate,
        model_arch="celeba_cnn", task_type="regression",
        dataset="celeba", data_split=split_name,
        local_epochs=le, fair_comparison=fair_comparison,
    )

    # Evaluate on test set
    test_ds = data["test_dataset"]
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            by = by.to(device)
            pred = model(bx)
            all_preds.append(pred.cpu())
            all_targets.append(by.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    # Calculate MSE for each attribute
    per_task_mse = []
    for t in range(num_tasks):
        mse = ((all_preds[:, t] - all_targets[:, t]) ** 2).mean().item()
        per_task_mse.append(mse)
    row["per_task_mse"] = ",".join(f"{v:.6f}" for v in per_task_mse)
    row["avg_mse"] = round(sum(per_task_mse) / len(per_task_mse), 6)
    row["max_mse"] = round(max(per_task_mse), 6)
    import numpy as np
    row["mse_std"] = round(float(np.std(per_task_mse)), 6)

    return row


def _run_common(exp_id, method, model, client_datasets, objective_fn, m, seed,
                device, num_rounds, num_clients, participation_rate, learning_rate,
                model_arch, task_type, dataset, data_split,
                local_epochs=1, fair_comparison=False):

    if method == "nfjd":
        clients = [NFJDClient(client_id=i, dataset=client_datasets[i], batch_size=256,
                              device=device, local_epochs=local_epochs, learning_rate=learning_rate,
                              local_momentum_beta=0.9, use_adaptive_rescaling=True,
                              use_stochastic_gramian=True, stochastic_subset_size=min(4, m),
                              stochastic_seed=seed, conflict_aware_momentum=False,
                              momentum_min_beta=0.1) for i in range(num_clients)]
        server = NFJDServer(model=model, clients=clients, objective_fn=objective_fn,
                            participation_rate=participation_rate, learning_rate=learning_rate,
                            device=device, global_momentum_beta=0.9,
                            conflict_aware_momentum=False, momentum_min_beta=0.1,
                            parallel_clients=True)
        trainer = NFJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "fedjd":
        clients = [FedJDClient(client_id=i, dataset=client_datasets[i], batch_size=256, device=device) for i in range(num_clients)]
        aggregator = MinNormAggregator(max_iters=250, lr=0.1, max_direction_norm=0.0)
        server = FedJDServer(model=model, clients=clients, aggregator=aggregator, objective_fn=objective_fn, participation_rate=participation_rate, learning_rate=learning_rate, device=device)
        trainer = FedJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "fmgda":
        clients = [FedJDClient(client_id=i, dataset=client_datasets[i], batch_size=256, device=device) for i in range(num_clients)]
        server = FMGDAServer(model=model, clients=clients, objective_fn=objective_fn, participation_rate=participation_rate, learning_rate=learning_rate, device=device)
        trainer = FedJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "weighted_sum":
        clients = [FedJDClient(client_id=i, dataset=client_datasets[i], batch_size=256, device=device) for i in range(num_clients)]
        server = WeightedSumServer(model=model, clients=clients, objective_fn=objective_fn, participation_rate=participation_rate, learning_rate=learning_rate, device=device)
        trainer = FedJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "direction_avg":
        clients = [FedJDClient(client_id=i, dataset=client_datasets[i], batch_size=256, device=device) for i in range(num_clients)]
        server = DirectionAvgServer(model=model, clients=clients, objective_fn=objective_fn, participation_rate=participation_rate, learning_rate=learning_rate, device=device)
        trainer = FedJDTrainer(server=server, num_rounds=num_rounds)
    elif method == "stl":
        from fedjd.core.client import FedJDClient as _FJC
        single_ds = torch.utils.data.ConcatDataset(client_datasets)
        client = _FJC(client_id=0, dataset=single_ds, batch_size=256, device=device)
        server = WeightedSumServer(model=model, clients=[client], objective_fn=objective_fn, participation_rate=1.0, learning_rate=learning_rate, device=device)
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

    total_local_steps = local_epochs * num_rounds

    row = {
        "exp_id": exp_id, "method": method, "dataset": dataset,
        "data_split": data_split, "m": m, "seed": seed,
        "num_rounds": num_rounds, "num_clients": num_clients,
        "participation_rate": participation_rate,
        "learning_rate": learning_rate, "local_epochs": local_epochs,
        "use_adaptive_rescaling": method == "nfjd",
        "use_stochastic_gramian": method == "nfjd",
        "conflict_aware_momentum": False,
        "model_arch": model_arch,
        "total_local_steps": total_local_steps,
        "fair_comparison": fair_comparison,
        "elapsed_time": round(elapsed, 2), "all_decreased": all_decreased,
        "hypervolume": round(hv, 6), "pareto_gap": round(pg, 6),
        "avg_relative_improvement": round(avg_ri, 6),
        "avg_upload_bytes": round(avg_upload, 0),
        "avg_round_time": round(avg_round_time, 4),
        "upload_per_client": round(upload_per_client, 0),
        "avg_rescale_factor": round(avg_rescale, 4),
        "avg_cosine_sim": round(avg_cosine_sim, 4),
        "avg_effective_beta": round(avg_effective_beta, 4),
        "task_L_acc": "", "task_R_acc": "", "avg_accuracy": "",
        "per_task_mse": "", "avg_mse": "", "max_mse": "", "mse_std": "",
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

    logger.info("[%s] %s: RI=%.4f NHV=%.4f steps=%d time=%.1fs", exp_id, method, avg_ri, hv, total_local_steps, elapsed)
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

    for method in METHODS + ["stl"]:
        for seed in SEEDS:
            experiments.append(dict(
                type="multimnist", method=method, seed=seed, iid=True))

    for method in METHODS:
        for seed in SEEDS:
            experiments.append(dict(
                type="multimnist", method=method, seed=seed, iid=False))

    for method in METHODS:
        for seed in SEEDS:
            experiments.append(dict(
                type="multimnist_fair", method=method, seed=seed, iid=True))

    for method in METHODS + ["stl"]:
        for seed in SEEDS:
            experiments.append(dict(
                type="celeba", method=method, seed=seed, iid=True, num_tasks=4))

    for method in METHODS:
        for seed in SEEDS:
            experiments.append(dict(
                type="celeba", method=method, seed=seed, iid=False, num_tasks=4))

    for m in [2, 4, 6]:
        for method in ["nfjd", "fedjd"]:
            for seed in SEEDS:
                experiments.append(dict(
                    type="celeba_scale", method=method, seed=seed, iid=True, num_tasks=m))

    for method in METHODS:
        for seed in SEEDS:
            experiments.append(dict(
                type="celeba_fair", method=method, seed=seed, iid=True, num_tasks=4))

    total = len(experiments)
    logger.info(f"Starting NFJD Phase 5 Real Data Benchmark: {total} experiments")

    mm_iid = sum(1 for e in experiments if e["type"] == "multimnist" and e.get("iid"))
    mm_niid = sum(1 for e in experiments if e["type"] == "multimnist" and not e.get("iid"))
    mm_fair = sum(1 for e in experiments if e["type"] == "multimnist_fair")
    ca_base = sum(1 for e in experiments if e["type"] == "celeba")
    ca_scale = sum(1 for e in experiments if e["type"] == "celeba_scale")
    ca_fair = sum(1 for e in experiments if e["type"] == "celeba_fair")
    logger.info(f"  MultiMNIST IID: {mm_iid}, NonIID: {mm_niid}, Fair: {mm_fair}")
    logger.info(f"  CelebA Base: {ca_base}, Scale: {ca_scale}, Fair: {ca_fair}")

    for idx, exp in enumerate(experiments):
        exp_type = exp.pop("type")
        logger.info(f"[{idx+1}/{total}] Running {exp_type} {exp}...")

        try:
            if exp_type == "multimnist":
                row = _run_multimnist(**exp)
            elif exp_type == "multimnist_fair":
                row = _run_multimnist(**exp, fair_comparison=True)
            elif exp_type == "celeba":
                row = _run_celeba(**exp)
            elif exp_type == "celeba_scale":
                row = _run_celeba(**exp)
            elif exp_type == "celeba_fair":
                row = _run_celeba(**exp, fair_comparison=True)
            else:
                raise ValueError(f"Unknown type: {exp_type}")
            all_rows.append(row)
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    csv_path = RESULTS_DIR / "phase5_results.csv"
    _write_csv(csv_path, all_rows)
    logger.info(f"Phase 5 complete! {len(all_rows)}/{total} experiments, saved to {csv_path}")


if __name__ == "__main__":
    main()
