from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from fedjd.data.celeba import make_celeba
from fedjd.models.celeba_cnn import CelebaCNN
from fedjd.problems import multi_objective_regression
from fedjd.experiments.nfjd_phases.phase5_utils import run_experiment, evaluate_model, write_csv, cleanup

RESULTS_DIR = Path("results/nfjd_phase5/celeba")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(RESULTS_DIR / "p5_celeba.log", mode="w"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

SEEDS = [7, 42, 123]
METHODS = ["nfjd", "fedjd", "fmgda", "weighted_sum", "direction_avg"]
CELEBA_ROOT = "/root/data/celeba/celeba"


def run_celeba(method, seed, iid=True, num_rounds=50,
               num_clients=10, participation_rate=0.5, learning_rate=0.0001,
               num_tasks=4, fair_comparison=False):
    split_name = "iid" if iid else "noniid"
    exp_id = f"P5-ca-{method}-{split_name}-m{num_tasks}-seed{seed}"
    if fair_comparison:
        exp_id = f"P5-fair-ca-{method}-{split_name}-m{num_tasks}-seed{seed}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    train_datasets, val_datasets, test_datasets = make_celeba(
        num_clients=num_clients, iid=iid, seed=seed, num_tasks=num_tasks,
        root=CELEBA_ROOT, download=False,
    )

    data = {
        "client_datasets": train_datasets,
        "test_dataset": torch.utils.data.ConcatDataset(test_datasets),
    }

    model = CelebaCNN(num_attributes=num_tasks)

    if method == "nfjd":
        le, nr = 3, num_rounds
    else:
        le = 1
        nr = num_rounds * 3 if fair_comparison else num_rounds

    row = run_experiment(
        exp_id=exp_id, method=method, model=model,
        client_datasets=data["client_datasets"],
        objective_fn=multi_objective_regression, m=num_tasks, seed=seed, device=device,
        num_rounds=nr, num_clients=num_clients,
        participation_rate=participation_rate, learning_rate=learning_rate,
        model_arch="celeba_cnn", dataset="celeba", data_split=split_name,
        local_epochs=le, fair_comparison=fair_comparison,
    )

    all_preds, all_targets = evaluate_model(model, data["test_dataset"], device)

    per_task_mse = []
    for t in range(num_tasks):
        mse = ((all_preds[:, t] - all_targets[:, t]) ** 2).mean().item()
        per_task_mse.append(mse)
    row["per_task_mse"] = ",".join(f"{v:.6f}" for v in per_task_mse)
    row["avg_mse"] = round(sum(per_task_mse) / len(per_task_mse), 6)
    row["max_mse"] = round(max(per_task_mse), 6)
    row["mse_std"] = round(float(np.std(per_task_mse)), 6)

    return row


def main():
    all_rows = []
    experiments = []

    for method in METHODS + ["stl"]:
        for seed in SEEDS:
            experiments.append(dict(method=method, seed=seed, iid=True, num_tasks=4))

    for method in METHODS:
        for seed in SEEDS:
            experiments.append(dict(method=method, seed=seed, iid=False, num_tasks=4))

    for m in [2, 4, 6]:
        for method in ["nfjd", "fedjd"]:
            for seed in SEEDS:
                experiments.append(dict(method=method, seed=seed, iid=True, num_tasks=m))

    for method in METHODS:
        for seed in SEEDS:
            experiments.append(dict(method=method, seed=seed, iid=True, num_tasks=4, fair_comparison=True))

    total = len(experiments)
    logger.info(f"Starting Phase 5 CelebA Benchmark: {total} experiments")

    for idx, exp in enumerate(experiments):
        logger.info(f"[{idx+1}/{total}] Running {exp}...")
        try:
            row = run_celeba(**exp)
            all_rows.append(row)
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cleanup()

    csv_path = RESULTS_DIR / "phase5_celeba_results.csv"
    write_csv(csv_path, all_rows)
    logger.info(f"Phase 5 CelebA complete! {len(all_rows)}/{total} experiments, saved to {csv_path}")


if __name__ == "__main__":
    main()
