from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from fedjd.data.multimnist import make_multimnist
from fedjd.models.lenet_mtl import LeNetMTL
from fedjd.problems import multi_task_classification
from fedjd.experiments.nfjd_phases.phase5_utils import (
    run_experiment, evaluate_model, fill_classification_metrics, write_csv, cleanup,
)

RESULTS_DIR = Path("results/nfjd_phase5/multimnist")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(RESULTS_DIR / "p5_multimnist.log", mode="w"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

SEEDS = [7, 42, 123]
METHODS = ["nfjd", "fedjd", "fmgda", "weighted_sum", "direction_avg"]


def run_multimnist(method, seed, iid=True, num_rounds=50,
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

    if method == "nfjd":
        le, nr = 3, num_rounds
    else:
        le = 1
        nr = num_rounds * 3 if fair_comparison else num_rounds

    row = run_experiment(
        exp_id=exp_id, method=method, model=model,
        client_datasets=data["client_datasets"],
        objective_fn=multi_task_classification, m=2, seed=seed, device=device,
        num_rounds=nr, num_clients=num_clients,
        participation_rate=participation_rate, learning_rate=learning_rate,
        model_arch="lenet_mtl", dataset="multimnist", data_split=split_name,
        local_epochs=le, fair_comparison=fair_comparison,
    )

    all_preds, all_targets = evaluate_model(model, data["test_dataset"], device)
    row = fill_classification_metrics(row, all_preds, all_targets, 2)

    return row


def main():
    all_rows = []
    experiments = []

    for method in METHODS + ["stl"]:
        for seed in SEEDS:
            experiments.append(dict(method=method, seed=seed, iid=True))

    for method in METHODS:
        for seed in SEEDS:
            experiments.append(dict(method=method, seed=seed, iid=False))

    for method in METHODS:
        for seed in SEEDS:
            experiments.append(dict(method=method, seed=seed, iid=True, fair_comparison=True))

    total = len(experiments)
    logger.info(f"Starting Phase 5 MultiMNIST Benchmark: {total} experiments")

    for idx, exp in enumerate(experiments):
        logger.info(f"[{idx+1}/{total}] Running {exp}...")
        try:
            row = run_multimnist(**exp)
            all_rows.append(row)
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cleanup()

    csv_path = RESULTS_DIR / "phase5_multimnist_results.csv"
    write_csv(csv_path, all_rows)
    logger.info(f"Phase 5 MultiMNIST complete! {len(all_rows)}/{total} experiments, saved to {csv_path}")


if __name__ == "__main__":
    main()
