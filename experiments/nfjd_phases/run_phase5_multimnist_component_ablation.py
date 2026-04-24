from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from fedjd.data.multimnist import make_multimnist
from fedjd.experiments.nfjd_phases.phase5_utils import (
    cleanup,
    evaluate_model,
    fill_classification_metrics,
    run_experiment,
    write_csv,
)
from fedjd.models.lenet_mtl import LeNetMTL
from fedjd.problems import multi_task_classification


RESULTS_DIR = Path("results/nfjd_phase5/multimnist_component_ablation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "p5_multimnist_component_ablation.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

DEFAULT_SEEDS = [7, 42, 123]
DEFAULT_METHODS = [
    "nfjd",
    "nfjd_cached",
    "nfjd_momentum",
    "nfjd_rescale",
    "nfjd_softweight",
    "nfjd_hybrid",
    "nfjd_fast",
    "fedavg_cagrad",
]


def run_multimnist_component(method, seed, iid=True, num_rounds=50,
                             num_clients=10, participation_rate=0.5,
                             learning_rate=0.001):
    split_name = "iid" if iid else "noniid"
    exp_id = f"P5-mm-comp-{method}-{split_name}-seed{seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    data = make_multimnist(num_clients=num_clients, iid=iid, seed=seed)
    model = LeNetMTL(input_channels=1, num_tasks=2, num_classes=10)

    row = run_experiment(
        exp_id=exp_id,
        method=method,
        model=model,
        client_datasets=data["client_datasets"],
        objective_fn=multi_task_classification,
        m=2,
        seed=seed,
        device=device,
        num_rounds=num_rounds,
        num_clients=num_clients,
        participation_rate=participation_rate,
        learning_rate=learning_rate,
        model_arch="lenet_mtl",
        dataset="multimnist",
        data_split=split_name,
        local_epochs=3,
        eval_dataset=data["val_dataset"],
    )
    preds, targets = evaluate_model(model, data["test_dataset"], device)
    return fill_classification_metrics(row, preds, targets, 2)


def parse_args():
    parser = argparse.ArgumentParser(description="Run MultiMNIST NFJD component ablations.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--splits", nargs="+", choices=["iid", "noniid"], default=["noniid", "iid"])
    parser.add_argument("--rounds", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    experiments = []
    for split in args.splits:
        iid = split == "iid"
        for method in args.methods:
            for seed in args.seeds:
                experiments.append(dict(method=method, seed=seed, iid=iid, num_rounds=args.rounds))

    logger.info("Starting MultiMNIST component ablation: %d experiments", len(experiments))
    for idx, exp in enumerate(experiments):
        logger.info("[%d/%d] Running %s", idx + 1, len(experiments), exp)
        try:
            rows.append(run_multimnist_component(**exp))
        except Exception as exc:
            logger.exception("Experiment failed: %s", exc)
        finally:
            cleanup()

    csv_path = RESULTS_DIR / "phase5_multimnist_component_ablation.csv"
    write_csv(csv_path, rows)
    logger.info("MultiMNIST component ablation complete: %d/%d saved to %s", len(rows), len(experiments), csv_path)


if __name__ == "__main__":
    main()
