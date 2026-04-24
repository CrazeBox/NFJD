from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from fedjd.data.river_flow import make_river_flow
from fedjd.experiments.nfjd_phases.phase5_utils import (
    cleanup,
    evaluate_model,
    fill_regression_metrics,
    run_experiment,
    write_csv,
)
from fedjd.models.river_flow_mlp import RiverFlowMLP
from fedjd.problems import multi_objective_regression


RESULTS_DIR = Path("results/nfjd_phase5/riverflow_component_ablation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "p5_riverflow_component_ablation.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

DEFAULT_SEEDS = [7, 42, 123]
DEFAULT_METHODS = [
    "nfjd_cached",
    "nfjd_softweight",
    "nfjd_hybrid",
    "nfjd_fast",
    "fedavg_cagrad",
    "fedavg_ls",
]


def run_riverflow_component(method, seed, iid=True, num_rounds=50,
                            num_clients=10, participation_rate=0.5,
                            learning_rate=0.001, num_tasks=8):
    split_name = "iid" if iid else "noniid"
    exp_id = f"P5-rf-comp-{method}-{split_name}-m{num_tasks}-seed{seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    data = make_river_flow(num_clients=num_clients, iid=iid, seed=seed, num_tasks=num_tasks)
    model = RiverFlowMLP(input_dim=data["input_dim"], num_tasks=num_tasks)

    row = run_experiment(
        exp_id=exp_id,
        method=method,
        model=model,
        client_datasets=data["client_datasets"],
        objective_fn=multi_objective_regression,
        m=num_tasks,
        seed=seed,
        device=device,
        num_rounds=num_rounds,
        num_clients=num_clients,
        participation_rate=participation_rate,
        learning_rate=learning_rate,
        model_arch="river_flow_mlp",
        dataset="riverflow",
        data_split=split_name,
        local_epochs=3,
        eval_dataset=data["val_dataset"],
    )
    preds, targets = evaluate_model(model, data["test_dataset"], device)
    return fill_regression_metrics(row, preds, targets, num_tasks)


def parse_args():
    parser = argparse.ArgumentParser(description="Run RiverFlow NFJD component ablations.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--tasks", nargs="+", type=int, default=[8])
    parser.add_argument("--splits", nargs="+", choices=["iid", "noniid"], default=["iid"])
    parser.add_argument("--rounds", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    experiments = []
    for num_tasks in args.tasks:
        for split in args.splits:
            iid = split == "iid"
            for method in args.methods:
                for seed in args.seeds:
                    experiments.append(dict(
                        method=method,
                        seed=seed,
                        iid=iid,
                        num_rounds=args.rounds,
                        num_tasks=num_tasks,
                    ))

    logger.info("Starting RiverFlow component ablation: %d experiments", len(experiments))
    for idx, exp in enumerate(experiments):
        logger.info("[%d/%d] Running %s", idx + 1, len(experiments), exp)
        try:
            rows.append(run_riverflow_component(**exp))
        except Exception as exc:
            logger.exception("Experiment failed: %s", exc)
        finally:
            cleanup()

    csv_path = RESULTS_DIR / "phase5_riverflow_component_ablation.csv"
    write_csv(csv_path, rows)
    logger.info("RiverFlow component ablation complete: %d/%d saved to %s", len(rows), len(experiments), csv_path)


if __name__ == "__main__":
    main()
