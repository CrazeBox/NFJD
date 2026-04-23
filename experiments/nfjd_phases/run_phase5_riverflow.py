from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from fedjd.data.river_flow import make_river_flow
from fedjd.models.river_flow_mlp import RiverFlowMLP
from fedjd.problems import multi_objective_regression
from fedjd.experiments.nfjd_phases.phase5_utils import (
    run_experiment, evaluate_model, fill_regression_metrics, write_csv, cleanup,
)

RESULTS_DIR = Path("results/nfjd_phase5/riverflow")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(RESULTS_DIR / "p5_riverflow.log", mode="w"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

SEEDS = [7, 42, 123]
METHODS = ["nfjd", "fedavg_ls", "fedavg_mgda", "fedavg_pcgrad", "fedavg_cagrad"]


def run_riverflow(method, seed, iid=True, num_rounds=50,
                  num_clients=10, participation_rate=0.5, learning_rate=0.001,
                  num_tasks=8, fair_comparison=False):
    if fair_comparison:
        raise ValueError("fair_comparison is deprecated: the default setup already uses matched local epochs and rounds across methods.")

    split_name = "iid" if iid else "noniid"
    exp_id = f"P5-rf-{method}-{split_name}-m{num_tasks}-seed{seed}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    data = make_river_flow(num_clients=num_clients, iid=iid, seed=seed, num_tasks=num_tasks)
    model = RiverFlowMLP(input_dim=data["input_dim"], num_tasks=num_tasks)

    le, nr = 3, num_rounds

    row = run_experiment(
        exp_id=exp_id, method=method, model=model,
        client_datasets=data["client_datasets"],
        objective_fn=multi_objective_regression, m=num_tasks, seed=seed, device=device,
        num_rounds=nr, num_clients=num_clients,
        participation_rate=participation_rate, learning_rate=learning_rate,
        model_arch="river_flow_mlp", dataset="riverflow", data_split=split_name,
        local_epochs=le, fair_comparison=fair_comparison, eval_dataset=data["val_dataset"],
    )

    all_preds, all_targets = evaluate_model(model, data["test_dataset"], device)
    row = fill_regression_metrics(row, all_preds, all_targets, num_tasks)

    return row


def main():
    all_rows = []
    experiments = []

    for method in METHODS:
        for seed in SEEDS:
            experiments.append(dict(method=method, seed=seed, iid=True))

    for method in METHODS:
        for seed in SEEDS:
            experiments.append(dict(method=method, seed=seed, iid=False))

    total = len(experiments)
    logger.info(f"Starting Phase 5 RiverFlow Benchmark: {total} experiments")

    for idx, exp in enumerate(experiments):
        logger.info(f"[{idx+1}/{total}] Running {exp}...")
        try:
            row = run_riverflow(**exp)
            all_rows.append(row)
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cleanup()

    csv_path = RESULTS_DIR / "phase5_riverflow_results.csv"
    write_csv(csv_path, all_rows)
    logger.info(f"Phase 5 RiverFlow complete! {len(all_rows)}/{total} experiments, saved to {csv_path}")


if __name__ == "__main__":
    main()
