from __future__ import annotations

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


RESULTS_DIR = Path("results/nfjd_phase5/riverflow_fast")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "p5_riverflow_fast.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

SEEDS = [7, 42, 123]
METHODS = ["nfjd_fast", "fedavg_cagrad", "fedavg_ls"]
TASK_COUNTS = [2, 4, 8]


def run_riverflow_fast(method, seed, iid=True, num_rounds=20,
                       num_clients=10, participation_rate=0.5,
                       learning_rate=0.001, num_tasks=8):
    split_name = "iid" if iid else "noniid"
    exp_id = f"P5-rf-fast-{method}-{split_name}-m{num_tasks}-seed{seed}"
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


def main():
    rows = []
    experiments = []
    for num_tasks in TASK_COUNTS:
        for method in METHODS:
            for seed in SEEDS:
                experiments.append(dict(method=method, seed=seed, iid=True, num_tasks=num_tasks))

    logger.info("Starting RiverFlow fast benchmark: %d experiments", len(experiments))
    for idx, exp in enumerate(experiments):
        logger.info("[%d/%d] Running %s", idx + 1, len(experiments), exp)
        try:
            rows.append(run_riverflow_fast(**exp))
        except Exception as exc:
            logger.exception("Experiment failed: %s", exc)
        finally:
            cleanup()

    csv_path = RESULTS_DIR / "phase5_riverflow_fast.csv"
    write_csv(csv_path, rows)
    logger.info("RiverFlow fast benchmark complete: %d/%d saved to %s", len(rows), len(experiments), csv_path)


if __name__ == "__main__":
    main()
