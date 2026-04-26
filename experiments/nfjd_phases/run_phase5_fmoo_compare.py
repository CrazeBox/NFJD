from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from fedjd.data.multimnist import make_multimnist
from fedjd.data.river_flow import make_river_flow
from fedjd.experiments.nfjd_phases.metric_utils import summarize_round_history
from fedjd.experiments.nfjd_phases.phase5_utils import (
    build_trainer,
    cleanup,
    evaluate_model,
    fill_classification_metrics,
    fill_regression_metrics,
)
from fedjd.models.basic_cnn_mtl import BasicCNNMTL
from fedjd.models.river_flow_mlp import RiverFlowMLP
from fedjd.problems import multi_objective_regression, multi_task_classification


RESULTS_DIR = Path("results/nfjd_phase5/fmoo_compare")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "p5_fmoo_compare.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

DEFAULT_METHODS = ["fmgda", "fedmgda_plus", "qfedavg", "fedclient_upgrad"]


def _write_csv(path: Path, rows: list[dict]) -> None:
    base_fields = [
        "exp_id", "method", "dataset", "data_split", "m", "seed", "num_rounds",
        "num_clients", "participation_rate", "learning_rate", "local_epochs",
        "fedclient_update_scale", "fedclient_normalize_updates",
        "fmgda_update_scale", "fedmgda_plus_update_scale",
        "qfedavg_q", "qfedavg_update_scale",
    ]

    datasets = {row.get("dataset") for row in rows}
    if datasets == {"riverflow"}:
        metric_fields = ["avg_mse", "max_mse", "mse_std", "avg_r2"]
    elif datasets == {"multimnist"}:
        metric_fields = ["avg_accuracy", "avg_f1", "min_task_acc", "min_task_f1"]
    else:
        metric_fields = [
            "avg_mse", "max_mse", "mse_std", "avg_r2",
            "avg_accuracy", "avg_f1", "min_task_acc", "min_task_f1",
        ]

    fieldnames = base_fields + metric_fields + [
        "elapsed_time", "avg_upload_bytes", "avg_round_time", "upload_per_client",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_one(
    dataset: str,
    method: str,
    seed: int,
    iid: bool,
    num_rounds: int,
    num_clients: int,
    participation_rate: float,
    learning_rate: float,
    local_epochs: int,
    riverflow_tasks: int,
    fedclient_update_scale: float,
    fedclient_normalize_updates: bool,
    fmgda_update_scale: float,
    fedmgda_plus_update_scale: float,
    qfedavg_q: float,
    qfedavg_update_scale: float,
) -> dict:
    split_name = "iid" if iid else "noniid"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    if dataset == "riverflow":
        data = make_river_flow(num_clients=num_clients, iid=iid, seed=seed, num_tasks=riverflow_tasks)
        model = RiverFlowMLP(input_dim=data["input_dim"], num_tasks=riverflow_tasks)
        objective_fn = multi_objective_regression
        metric_kind = "regression"
        m = riverflow_tasks
        model_arch = "river_flow_mlp"
    elif dataset == "multimnist":
        data = make_multimnist(num_clients=num_clients, iid=iid, seed=seed)
        model = BasicCNNMTL(input_channels=1, num_tasks=2, num_classes=10)
        objective_fn = multi_task_classification
        metric_kind = "classification"
        m = 2
        model_arch = "basic_cnn_mtl"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    exp_id = (
        f"P5-fmoo-{dataset}-{method}-{split_name}-m{m}-le{local_epochs}"
        f"-seed{seed}"
    )

    trainer = build_trainer(
        method=method,
        model=model,
        client_datasets=data["client_datasets"],
        objective_fn=objective_fn,
        m=m,
        seed=seed,
        device=device,
        num_rounds=num_rounds,
        num_clients=num_clients,
        participation_rate=participation_rate,
        learning_rate=learning_rate,
        local_epochs=local_epochs,
        eval_dataset=data["val_dataset"],
        fedclient_update_scale=fedclient_update_scale,
        fedclient_normalize_updates=fedclient_normalize_updates,
        fmgda_update_scale=fmgda_update_scale,
        fedmgda_plus_update_scale=fedmgda_plus_update_scale,
        qfedavg_q=qfedavg_q,
        qfedavg_update_scale=qfedavg_update_scale,
    )

    start = time.time()
    history = trainer.fit()
    elapsed = time.time() - start
    round_summary = summarize_round_history(history)
    preds, targets = evaluate_model(trainer.server.model, data["test_dataset"], device)

    row = {
        "exp_id": exp_id,
        "method": method,
        "dataset": dataset,
        "data_split": split_name,
        "m": m,
        "seed": seed,
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "participation_rate": participation_rate,
        "learning_rate": learning_rate,
        "local_epochs": local_epochs,
        "fedclient_update_scale": fedclient_update_scale if method == "fedclient_upgrad" else "",
        "fedclient_normalize_updates": fedclient_normalize_updates if method == "fedclient_upgrad" else "",
        "fmgda_update_scale": fmgda_update_scale if method == "fmgda" else "",
        "fedmgda_plus_update_scale": fedmgda_plus_update_scale if method == "fedmgda_plus" else "",
        "qfedavg_q": qfedavg_q if method == "qfedavg" else "",
        "qfedavg_update_scale": qfedavg_update_scale if method == "qfedavg" else "",
        "avg_mse": "",
        "max_mse": "",
        "mse_std": "",
        "avg_r2": "",
        "avg_accuracy": "",
        "avg_f1": "",
        "min_task_acc": "",
        "min_task_f1": "",
        "elapsed_time": round(elapsed, 2),
        "avg_upload_bytes": round(round_summary["avg_upload_bytes"], 0),
        "avg_round_time": round(round_summary["avg_round_time"], 4),
        "upload_per_client": round(round_summary["upload_per_client"], 0),
        "model_arch": model_arch,
    }
    if metric_kind == "regression":
        row = fill_regression_metrics(row, preds, targets, m)
    else:
        row = fill_classification_metrics(row, preds, targets, m)
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Compare FMOO/client-objective baselines with shared training parameters.")
    parser.add_argument("--datasets", nargs="+", choices=["riverflow", "multimnist"], default=["riverflow"])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--splits", nargs="+", choices=["iid", "noniid"], default=["noniid"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--participation-rate", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--riverflow-tasks", type=int, default=4)
    parser.add_argument("--fedclient-update-scale", type=float, default=1.0)
    parser.add_argument("--fedclient-normalize-updates", type=int, choices=[0, 1], default=0)
    parser.add_argument("--fmgda-update-scale", type=float, default=1.0)
    parser.add_argument("--fedmgda-plus-update-scale", type=float, default=1.0)
    parser.add_argument("--qfedavg-q", type=float, default=0.5)
    parser.add_argument("--qfedavg-update-scale", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="phase5_fmoo_compare.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    experiments = []
    for dataset in args.datasets:
        for split in args.splits:
            for method in args.methods:
                for seed in args.seeds:
                    experiments.append((dataset, split == "iid", method, seed))

    logger.info("Starting FMOO comparison: %d experiments", len(experiments))
    for idx, (dataset, iid, method, seed) in enumerate(experiments):
        logger.info("[%d/%d] dataset=%s split=%s method=%s seed=%d", idx + 1, len(experiments), dataset, "iid" if iid else "noniid", method, seed)
        try:
            rows.append(run_one(
                dataset=dataset,
                method=method,
                seed=seed,
                iid=iid,
                num_rounds=args.rounds,
                num_clients=args.num_clients,
                participation_rate=args.participation_rate,
                learning_rate=args.learning_rate,
                local_epochs=args.local_epochs,
                riverflow_tasks=args.riverflow_tasks,
                fedclient_update_scale=args.fedclient_update_scale,
                fedclient_normalize_updates=bool(args.fedclient_normalize_updates),
                fmgda_update_scale=args.fmgda_update_scale,
                fedmgda_plus_update_scale=args.fedmgda_plus_update_scale,
                qfedavg_q=args.qfedavg_q,
                qfedavg_update_scale=args.qfedavg_update_scale,
            ))
        except Exception as exc:
            logger.exception("Experiment failed: %s", exc)
        finally:
            cleanup()

    csv_path = RESULTS_DIR / args.output
    _write_csv(csv_path, rows)
    logger.info("FMOO comparison complete: %d/%d saved to %s", len(rows), len(experiments), csv_path)


if __name__ == "__main__":
    main()
