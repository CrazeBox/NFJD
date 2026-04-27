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

from fedjd.data.image_classification import make_federated_image_classification
from fedjd.data.multimnist import make_multimnist
from fedjd.experiments.nfjd_phases.metric_utils import summarize_round_history
from fedjd.experiments.nfjd_phases.phase5_utils import (
    build_trainer,
    cleanup,
    evaluate_model,
    fill_classification_metrics,
)
from fedjd.models.basic_cnn_mtl import BasicCNNMTL
from fedjd.models.cifar_resnet import CIFARResNet18MTL
from fedjd.models.femnist_cnn import FEMNISTCNN
from fedjd.problems import multi_task_classification


RESULTS_DIR = Path("results/nfjd_phase5/official_baseline_compare")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "official_baseline_compare.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

METHODS = ["fedavg_ls", "qfedavg", "fmgda", "fedmgda_plus", "fedclient_upgrad"]


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "exp_id", "dataset", "dataset_note", "data_split", "protocol", "method",
        "seed", "num_rounds", "local_epochs", "num_clients", "participation_rate",
        "learning_rate", "qfedavg_q", "qfedavg_update_scale",
        "qfedavg_mode",
        "fmgda_update_scale", "fedmgda_plus_update_scale", "fedclient_update_scale",
        "model_arch",
        "avg_accuracy", "avg_f1", "min_task_acc", "min_task_f1",
        "elapsed_time", "avg_upload_bytes", "avg_round_time", "upload_per_client",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_dataset(name: str, num_clients: int, iid: bool, seed: int,
                  max_train_samples: int | None, max_eval_samples: int | None):
    if name == "multimnist":
        data = make_multimnist(num_clients=num_clients, iid=iid, seed=seed)
        return data, 2, 10, 1, "generated_multimnist"
    if name in {"cifar10", "femnist"}:
        data = make_federated_image_classification(
            dataset=name,
            num_clients=num_clients,
            iid=iid,
            seed=seed,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
        )
        return data, 1, data["num_classes"], data["input_channels"], data["dataset_note"]
    raise ValueError(f"Unsupported dataset: {name}")


def _build_model(dataset: str, num_tasks: int, num_classes: int, input_channels: int) -> tuple[torch.nn.Module, str]:
    if dataset == "multimnist":
        return BasicCNNMTL(input_channels=1, num_tasks=2, num_classes=10), "basic_cnn_mtl"
    if dataset == "cifar10":
        return CIFARResNet18MTL(num_tasks=num_tasks, num_classes=num_classes), "cifar_resnet18_3x3"
    if dataset == "femnist":
        return FEMNISTCNN(num_tasks=num_tasks, num_classes=num_classes), "femnist_small_cnn"
    raise ValueError(f"Unsupported dataset: {dataset}")


def run_one(args, dataset: str, split: str, protocol_name: str,
            local_epochs: int, rounds: int, method: str, seed: int) -> dict:
    iid = split == "iid"
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, num_tasks, num_classes, input_channels, dataset_note = _load_dataset(
        dataset,
        num_clients=args.num_clients,
        iid=iid,
        seed=seed,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    model, model_arch = _build_model(dataset, num_tasks, num_classes, input_channels)
    exp_id = f"official-{dataset}-{split}-{protocol_name}-{method}-seed{seed}"
    trainer = build_trainer(
        method=method,
        model=model,
        client_datasets=data["client_datasets"],
        objective_fn=multi_task_classification,
        m=num_tasks,
        seed=seed,
        device=device,
        num_rounds=rounds,
        num_clients=args.num_clients,
        participation_rate=args.participation_rate,
        learning_rate=args.learning_rate,
        local_epochs=local_epochs,
        eval_dataset=data["val_dataset"],
        fedclient_update_scale=args.fedclient_update_scale,
        fmgda_update_scale=args.fmgda_update_scale,
        fedmgda_plus_update_scale=args.fedmgda_plus_update_scale,
        qfedavg_q=args.qfedavg_q,
        qfedavg_update_scale=args.qfedavg_update_scale,
        qfedavg_mode=args.qfedavg_mode,
    )
    start = time.time()
    history = trainer.fit()
    elapsed = time.time() - start
    summary = summarize_round_history(history)
    preds, targets = evaluate_model(trainer.server.model, data["test_dataset"], device)
    row = {
        "exp_id": exp_id,
        "dataset": dataset,
        "dataset_note": dataset_note,
        "data_split": split,
        "protocol": protocol_name,
        "method": method,
        "seed": seed,
        "num_rounds": rounds,
        "local_epochs": local_epochs,
        "num_clients": args.num_clients,
        "participation_rate": args.participation_rate,
        "learning_rate": args.learning_rate,
        "qfedavg_q": args.qfedavg_q if method == "qfedavg" else "",
        "qfedavg_update_scale": args.qfedavg_update_scale if method == "qfedavg" else "",
        "qfedavg_mode": args.qfedavg_mode if method == "qfedavg" else "",
        "fmgda_update_scale": args.fmgda_update_scale if method == "fmgda" else "",
        "fedmgda_plus_update_scale": args.fedmgda_plus_update_scale if method == "fedmgda_plus" else "",
        "fedclient_update_scale": args.fedclient_update_scale if method == "fedclient_upgrad" else "",
        "model_arch": model_arch,
        "elapsed_time": round(elapsed, 2),
        "avg_upload_bytes": round(summary["avg_upload_bytes"], 0),
        "avg_round_time": round(summary["avg_round_time"], 4),
        "upload_per_client": round(summary["upload_per_client"], 0),
    }
    return fill_classification_metrics(row, preds, targets, num_tasks)


def parse_args():
    parser = argparse.ArgumentParser(description="Official-aligned qFedAvg/FMOO baseline comparison.")
    parser.add_argument("--datasets", nargs="+", choices=["multimnist", "cifar10", "femnist"], default=["multimnist", "cifar10", "femnist"])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--splits", nargs="+", choices=["iid", "noniid"], default=["iid", "noniid"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--protocols", nargs="+", choices=["E5R100", "E1R500"], default=["E5R100", "E1R500"])
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--participation-rate", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--qfedavg-q", type=float, default=0.5)
    parser.add_argument("--qfedavg-update-scale", type=float, default=1.0)
    parser.add_argument("--qfedavg-mode", choices=["official_delta", "loss_weighted_delta"], default="official_delta")
    parser.add_argument("--fmgda-update-scale", type=float, default=1.0)
    parser.add_argument("--fedmgda-plus-update-scale", type=float, default=1.0)
    parser.add_argument("--fedclient-update-scale", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--output", default="official_baseline_compare.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol_map = {"E5R100": (5, 100), "E1R500": (1, 500)}
    jobs = []
    for dataset in args.datasets:
        for split in args.splits:
            for protocol_name in args.protocols:
                local_epochs, rounds = protocol_map[protocol_name]
                for method in args.methods:
                    for seed in args.seeds:
                        jobs.append((dataset, split, protocol_name, local_epochs, rounds, method, seed))

    rows = []
    logger.info("Starting official baseline comparison: %d experiments", len(jobs))
    for idx, job in enumerate(jobs, start=1):
        dataset, split, protocol_name, local_epochs, rounds, method, seed = job
        logger.info("[%d/%d] dataset=%s split=%s protocol=%s method=%s seed=%d", idx, len(jobs), dataset, split, protocol_name, method, seed)
        try:
            rows.append(run_one(args, dataset, split, protocol_name, local_epochs, rounds, method, seed))
        except Exception as exc:
            logger.exception("Experiment failed: %s", exc)
        finally:
            cleanup()

    path = RESULTS_DIR / args.output
    _write_csv(path, rows)
    logger.info("Official baseline comparison complete: %d/%d saved to %s", len(rows), len(jobs), path)


if __name__ == "__main__":
    main()
