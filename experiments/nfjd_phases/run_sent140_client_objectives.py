from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedjd.data.federated_vision import bottom_fraction_mean  # noqa: E402
from fedjd.data.sent140 import make_sent140_clients  # noqa: E402
from fedjd.experiments.federated_vision.run_femnist_cifar10 import (  # noqa: E402
    pareto_mask,
    plot_method_pareto,
    plot_sorted_client_accuracy,
    summarize_history,
)
from fedjd.experiments.nfjd_phases.phase5_utils import build_trainer  # noqa: E402
from fedjd.models.text_classifier import MeanPooledTextClassifier  # noqa: E402
from fedjd.paths import resolve_project_path  # noqa: E402
from fedjd.problems.classification import multi_task_binary_classification  # noqa: E402


LOGGER = logging.getLogger("sent140_client_objectives")
DEFAULT_METHODS = ["fedavg", "qfedavg", "fedmgda_plus", "fedclient_upgrad"]
SUMMARY_FIELDS = [
    "exp_id", "dataset", "dataset_note", "split", "method", "seed",
    "num_clients", "num_rounds", "local_epochs", "participation_rate", "learning_rate",
    "mean_client_test_accuracy", "worst10_client_accuracy",
    "client_accuracy_std", "mean_client_test_loss", "avg_round_time", "avg_upload_bytes",
    "avg_aggregation_compute_time", "max_aggregation_compute_time", "elapsed_time",
    "pareto2d_front_clients", "pareto3d_front_clients", "model_arch", "vocab_size",
]
CLIENT_FIELDS = [
    "exp_id", "client_id", "train_samples", "test_samples", "test_accuracy", "test_loss",
    "neg_test_loss", "train_size_normalized", "pareto2d", "pareto3d",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_dataset(model: torch.nn.Module, dataset: Dataset, device: torch.device, batch_size: int) -> tuple[float, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    correct = 0
    loss_sum = 0.0
    model.eval()
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            targets = by.to(device).float()
            logits = model(bx)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
            pred = (torch.sigmoid(logits) >= 0.5).to(targets.dtype)
            correct += int((pred == targets).sum().item())
            total += int(targets.numel())
            loss_sum += float(loss.item())
    model.train()
    if total == 0:
        return math.nan, math.nan
    return correct / total, loss_sum / total


def evaluate_clients(
    exp_id: str,
    model: torch.nn.Module,
    client_train: list[Dataset],
    client_test: list[Dataset],
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict], dict]:
    rows = []
    train_sizes = np.asarray([len(ds) for ds in client_train], dtype=np.float64)
    max_train = max(float(train_sizes.max()), 1.0) if len(train_sizes) else 1.0
    for client_id, dataset in enumerate(client_test):
        accuracy, loss = evaluate_dataset(model, dataset, device, batch_size)
        rows.append({
            "exp_id": exp_id,
            "client_id": client_id,
            "train_samples": len(client_train[client_id]),
            "test_samples": len(dataset),
            "test_accuracy": accuracy,
            "test_loss": loss,
            "neg_test_loss": -loss,
            "train_size_normalized": len(client_train[client_id]) / max_train,
        })

    points2d = np.asarray([[row["test_accuracy"], row["neg_test_loss"]] for row in rows], dtype=np.float64)
    points3d = np.asarray(
        [[row["test_accuracy"], row["neg_test_loss"], row["train_size_normalized"]] for row in rows],
        dtype=np.float64,
    )
    mask2d = pareto_mask(points2d)
    mask3d = pareto_mask(points3d)
    for row, p2, p3 in zip(rows, mask2d, mask3d):
        row["pareto2d"] = bool(p2)
        row["pareto3d"] = bool(p3)

    accuracies = [float(row["test_accuracy"]) for row in rows]
    losses = [float(row["test_loss"]) for row in rows]
    metrics = {
        "mean_client_test_accuracy": float(np.mean(accuracies)) if accuracies else math.nan,
        "worst10_client_accuracy": bottom_fraction_mean(accuracies, 0.1),
        "client_accuracy_std": float(np.std(accuracies)) if accuracies else math.nan,
        "mean_client_test_loss": float(np.mean(losses)) if losses else math.nan,
        "pareto2d_front_clients": int(mask2d.sum()),
        "pareto3d_front_clients": int(mask3d.sum()),
    }
    return rows, metrics


def save_client_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLIENT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_one(args: argparse.Namespace, method: str, output_dir: Path) -> dict:
    set_seed(args.seed)
    data = make_sent140_clients(
        num_clients=args.num_clients,
        seed=args.seed,
        leaf_root=args.sent140_root,
        test_fraction=args.client_test_fraction,
        min_samples_per_client=args.min_samples_per_client,
        max_samples_per_client=args.max_samples_per_client,
        vocab_size=args.vocab_size,
        min_token_freq=args.min_token_freq,
        sequence_length=args.sequence_length,
        auto_prepare=not args.no_auto_prepare_sent140,
    )
    device = torch.device(args.device)
    model = MeanPooledTextClassifier(
        vocab_size=len(data.vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    exp_id = f"sent140-user-{method}-seed{args.seed}"
    LOGGER.info("Running %s", exp_id)

    trainer = build_trainer(
        method=method,
        model=model,
        client_datasets=data.client_train_datasets,
        objective_fn=multi_task_binary_classification,
        m=data.num_tasks,
        seed=args.seed,
        device=device,
        num_rounds=args.num_rounds,
        num_clients=len(data.client_train_datasets),
        participation_rate=args.participation_rate,
        learning_rate=args.learning_rate,
        local_epochs=args.local_epochs,
        eval_dataset=data.global_test_dataset,
        fedclient_update_scale=args.fedclient_update_scale,
        fedclient_normalize_updates=args.fedclient_normalize_updates,
        qfedavg_q=args.qfedavg_q,
        qfedavg_update_scale=args.qfedavg_update_scale,
        qfedavg_mode=args.qfedavg_mode,
        fedmgda_plus_update_scale=args.fedmgda_plus_update_scale,
    )
    if hasattr(trainer.server, "evaluate_each_round"):
        trainer.server.evaluate_each_round = args.eval_interval > 0

    start = time.time()
    history = trainer.fit()
    elapsed = time.time() - start
    history_metrics = summarize_history(history)
    client_rows, client_metrics = evaluate_clients(
        exp_id=exp_id,
        model=trainer.server.model,
        client_train=data.client_train_datasets,
        client_test=data.client_test_datasets,
        device=device,
        batch_size=args.eval_batch_size,
    )
    save_client_rows(output_dir / f"clients_{exp_id}.csv", client_rows)
    row = {
        "exp_id": exp_id,
        "dataset": "sent140",
        "dataset_note": data.dataset_note,
        "split": "users",
        "method": method,
        "seed": args.seed,
        "num_clients": len(data.client_train_datasets),
        "num_rounds": args.num_rounds,
        "local_epochs": args.local_epochs,
        "participation_rate": args.participation_rate,
        "learning_rate": args.learning_rate,
        "elapsed_time": elapsed,
        "model_arch": "mean_pooled_text_classifier",
        "vocab_size": len(data.vocab),
    }
    row.update(client_metrics)
    row.update(history_metrics)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sent140 client-level federated benchmark.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--participation-rate", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--client-test-fraction", type=float, default=0.2)
    parser.add_argument("--min-samples-per-client", type=int, default=20)
    parser.add_argument("--max-samples-per-client", type=int, default=None)
    parser.add_argument("--sent140-root", default="data/sent140")
    parser.add_argument("--no-auto-prepare-sent140", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--min-token-freq", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="results/sent140_client_objectives")
    parser.add_argument("--fedclient-update-scale", type=float, default=1.0)
    parser.add_argument("--fedclient-normalize-updates", action="store_true")
    parser.add_argument("--fedmgda-plus-update-scale", type=float, default=1.0)
    parser.add_argument("--qfedavg-q", type=float, default=0.5)
    parser.add_argument("--qfedavg-update-scale", type=float, default=1.0)
    parser.add_argument("--qfedavg-mode", choices=["official_delta", "loss_weighted_delta"], default="official_delta")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    args.sent140_root = str(resolve_project_path(args.sent140_root))
    output_dir = resolve_project_path(args.output_dir)
    rows = []
    for method in args.methods:
        rows.append(run_one(args, method, output_dir))
        write_summary(output_dir / "summary.csv", rows)
        plot_method_pareto(output_dir, rows)
        plot_sorted_client_accuracy(output_dir, rows)
    LOGGER.info("Wrote summary to %s", output_dir / "summary.csv")


if __name__ == "__main__":
    main()
