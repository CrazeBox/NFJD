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

from fedjd.data.federated_vision import (  # noqa: E402
    VisionFederatedData,
    bottom_fraction_mean,
    make_cifar10_dirichlet,
    make_femnist_writers,
)
from fedjd.experiments.nfjd_phases.phase5_utils import build_trainer  # noqa: E402
from fedjd.models.cifar_resnet import CIFARResNet18MTL  # noqa: E402
from fedjd.models.femnist_cnn import FEMNISTCNN  # noqa: E402
from fedjd.problems.classification import multi_task_classification  # noqa: E402

LOGGER = logging.getLogger("federated_vision")
DEFAULT_METHODS = ["fedavg", "qfedavg", "fedmgda_plus", "fedclient_upgrad"]
SUMMARY_FIELDS = [
    "exp_id", "dataset", "dataset_note", "split", "method", "seed",
    "num_clients", "num_rounds", "local_epochs", "participation_rate", "learning_rate",
    "mean_client_test_accuracy", "global_iid_test_accuracy", "worst10_client_accuracy",
    "client_accuracy_std", "mean_client_test_loss", "avg_round_time", "avg_upload_bytes",
    "avg_aggregation_compute_time", "max_aggregation_compute_time", "elapsed_time",
    "pareto2d_front_clients", "pareto3d_front_clients", "model_arch",
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


def build_model(dataset: str, num_classes: int):
    if dataset == "femnist":
        return FEMNISTCNN(num_tasks=1, num_classes=num_classes), "femnist_small_cnn"
    if dataset == "cifar10":
        return CIFARResNet18MTL(num_tasks=1, num_classes=num_classes), "cifar_resnet18_3x3"
    raise ValueError(f"Unsupported dataset: {dataset}")


def logits_for_single_task(predictions: torch.Tensor) -> torch.Tensor:
    if predictions.ndim == 3:
        return predictions[:, 0]
    return predictions


def evaluate_dataset(model: torch.nn.Module, dataset: Dataset, device: torch.device, batch_size: int) -> tuple[float, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    correct = 0
    loss_sum = 0.0
    model.eval()
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            labels = by[:, 0].long() if by.ndim > 1 else by.long()
            logits = logits_for_single_task(model(bx))
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
            loss_sum += float(loss.item())
    model.train()
    if total == 0:
        return math.nan, math.nan
    return correct / total, loss_sum / total


def pareto_mask(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros(0, dtype=bool)
    mask = np.ones(points.shape[0], dtype=bool)
    for i in range(points.shape[0]):
        if not mask[i]:
            continue
        dominates_i = np.all(points >= points[i], axis=1) & np.any(points > points[i], axis=1)
        if np.any(dominates_i):
            mask[i] = False
    return mask


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


def plot_pareto(output_dir: Path, exp_id: str, client_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        LOGGER.warning("matplotlib is unavailable; skipping Pareto plots for %s", exp_id)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    acc = np.asarray([row["test_accuracy"] for row in client_rows], dtype=np.float64)
    neg_loss = np.asarray([row["neg_test_loss"] for row in client_rows], dtype=np.float64)
    size = np.asarray([row["train_size_normalized"] for row in client_rows], dtype=np.float64)
    mask2d = np.asarray([row["pareto2d"] for row in client_rows], dtype=bool)
    mask3d = np.asarray([row["pareto3d"] for row in client_rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(acc[~mask2d], neg_loss[~mask2d], s=22, alpha=0.55, label="clients")
    ax.scatter(acc[mask2d], neg_loss[mask2d], s=38, alpha=0.9, label="Pareto front")
    if mask2d.any():
        order = np.argsort(acc[mask2d])
        ax.plot(acc[mask2d][order], neg_loss[mask2d][order], linewidth=1.5)
    ax.set_xlabel("Client test accuracy")
    ax.set_ylabel("Negative client test loss")
    ax.set_title(exp_id)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"pareto2d_{exp_id}.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 5))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.scatter(acc[~mask3d], neg_loss[~mask3d], size[~mask3d], s=18, alpha=0.45)
    ax3.scatter(acc[mask3d], neg_loss[mask3d], size[mask3d], s=34, alpha=0.9)
    ax3.set_xlabel("Client acc")
    ax3.set_ylabel("-Client loss")
    ax3.set_zlabel("Norm. train size")
    ax3.set_title(exp_id)
    fig.tight_layout()
    fig.savefig(output_dir / f"pareto3d_{exp_id}.png", dpi=180)
    plt.close(fig)


def summarize_history(history) -> dict:
    if not history:
        return {
            "avg_round_time": math.nan,
            "avg_upload_bytes": math.nan,
            "avg_aggregation_compute_time": math.nan,
            "max_aggregation_compute_time": math.nan,
        }
    round_times = [float(getattr(stats, "round_time", 0.0)) for stats in history]
    upload_bytes = [float(getattr(stats, "upload_bytes", 0.0)) for stats in history]
    agg_times = []
    for stats in history:
        agg = float(getattr(stats, "aggregation_time", 0.0) or 0.0)
        direction = float(getattr(stats, "direction_time", 0.0) or 0.0)
        agg_times.append(agg + direction)
    return {
        "avg_round_time": float(np.mean(round_times)),
        "avg_upload_bytes": float(np.mean(upload_bytes)),
        "avg_aggregation_compute_time": float(np.mean(agg_times)),
        "max_aggregation_compute_time": float(np.max(agg_times)),
    }


def load_scenario(args, scenario: str) -> tuple[str, str, VisionFederatedData]:
    if scenario == "femnist":
        data = make_femnist_writers(
            num_clients=args.femnist_clients,
            seed=args.seed,
            leaf_root=args.femnist_leaf_root,
            torchvision_root=args.torchvision_root,
            test_fraction=args.client_test_fraction,
            min_samples_per_writer=args.min_samples_per_client,
            max_samples_per_writer=args.max_samples_per_writer,
            use_emnist_global_test=not args.femnist_use_client_test_union_global,
            auto_prepare=not args.no_auto_prepare_femnist,
            apply_emnist_orientation_fix=not args.no_femnist_orientation_fix,
        )
        return "femnist", "writers", data
    if scenario.startswith("cifar10_alpha"):
        alpha = float(scenario.replace("cifar10_alpha", ""))
        data = make_cifar10_dirichlet(
            num_clients=args.cifar_clients,
            alpha=alpha,
            seed=args.seed,
            root=args.torchvision_root,
            test_fraction=args.client_test_fraction,
            max_train_samples=args.max_cifar_train_samples,
            min_samples_per_client=args.min_samples_per_client,
        )
        return "cifar10", f"dirichlet_alpha_{alpha}", data
    raise ValueError(f"Unknown scenario: {scenario}")


def run_one(args, scenario: str, method: str, output_dir: Path) -> dict:
    set_seed(args.seed)
    dataset_name, split_name, data = load_scenario(args, scenario)
    model, model_arch = build_model(dataset_name, data.num_classes)
    model = model.to(args.device)
    exp_id = f"fv-{dataset_name}-{split_name}-{method}-seed{args.seed}"
    LOGGER.info("Running %s", exp_id)

    trainer = build_trainer(
        method=method,
        model=model,
        client_datasets=data.client_train_datasets,
        objective_fn=multi_task_classification,
        m=1,
        seed=args.seed,
        device=torch.device(args.device),
        num_rounds=args.num_rounds,
        num_clients=len(data.client_train_datasets),
        participation_rate=args.participation_rate,
        learning_rate=args.learning_rate,
        local_epochs=args.local_epochs,
        eval_dataset=data.global_test_dataset,
        fedclient_update_scale=args.fedclient_update_scale,
        qfedavg_q=args.qfedavg_q,
        qfedavg_update_scale=args.qfedavg_update_scale,
        qfedavg_mode=args.qfedavg_mode,
        fedmgda_plus_update_scale=args.fedmgda_plus_update_scale,
    )

    start = time.time()
    history = trainer.fit()
    elapsed = time.time() - start
    history_metrics = summarize_history(history)

    client_rows, client_metrics = evaluate_clients(
        exp_id=exp_id,
        model=trainer.server.model,
        client_train=data.client_train_datasets,
        client_test=data.client_test_datasets,
        device=torch.device(args.device),
        batch_size=args.eval_batch_size,
    )
    global_acc, _global_loss = evaluate_dataset(
        trainer.server.model,
        data.global_test_dataset,
        torch.device(args.device),
        args.eval_batch_size,
    )

    save_client_rows(output_dir / f"clients_{exp_id}.csv", client_rows)
    plot_pareto(output_dir, exp_id, client_rows)

    row = {
        "exp_id": exp_id,
        "dataset": dataset_name,
        "dataset_note": data.dataset_note,
        "split": split_name,
        "method": method,
        "seed": args.seed,
        "num_clients": len(data.client_train_datasets),
        "num_rounds": args.num_rounds,
        "local_epochs": args.local_epochs,
        "participation_rate": args.participation_rate,
        "learning_rate": args.learning_rate,
        "global_iid_test_accuracy": global_acc,
        "elapsed_time": elapsed,
        "model_arch": model_arch,
    }
    row.update(client_metrics)
    row.update(history_metrics)
    return row


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Current FEMNIST+CIFAR10 federated vision benchmark.")
    parser.add_argument("--scenarios", nargs="+", default=["cifar10_alpha0.1", "cifar10_alpha0.5", "femnist"])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--participation-rate", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--client-test-fraction", type=float, default=0.2)
    parser.add_argument("--min-samples-per-client", type=int, default=20)
    parser.add_argument("--femnist-clients", type=int, default=50)
    parser.add_argument("--cifar-clients", type=int, default=50)
    parser.add_argument("--max-cifar-train-samples", type=int, default=None)
    parser.add_argument("--max-samples-per-writer", type=int, default=None)
    parser.add_argument("--torchvision-root", default="data/torchvision")
    parser.add_argument("--femnist-leaf-root", default="data/femnist")
    parser.add_argument("--no-auto-prepare-femnist", action="store_true")
    parser.add_argument("--no-femnist-orientation-fix", action="store_true")
    parser.add_argument("--femnist-use-client-test-union-global", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="results/federated_vision")
    parser.add_argument("--fedclient-update-scale", type=float, default=1.0)
    parser.add_argument("--fedmgda-plus-update-scale", type=float, default=1.0)
    parser.add_argument("--qfedavg-q", type=float, default=0.5)
    parser.add_argument("--qfedavg-update-scale", type=float, default=1.0)
    parser.add_argument("--qfedavg-mode", choices=["official_delta", "loss_weighted_delta"], default="official_delta")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = []
    for scenario in args.scenarios:
        for method in args.methods:
            rows.append(run_one(args, scenario, method, output_dir))
            write_summary(output_dir / "summary.csv", rows)
    LOGGER.info("Wrote summary to %s", output_dir / "summary.csv")


if __name__ == "__main__":
    main()
