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
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedjd.data.federated_vision import (  # noqa: E402
    VisionFederatedData,
    bottom_fraction_mean,
    make_cifar10_dirichlet,
    make_cifar10_fedmgda_paper_shards,
    make_femnist_writers,
)
from fedjd.data.celeba import make_celeba  # noqa: E402
from fedjd.experiments.nfjd_phases.phase5_utils import build_trainer  # noqa: E402
from fedjd.models.basic_cnn_mtl import BasicCNNMTL  # noqa: E402
from fedjd.models.celeba_cnn import CelebaCNN  # noqa: E402
from fedjd.models.cifar10_cnn import FedMGDAPlusCIFAR10CNN  # noqa: E402
from fedjd.models.femnist_cnn import FEMNISTCNN, FedMGDAPlusFEMNISTCNN  # noqa: E402
from fedjd.paths import resolve_project_path  # noqa: E402
from fedjd.problems.classification import multi_task_binary_classification, multi_task_classification  # noqa: E402

LOGGER = logging.getLogger("federated_vision")
DEFAULT_METHODS = ["fedavg", "qfedavg", "fedmgda_plus", "fedclient_upgrad"]
SUMMARY_FIELDS = [
    "exp_id", "dataset", "dataset_note", "split", "method", "seed",
    "num_clients", "num_rounds", "local_epochs", "participation_rate", "learning_rate",
    "mean_client_test_accuracy", "worst5_client_accuracy", "worst10_client_accuracy", "best5_client_accuracy",
    "client_accuracy_std", "mean_client_test_loss", "avg_round_time", "avg_upload_bytes",
    "avg_aggregation_compute_time", "max_aggregation_compute_time", "elapsed_time",
    "pareto2d_front_clients", "pareto3d_front_clients", "model_arch",
]
CLIENT_FIELDS = [
    "exp_id", "client_id", "train_samples", "test_samples", "test_accuracy", "test_loss",
    "neg_test_loss", "train_size_normalized", "pareto2d", "pareto3d",
]
CURVE_FIELDS = [
    "exp_id", "round", "mean_client_test_accuracy",
    "worst10_client_accuracy", "client_accuracy_std", "mean_client_test_loss",
    "objective_loss", "avg_round_time_so_far",
    "avg_upload_bytes_so_far", "avg_aggregation_compute_time_so_far",
    "max_aggregation_compute_time_so_far",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(dataset: str, num_classes: int, args: argparse.Namespace):
    if dataset == "femnist":
        if args.femnist_model == "paper_fedmgda_plus":
            return FedMGDAPlusFEMNISTCNN(num_tasks=1, num_classes=num_classes), "fedmgda_plus_table4_femnist_cnn"
        return FEMNISTCNN(num_tasks=1, num_classes=num_classes), "femnist_small_cnn"
    if dataset == "cifar10":
        if args.cifar_model == "paper_fedmgda_plus":
            return FedMGDAPlusCIFAR10CNN(num_tasks=1, num_classes=num_classes), "fedmgda_plus_table2_cifar10_cnn"
        return BasicCNNMTL(input_channels=3, num_tasks=1, num_classes=num_classes), "cifar_basic_cnn"
    if dataset == "celeba":
        return CelebaCNN(num_attributes=num_classes), "celeba_cnn"
    raise ValueError(f"Unsupported dataset: {dataset}")


def objective_for_data(data: VisionFederatedData):
    if data.task_type == "binary_multitask":
        return multi_task_binary_classification
    return multi_task_classification


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
            outputs = model(bx)
            if outputs.ndim == 2 and by.ndim > 1 and outputs.shape == by.shape:
                targets = by.float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets, reduction="sum")
                pred = (torch.sigmoid(outputs) >= 0.5).to(targets.dtype)
                correct += int((pred == targets).sum().item())
                total += int(targets.numel())
            else:
                labels = by[:, 0].long() if by.ndim > 1 else by.long()
                logits = logits_for_single_task(outputs)
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
        "worst5_client_accuracy": bottom_fraction_mean(accuracies, 0.05),
        "worst10_client_accuracy": bottom_fraction_mean(accuracies, 0.1),
        "best5_client_accuracy": bottom_fraction_mean([-value for value in accuracies], 0.05) * -1.0,
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


def save_curve_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CURVE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def make_curve_row(
    exp_id: str,
    round_idx: int,
    model: torch.nn.Module,
    data: VisionFederatedData,
    device: torch.device,
    batch_size: int,
    history,
) -> dict:
    _client_rows, client_metrics = evaluate_clients(
        exp_id=exp_id,
        model=model,
        client_train=data.client_train_datasets,
        client_test=data.client_test_datasets,
        device=device,
        batch_size=batch_size,
    )
    history_metrics = summarize_history(history)
    objective_loss = math.nan
    if history:
        values = getattr(history[-1], "objective_values", None)
        if values:
            objective_loss = float(np.mean([float(value) for value in values]))
    return {
        "exp_id": exp_id,
        "round": round_idx,
        "mean_client_test_accuracy": client_metrics["mean_client_test_accuracy"],
        "worst10_client_accuracy": client_metrics["worst10_client_accuracy"],
        "client_accuracy_std": client_metrics["client_accuracy_std"],
        "mean_client_test_loss": client_metrics["mean_client_test_loss"],
        "objective_loss": objective_loss,
        "avg_round_time_so_far": history_metrics["avg_round_time"],
        "avg_upload_bytes_so_far": history_metrics["avg_upload_bytes"],
        "avg_aggregation_compute_time_so_far": history_metrics["avg_aggregation_compute_time"],
        "max_aggregation_compute_time_so_far": history_metrics["max_aggregation_compute_time"],
    }


def fit_with_curve_tracking(
    trainer,
    exp_id: str,
    data: VisionFederatedData,
    device: torch.device,
    batch_size: int,
    eval_interval: int,
    output_dir: Path,
) -> tuple[list, list[dict]]:
    history = []
    curve_rows: list[dict] = []

    initial_objectives = trainer.server.evaluate_global_objectives()
    trainer.initial_objectives = initial_objectives
    if hasattr(trainer.server, "set_initial_objectives"):
        trainer.server.set_initial_objectives(initial_objectives)
    LOGGER.info("Initial objectives: %s", ", ".join(f"{value:.4f}" for value in initial_objectives))

    if eval_interval > 0:
        curve_rows.append(make_curve_row(exp_id, 0, trainer.server.model, data, device, batch_size, history))
        save_curve_rows(output_dir / f"curves_{exp_id}.csv", curve_rows)

    for round_idx in range(trainer.num_rounds):
        stats = trainer.server.run_round(round_idx)
        history.append(stats)
        LOGGER.info(
            "Round %03d | sampled=%s | time=%.3fs | upload=%d B",
            round_idx,
            stats.sampled_client_ids,
            stats.round_time,
            stats.upload_bytes,
        )
        completed_round = round_idx + 1
        if eval_interval > 0 and (completed_round % eval_interval == 0 or completed_round == trainer.num_rounds):
            curve_rows.append(make_curve_row(exp_id, completed_round, trainer.server.model, data, device, batch_size, history))
            save_curve_rows(output_dir / f"curves_{exp_id}.csv", curve_rows)
            LOGGER.info("Curve eval round %d written for %s", completed_round, exp_id)
    return history, curve_rows


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


def plot_curves(output_dir: Path, exp_id: str, curve_rows: list[dict]) -> None:
    if not curve_rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        LOGGER.warning("matplotlib is unavailable; skipping curve plots for %s", exp_id)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    rounds = [int(row["round"]) for row in curve_rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rounds, [float(row["mean_client_test_accuracy"]) for row in curve_rows], label="mean client acc")
    ax.plot(rounds, [float(row["worst10_client_accuracy"]) for row in curve_rows], label="worst 10% acc")
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy curves: {exp_id}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"curve_accuracy_{exp_id}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rounds, [float(row["mean_client_test_loss"]) for row in curve_rows], label="mean client loss")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"Loss curves: {exp_id}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"curve_loss_{exp_id}.png", dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(rounds, [float(row["worst10_client_accuracy"]) for row in curve_rows], label="worst 10% acc", color="tab:blue")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Worst 10% accuracy", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(rounds, [float(row["client_accuracy_std"]) for row in curve_rows], label="client acc std", color="tab:red")
    ax2.set_ylabel("Client accuracy std", color="tab:red")
    fig.suptitle(f"Fairness curves: {exp_id}")
    fig.tight_layout()
    fig.savefig(output_dir / f"curve_fairness_{exp_id}.png", dpi=180)
    plt.close(fig)


def _safe_plot_name(value: object) -> str:
    text = str(value)
    safe = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        elif char == ".":
            safe.append("p")
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "plot"


def _row_float(row: dict, key: str) -> float:
    value = row.get(key, math.nan)
    if value in {None, ""}:
        return math.nan
    return float(value)


def plot_method_pareto(output_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        LOGGER.warning("matplotlib is unavailable; skipping method-level Pareto plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row.get("dataset"), row.get("split"), row.get("seed"), row.get("num_rounds"), row.get("local_epochs"), row.get("participation_rate"))
        grouped.setdefault(key, []).append(row)

    for (dataset, split, seed, rounds, local_epochs, participation), group_rows in grouped.items():
        valid_rows = [
            row for row in group_rows
            if not math.isnan(_row_float(row, "mean_client_test_accuracy"))
            and not math.isnan(_row_float(row, "worst10_client_accuracy"))
        ]
        if not valid_rows:
            continue

        means = np.asarray([_row_float(row, "mean_client_test_accuracy") for row in valid_rows], dtype=np.float64)
        worst10 = np.asarray([_row_float(row, "worst10_client_accuracy") for row in valid_rows], dtype=np.float64)
        round_times = np.asarray([_row_float(row, "avg_round_time") for row in valid_rows], dtype=np.float64)
        methods = [str(row.get("method", "method")) for row in valid_rows]
        points = np.stack([means, worst10], axis=1)
        mask = pareto_mask(points)
        group_name = _safe_plot_name(f"{dataset}_{split}_seed{seed}_R{rounds}_E{local_epochs}_pr{participation}")

        fig, ax = plt.subplots(figsize=(7, 5))
        if np.isfinite(round_times).any():
            scatter = ax.scatter(means, worst10, c=round_times, s=95, cmap="viridis_r", alpha=0.9, edgecolors="black", linewidths=0.8)
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label("Avg round time (s)")
        else:
            ax.scatter(means, worst10, s=95, alpha=0.9, edgecolors="black", linewidths=0.8)
        for method, x, y in zip(methods, means, worst10):
            ax.annotate(method, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
        ax.set_xlabel("Mean client test accuracy")
        ax.set_ylabel("Worst-10% client accuracy")
        ax.set_title(f"Mean vs worst-10%: {dataset} {split}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"method_mean_vs_worst10_{group_name}.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(means[~mask], worst10[~mask], s=80, alpha=0.45, label="dominated methods")
        ax.scatter(means[mask], worst10[mask], s=110, alpha=0.95, label="Pareto methods", edgecolors="black", linewidths=0.8)
        if mask.any():
            order = np.argsort(means[mask])
            pareto_x = means[mask][order]
            pareto_y = worst10[mask][order]
            ax.plot(pareto_x, pareto_y, linewidth=1.6, linestyle="--", color="tab:red", label="Pareto front")
        for method, x, y in zip(methods, means, worst10):
            ax.annotate(method, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
        ax.set_xlabel("Mean client test accuracy")
        ax.set_ylabel("Worst-10% client accuracy")
        ax.set_title(f"Method-level Pareto front: {dataset} {split}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"method_pareto_mean_worst10_{group_name}.png", dpi=180)
        plt.close(fig)

        finite_time = np.isfinite(round_times)
        if finite_time.any():
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
            axes[0].scatter(round_times[finite_time], means[finite_time], s=95, alpha=0.9, edgecolors="black", linewidths=0.8)
            axes[1].scatter(round_times[finite_time], worst10[finite_time], s=95, alpha=0.9, edgecolors="black", linewidths=0.8, color="tab:orange")
            for method, t, mean, worst in zip(methods, round_times, means, worst10):
                if not math.isfinite(t):
                    continue
                axes[0].annotate(method, (t, mean), xytext=(5, 5), textcoords="offset points", fontsize=9)
                axes[1].annotate(method, (t, worst), xytext=(5, 5), textcoords="offset points", fontsize=9)
            axes[0].set_xlabel("Avg round time (s)")
            axes[0].set_ylabel("Mean client test accuracy")
            axes[0].set_title("Efficiency vs mean accuracy")
            axes[1].set_xlabel("Avg round time (s)")
            axes[1].set_ylabel("Worst-10% client accuracy")
            axes[1].set_title("Efficiency vs tail accuracy")
            for ax in axes:
                ax.grid(True, alpha=0.25)
            fig.suptitle(f"Time/accuracy trade-off: {dataset} {split}")
            fig.tight_layout()
            fig.savefig(output_dir / f"method_efficiency_tradeoff_{group_name}.png", dpi=180)
            plt.close(fig)


def plot_sorted_client_accuracy(output_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        LOGGER.warning("matplotlib is unavailable; skipping sorted client accuracy plots")
        return

    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row.get("dataset"), row.get("split"), row.get("seed"), row.get("num_rounds"), row.get("local_epochs"), row.get("participation_rate"))
        grouped.setdefault(key, []).append(row)

    for (dataset, split, seed, rounds, local_epochs, participation), group_rows in grouped.items():
        series = []
        for row in group_rows:
            exp_id = row.get("exp_id")
            method = str(row.get("method", "method"))
            client_path = output_dir / f"clients_{exp_id}.csv"
            if not exp_id or not client_path.exists():
                continue
            with open(client_path, newline="", encoding="utf-8") as f:
                client_rows = list(csv.DictReader(f))
            accuracies = sorted(
                float(client_row["test_accuracy"])
                for client_row in client_rows
                if client_row.get("test_accuracy") not in {None, ""}
            )
            if accuracies:
                series.append((method, np.asarray(accuracies, dtype=np.float64)))
        if not series:
            continue

        group_name = _safe_plot_name(f"{dataset}_{split}_seed{seed}_R{rounds}_E{local_epochs}_pr{participation}")
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, accuracies in series:
            ranks = np.linspace(0.0, 100.0, num=len(accuracies), endpoint=True)
            ax.plot(ranks, accuracies, linewidth=1.8, label=method)
        ax.set_xlabel("Client percentile sorted by accuracy")
        ax.set_ylabel("Client test accuracy")
        ax.set_title(f"Sorted client accuracy: {dataset} {split}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"sorted_client_accuracy_{group_name}.png", dpi=180)
        plt.close(fig)


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
            use_leaf_train_test_split=args.femnist_use_leaf_train_test_split,
            leaf_preprocess_kind=args.femnist_leaf_preprocess_kind,
        )
        return "femnist", "writers", data
    if scenario == "celeba":
        train_datasets, val_datasets, test_datasets = make_celeba(
            num_clients=args.celeba_clients,
            root=args.celeba_root,
            download=not args.no_auto_prepare_celeba,
            iid=not args.celeba_noniid,
            num_tasks=args.celeba_tasks,
            seed=args.seed,
        )
        data = VisionFederatedData(
            client_train_datasets=train_datasets,
            client_test_datasets=test_datasets,
            global_test_dataset=ConcatDataset(val_datasets),
            num_classes=args.celeba_tasks,
            input_channels=3,
            dataset_note="torchvision_celeba_attr_noniid" if args.celeba_noniid else "torchvision_celeba_attr_iid",
            num_tasks=args.celeba_tasks,
            task_type="binary_multitask",
        )
        return "celeba", "noniid" if args.celeba_noniid else "iid", data
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
    if scenario == "cifar10_fedmgda_paper":
        data = make_cifar10_fedmgda_paper_shards(
            seed=args.seed,
            root=args.torchvision_root,
            num_clients=args.cifar_clients,
        )
        return "cifar10", "fedmgda_plus_500shards_5peruser", data
    raise ValueError(f"Unknown scenario: {scenario}")


def run_loaded_scenario(
    args,
    dataset_name: str,
    split_name: str,
    data: VisionFederatedData,
    method: str,
    output_dir: Path,
) -> dict:
    set_seed(args.seed)
    model, model_arch = build_model(dataset_name, data.num_classes, args)
    device = torch.device(args.device)
    model = model.to(device)
    exp_id = f"fv-{dataset_name}-{split_name}-{method}-seed{args.seed}"
    LOGGER.info("Running %s", exp_id)

    trainer = build_trainer(
        method=method,
        model=model,
        client_datasets=data.client_train_datasets,
        objective_fn=objective_for_data(data),
        m=data.num_tasks,
        seed=args.seed,
        device=device,
        num_rounds=args.num_rounds,
        num_clients=len(data.client_train_datasets),
        participation_rate=args.participation_rate,
        learning_rate=args.learning_rate,
        local_epochs=args.local_epochs,
        local_batch_size=args.local_batch_size,
        eval_dataset=data.global_test_dataset,
        fedclient_update_scale=args.fedclient_update_scale,
        fedclient_upgrad_solver=args.fedclient_upgrad_solver,
        fedclient_upgrad_max_iters=args.fedclient_upgrad_max_iters,
        fedclient_upgrad_lr=args.fedclient_upgrad_lr,
        qfedavg_q=args.qfedavg_q,
        qfedavg_update_scale=args.qfedavg_update_scale,
        qfedavg_lipschitz=args.qfedavg_lipschitz,
        qfedavg_mode=args.qfedavg_mode,
        fedmgda_plus_update_scale=args.fedmgda_plus_update_scale,
        fedmgda_plus_update_decay=args.fedmgda_plus_update_decay,
        fedmgda_plus_normalize_updates=args.fedmgda_plus_normalize_updates,
    )
    if hasattr(trainer.server, "evaluate_each_round"):
        trainer.server.evaluate_each_round = args.eval_interval > 0

    start = time.time()
    if args.eval_interval > 0:
        trainer.server.evaluate_each_round = False
        history, curve_rows = fit_with_curve_tracking(
            trainer=trainer,
            exp_id=exp_id,
            data=data,
            device=device,
            batch_size=args.eval_batch_size,
            eval_interval=args.eval_interval,
            output_dir=output_dir,
        )
    else:
        history = trainer.fit()
        curve_rows = []
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
    plot_curves(output_dir, exp_id, curve_rows)

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
        "elapsed_time": elapsed,
        "model_arch": model_arch,
    }
    row.update(client_metrics)
    row.update(history_metrics)
    return row


def run_one(args, scenario: str, method: str, output_dir: Path) -> dict:
    set_seed(args.seed)
    dataset_name, split_name, data = load_scenario(args, scenario)
    return run_loaded_scenario(args, dataset_name, split_name, data, method, output_dir)


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
    parser.add_argument("--num-rounds", type=int, default=1000)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--local-batch-size", type=int, default=256, help="Use <=0 for full local batch.")
    parser.add_argument("--participation-rate", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--client-test-fraction", type=float, default=0.2)
    parser.add_argument("--min-samples-per-client", type=int, default=20)
    parser.add_argument("--femnist-clients", type=int, default=50)
    parser.add_argument("--femnist-paper-clients-per-round", type=int, default=10)
    parser.add_argument("--cifar-clients", type=int, default=50)
    parser.add_argument("--cifar-paper-clients-per-round", type=int, default=10)
    parser.add_argument("--cifar-paper-batch", choices=["small", "full"], default="full")
    parser.add_argument("--celeba-clients", type=int, default=50)
    parser.add_argument("--celeba-tasks", type=int, default=4)
    parser.add_argument("--max-cifar-train-samples", type=int, default=None)
    parser.add_argument("--max-samples-per-writer", type=int, default=None)
    parser.add_argument("--torchvision-root", default="data/torchvision")
    parser.add_argument("--femnist-leaf-root", default="data/femnist")
    parser.add_argument("--celeba-root", default="data/celeba")
    parser.add_argument("--no-auto-prepare-femnist", action="store_true")
    parser.add_argument("--no-auto-prepare-celeba", action="store_true")
    parser.add_argument("--celeba-noniid", action="store_true")
    parser.add_argument("--no-femnist-orientation-fix", action="store_true")
    parser.add_argument("--femnist-use-client-test-union-global", action="store_true")
    parser.add_argument("--femnist-use-leaf-train-test-split", action="store_true")
    parser.add_argument("--femnist-leaf-preprocess-kind", choices=["sample", "full"], default="sample")
    parser.add_argument("--femnist-model", choices=["small_cnn", "paper_fedmgda_plus"], default="small_cnn")
    parser.add_argument("--cifar-model", choices=["basic_cnn", "paper_fedmgda_plus"], default="basic_cnn")
    parser.add_argument("--fedmgda-paper-femnist-preset", action="store_true")
    parser.add_argument("--fedmgda-paper-cifar10-preset", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="results/federated_vision")
    parser.add_argument("--fedclient-update-scale", type=float, default=1.0)
    parser.add_argument("--fedclient-upgrad-solver", choices=["auto", "active_set", "pgd", "batched_pgd"], default="batched_pgd")
    parser.add_argument("--fedclient-upgrad-max-iters", type=int, default=250)
    parser.add_argument("--fedclient-upgrad-lr", type=float, default=0.1)
    parser.add_argument("--fedmgda-plus-update-scale", type=float, default=1.0)
    parser.add_argument("--fedmgda-plus-update-decay", type=float, default=None)
    parser.add_argument("--fedmgda-plus-normalize-updates", action="store_true")
    parser.add_argument("--qfedavg-q", type=float, default=0.5)
    parser.add_argument("--qfedavg-update-scale", type=float, default=None)
    parser.add_argument("--qfedavg-lipschitz", type=float, default=None)
    parser.add_argument("--qfedavg-mode", choices=["official_delta", "loss_weighted_delta"], default="official_delta")
    return parser.parse_args()


def apply_qfedavg_scale_default(args: argparse.Namespace) -> None:
    if args.qfedavg_update_scale is None:
        args.qfedavg_update_scale = args.learning_rate if args.qfedavg_mode == "official_delta" else 1.0


def apply_fedmgda_paper_femnist_preset(args: argparse.Namespace) -> None:
    if not args.fedmgda_paper_femnist_preset:
        return
    args.scenarios = ["femnist"]
    args.femnist_model = "paper_fedmgda_plus"
    args.femnist_use_leaf_train_test_split = True
    args.femnist_use_client_test_union_global = True
    args.femnist_leaf_preprocess_kind = "full"
    args.min_samples_per_client = 1
    if args.femnist_clients == 50:
        args.femnist_clients = 3406
    args.num_rounds = 1500
    args.local_epochs = 1
    args.local_batch_size = 0
    args.learning_rate = 0.1
    args.participation_rate = args.femnist_paper_clients_per_round / max(args.femnist_clients, 1)
    args.qfedavg_q = 0.1
    args.qfedavg_lipschitz = 0.1
    args.qfedavg_update_scale = args.learning_rate
    args.qfedavg_mode = "official_delta"
    args.fedmgda_plus_update_scale = 2.0
    args.fedmgda_plus_update_decay = 0.2
    args.fedmgda_plus_normalize_updates = True


def apply_fedmgda_paper_cifar10_preset(args: argparse.Namespace) -> None:
    if not args.fedmgda_paper_cifar10_preset:
        return
    args.scenarios = ["cifar10_fedmgda_paper"]
    args.cifar_model = "paper_fedmgda_plus"
    args.cifar_clients = 100
    args.max_cifar_train_samples = None
    args.min_samples_per_client = 1
    args.local_epochs = 1
    args.participation_rate = args.cifar_paper_clients_per_round / max(args.cifar_clients, 1)
    args.qfedavg_mode = "official_delta"
    args.fedmgda_plus_normalize_updates = True
    if args.cifar_paper_batch == "small":
        args.num_rounds = 2000
        args.local_batch_size = 10
        args.learning_rate = 0.01
        args.qfedavg_q = 0.5
        args.qfedavg_lipschitz = 1.0
        args.qfedavg_update_scale = args.learning_rate
        args.fedmgda_plus_update_scale = 1.5
        args.fedmgda_plus_update_decay = 0.1
    else:
        args.num_rounds = 3000
        args.local_batch_size = 400
        args.learning_rate = 0.1
        args.qfedavg_q = 0.1
        args.qfedavg_lipschitz = 0.1
        args.qfedavg_update_scale = args.learning_rate
        args.fedmgda_plus_update_scale = 1.0
        args.fedmgda_plus_update_decay = 0.025


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    apply_fedmgda_paper_femnist_preset(args)
    apply_fedmgda_paper_cifar10_preset(args)
    apply_qfedavg_scale_default(args)
    args.torchvision_root = str(resolve_project_path(args.torchvision_root))
    args.femnist_leaf_root = str(resolve_project_path(args.femnist_leaf_root))
    args.celeba_root = str(resolve_project_path(args.celeba_root))
    output_dir = resolve_project_path(args.output_dir)
    rows = []
    for scenario in args.scenarios:
        set_seed(args.seed)
        LOGGER.info("Loading scenario %s", scenario)
        dataset_name, split_name, data = load_scenario(args, scenario)
        LOGGER.info(
            "Loaded %s/%s: clients=%d train_samples=%d test_samples=%d",
            dataset_name,
            split_name,
            len(data.client_train_datasets),
            sum(len(dataset) for dataset in data.client_train_datasets),
            sum(len(dataset) for dataset in data.client_test_datasets),
        )
        for method in args.methods:
            rows.append(run_loaded_scenario(args, dataset_name, split_name, data, method, output_dir))
            write_summary(output_dir / "summary.csv", rows)
            plot_method_pareto(output_dir, rows)
            plot_sorted_client_accuracy(output_dir, rows)
    LOGGER.info("Wrote summary to %s", output_dir / "summary.csv")


if __name__ == "__main__":
    main()
