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

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedjd.data.shakespeare import make_shakespeare  # noqa: E402
from fedjd.experiments.nfjd_phases.phase5_utils import build_trainer  # noqa: E402
from fedjd.experiments.nfjd_phases.metric_utils import summarize_round_history  # noqa: E402
from fedjd.models.char_lstm import CharLSTM  # noqa: E402
from fedjd.paths import resolve_project_path  # noqa: E402
from fedjd.problems.language import make_client_level_next_char_objective  # noqa: E402


LOGGER = logging.getLogger("shakespeare_client_objectives")
DEFAULT_METHODS = ["nfjd", "fedavg", "qfedavg", "fedmgda_plus", "fedclient_upgrad"]
SUMMARY_FIELDS = [
    "exp_id", "dataset", "dataset_note", "method", "seed", "num_clients", "num_rounds",
    "local_epochs", "participation_rate", "learning_rate", "model_arch", "vocab_size",
    "sequence_length", "mean_client_accuracy", "worst10_client_accuracy", "client_accuracy_std",
    "mean_client_loss", "worst10_client_loss", "client_loss_std", "avg_round_time",
    "avg_upload_bytes", "elapsed_time", "train_samples_total", "test_samples_total",
    "train_samples_min", "train_samples_median", "test_samples_min", "test_samples_median",
]
CLIENT_FIELDS = ["exp_id", "client_id", "client_name", "train_samples", "test_samples", "test_accuracy", "test_loss"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bottom_fraction_mean(values: list[float], fraction: float = 0.1) -> float:
    if not values:
        return math.nan
    count = max(1, int(math.ceil(len(values) * fraction)))
    return float(np.mean(sorted(values)[:count]))


def top_fraction_mean(values: list[float], fraction: float = 0.1) -> float:
    if not values:
        return math.nan
    count = max(1, int(math.ceil(len(values) * fraction)))
    return float(np.mean(sorted(values, reverse=True)[:count]))


def evaluate_client(model: torch.nn.Module, dataset: Dataset, device: torch.device, batch_size: int) -> tuple[float, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    correct = 0
    loss_sum = 0.0
    model.eval()
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            labels = by[:, 0].to(device).long()
            logits = model(bx)
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
            loss_sum += float(loss.item())
    model.train()
    if total == 0:
        return math.nan, math.nan
    return correct / total, loss_sum / total


def evaluate_clients(
    exp_id: str,
    model: torch.nn.Module,
    client_train_datasets: list[Dataset],
    client_test_datasets: list[Dataset],
    client_names: list[str],
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict], dict]:
    rows = []
    for client_id, dataset in enumerate(client_test_datasets):
        accuracy, loss = evaluate_client(model, dataset, device, batch_size)
        rows.append({
            "exp_id": exp_id,
            "client_id": client_id,
            "client_name": client_names[client_id] if client_id < len(client_names) else str(client_id),
            "train_samples": len(client_train_datasets[client_id]),
            "test_samples": len(dataset),
            "test_accuracy": accuracy,
            "test_loss": loss,
        })
    accuracies = [float(row["test_accuracy"]) for row in rows if not math.isnan(float(row["test_accuracy"]))]
    losses = [float(row["test_loss"]) for row in rows if not math.isnan(float(row["test_loss"]))]
    metrics = {
        "mean_client_accuracy": float(np.mean(accuracies)) if accuracies else math.nan,
        "worst10_client_accuracy": bottom_fraction_mean(accuracies, 0.1),
        "client_accuracy_std": float(np.std(accuracies)) if accuracies else math.nan,
        "mean_client_loss": float(np.mean(losses)) if losses else math.nan,
        "worst10_client_loss": top_fraction_mean(losses, 0.1),
        "client_loss_std": float(np.std(losses)) if losses else math.nan,
    }
    return rows, metrics


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_one(args: argparse.Namespace, method: str, output_dir: Path) -> dict:
    set_seed(args.seed)
    data = make_shakespeare(
        num_clients=args.num_clients,
        seed=args.seed,
        root=args.shakespeare_root,
        auto_prepare=not args.no_auto_prepare_shakespeare,
        min_samples_per_client=args.min_samples_per_client,
        max_samples_per_client=args.max_samples_per_client,
        test_fraction=args.client_test_fraction,
        sequence_length=args.sequence_length,
        stride=args.sequence_stride,
        source=args.shakespeare_source,
        client_selection="random" if args.random_clients else args.client_selection,
        sample_fraction=args.sample_fraction,
        vocab_scope=args.vocab_scope,
    )
    device = torch.device(args.device)
    model = CharLSTM(
        vocab_size=len(data.vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    objective_fn = make_client_level_next_char_objective(args.num_clients)
    exp_id = f"shakespeare-clientobj-{method}-K{args.num_clients}-seed{args.seed}"
    LOGGER.info("Running %s", exp_id)

    trainer = build_trainer(
        method=method,
        model=model,
        client_datasets=data.client_train_datasets,
        objective_fn=objective_fn,
        m=args.num_clients,
        seed=args.seed,
        device=device,
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        participation_rate=args.participation_rate,
        learning_rate=args.learning_rate,
        local_epochs=args.local_epochs,
        eval_dataset=None,
        fedclient_update_scale=args.fedclient_update_scale,
        fedmgda_plus_update_scale=args.fedmgda_plus_update_scale,
        qfedavg_q=args.qfedavg_q,
        qfedavg_update_scale=args.qfedavg_update_scale,
        qfedavg_mode=args.qfedavg_mode,
    )

    start = time.time()
    history = trainer.fit()
    elapsed = time.time() - start
    round_summary = summarize_round_history(history)
    client_rows, client_metrics = evaluate_clients(
        exp_id=exp_id,
        model=trainer.server.model,
        client_train_datasets=data.client_train_datasets,
        client_test_datasets=data.client_test_datasets,
        client_names=data.client_names,
        device=device,
        batch_size=args.eval_batch_size,
    )
    write_rows(output_dir / f"clients_{exp_id}.csv", client_rows, CLIENT_FIELDS)

    row = {
        "exp_id": exp_id,
        "dataset": "shakespeare",
        "dataset_note": data.dataset_note,
        "method": method,
        "seed": args.seed,
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "local_epochs": args.local_epochs,
        "participation_rate": args.participation_rate,
        "learning_rate": args.learning_rate,
        "model_arch": "char_lstm",
        "vocab_size": len(data.vocab),
        "sequence_length": data.sequence_length,
        "avg_round_time": round_summary["avg_round_time"],
        "avg_upload_bytes": round_summary["avg_upload_bytes"],
        "elapsed_time": elapsed,
        "train_samples_total": sum(data.train_sizes),
        "test_samples_total": sum(data.test_sizes),
        "train_samples_min": min(data.train_sizes),
        "train_samples_median": float(np.median(data.train_sizes)),
        "test_samples_min": min(data.test_sizes),
        "test_samples_median": float(np.median(data.test_sizes)),
    }
    row.update(client_metrics)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shakespeare client-level multi-objective federated test.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--participation-rate", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--client-test-fraction", type=float, default=0.2)
    parser.add_argument("--min-samples-per-client", type=int, default=64)
    parser.add_argument("--max-samples-per-client", type=int, default=2000, help="Use <=0 to keep all generated samples per client.")
    parser.add_argument("--sequence-length", type=int, default=80)
    parser.add_argument("--sequence-stride", type=int, default=1)
    parser.add_argument("--shakespeare-source", choices=["auto", "custom", "leaf"], default="auto")
    parser.add_argument("--client-selection", choices=["top", "random", "leaf"], default="leaf")
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--vocab-scope", choices=["all", "sampled", "selected"], default="all")
    parser.add_argument("--random-clients", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shakespeare-root", default="data/shakespeare")
    parser.add_argument("--no-auto-prepare-shakespeare", action="store_true")
    parser.add_argument("--output-dir", default="results/shakespeare_client_objectives")
    parser.add_argument("--fedclient-update-scale", type=float, default=1.0)
    parser.add_argument("--fedmgda-plus-update-scale", type=float, default=1.0)
    parser.add_argument("--qfedavg-q", type=float, default=0.5)
    parser.add_argument("--qfedavg-update-scale", type=float, default=1.0)
    parser.add_argument("--qfedavg-mode", choices=["official_delta", "loss_weighted_delta"], default="official_delta")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    args.shakespeare_root = str(resolve_project_path(args.shakespeare_root))
    output_dir = resolve_project_path(args.output_dir)
    rows = []
    for method in args.methods:
        row = run_one(args, method, output_dir)
        rows.append(row)
        write_rows(output_dir / "summary.csv", rows, SUMMARY_FIELDS)
    LOGGER.info("Wrote summary to %s", output_dir / "summary.csv")


if __name__ == "__main__":
    main()
