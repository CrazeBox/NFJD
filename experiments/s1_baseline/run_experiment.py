from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch

from fedjd import (
    ExperimentConfig,
    FedJDClient,
    FedJDServer,
    FedJDTrainer,
    JacobianAggregator,
    MeanAggregator,
    MinNormAggregator,
    RandomAggregator,
    plot_training_curves,
)
from fedjd.data import make_synthetic_federated_regression
from fedjd.models import SmallRegressor
from fedjd.problems import two_objective_regression

AGGREGATOR_REGISTRY: dict[str, type[JacobianAggregator]] = {
    "minnorm": MinNormAggregator,
    "mean": MeanAggregator,
    "random": RandomAggregator,
}


def build_aggregator(config: ExperimentConfig) -> JacobianAggregator:
    name = config.aggregator.lower()
    if name not in AGGREGATOR_REGISTRY:
        raise ValueError(f"Unknown aggregator '{name}'. Available: {list(AGGREGATOR_REGISTRY.keys())}")
    cls = AGGREGATOR_REGISTRY[name]
    if name == "minnorm":
        return cls(max_iters=config.aggregator_max_iters, lr=config.aggregator_lr)
    if name == "random":
        return cls(seed=config.seed)
    return cls()


def setup_logging(output_dir: Path, experiment_id: str) -> logging.Logger:
    logger = logging.getLogger("fedjd.toy")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    log_path = output_dir / "stdout.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_experiment(config: ExperimentConfig) -> dict:
    output_dir = config.get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    logger = setup_logging(output_dir, config.experiment_id)

    logger.info("=" * 60)
    logger.info("FedJD Toy Example - Stage 1 Baseline Smoke Test")
    logger.info("Experiment ID: %s", config.experiment_id)
    logger.info("=" * 60)

    set_seed(config.seed)
    logger.info("Random seed set to %d", config.seed)

    config.save_yaml(output_dir / "config.yaml")
    logger.info("Config saved to %s", output_dir / "config.yaml")

    device = torch.device(config.device)
    logger.info("Device: %s", device)

    logger.info("Generating synthetic federated regression data...")
    data = make_synthetic_federated_regression(
        num_clients=config.num_clients,
        samples_per_client=config.samples_per_client,
        input_dim=config.input_dim,
        noise_std=config.noise_std,
        seed=config.seed,
    )
    logger.info(
        "Data generated: %d clients, %d samples/client, input_dim=%d",
        config.num_clients,
        config.samples_per_client,
        config.input_dim,
    )

    clients = [
        FedJDClient(client_id=i, dataset=ds, batch_size=config.batch_size, device=device)
        for i, ds in enumerate(data.client_datasets)
    ]

    model = SmallRegressor(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
    )
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: SmallRegressor | Parameters: %d", num_params)

    aggregator = build_aggregator(config)
    logger.info("Aggregator: %s", type(aggregator).__name__)

    server = FedJDServer(
        model=model,
        clients=clients,
        aggregator=aggregator,
        objective_fn=two_objective_regression,
        participation_rate=config.participation_rate,
        learning_rate=config.learning_rate,
        device=device,
    )

    initial_objectives = server.evaluate_global_objectives()
    logger.info("Initial objectives: %s", _fmt_obj(initial_objectives))

    trainer = FedJDTrainer(
        server=server,
        num_rounds=config.num_rounds,
        output_dir=str(output_dir),
        save_checkpoints=config.save_checkpoints,
        checkpoint_interval=config.checkpoint_interval,
    )

    logger.info("Starting training for %d rounds...", config.num_rounds)
    total_start = time.time()
    history = trainer.fit()
    total_time = time.time() - total_start

    final_objectives = server.evaluate_global_objectives()
    logger.info("Final objectives: %s", _fmt_obj(final_objectives))
    logger.info("Total training time: %.2f seconds", total_time)

    logger.info("Generating plots...")
    plot_training_curves(history, output_dir, config.experiment_id)

    peak_mem = _peak_memory_mb()
    summary = _build_summary(config, history, initial_objectives, final_objectives, total_time, num_params, peak_mem)
    _save_summary(output_dir, summary, logger)

    logger.info("Experiment complete. Results in %s", output_dir)

    return summary


def _fmt_obj(values: list[float]) -> str:
    return ", ".join(f"{v:.6f}" for v in values)


def _peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)


def _build_summary(
    config: ExperimentConfig,
    history: list,
    initial_objectives: list[float],
    final_objectives: list[float],
    total_time: float,
    num_params: int,
    peak_mem: float,
) -> dict:
    num_objectives = len(initial_objectives)
    total_upload = sum(s.upload_bytes for s in history)
    total_download = sum(s.download_bytes for s in history)
    total_nan_inf = sum(s.nan_inf_count for s in history)

    objective_deltas = [final_objectives[i] - initial_objectives[i] for i in range(num_objectives)]

    avg_round_time = sum(s.round_time for s in history) / max(len(history), 1)
    avg_jacobian_norm = sum(s.jacobian_norm for s in history) / max(len(history), 1)
    avg_direction_norm = sum(s.direction_norm for s in history) / max(len(history), 1)

    converged = all(d < 0 for d in objective_deltas)
    no_crash = total_nan_inf == 0

    return {
        "experiment_id": config.experiment_id,
        "seed": config.seed,
        "num_clients": config.num_clients,
        "participation_rate": config.participation_rate,
        "num_rounds": config.num_rounds,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "aggregator": config.aggregator,
        "num_params": num_params,
        "num_objectives": num_objectives,
        "initial_objectives": initial_objectives,
        "final_objectives": final_objectives,
        "objective_deltas": objective_deltas,
        "total_time": total_time,
        "avg_round_time": avg_round_time,
        "avg_jacobian_norm": avg_jacobian_norm,
        "avg_direction_norm": avg_direction_norm,
        "total_upload_bytes": total_upload,
        "total_download_bytes": total_download,
        "total_nan_inf": total_nan_inf,
        "peak_memory_mb": peak_mem,
        "converged": converged,
        "no_crash": no_crash,
        "stage1_pass": no_crash and converged,
    }


def _save_summary(output_dir: Path, summary: dict, logger: logging.Logger) -> None:
    lines = [
        "# FedJD Stage 1 Experiment Summary",
        "",
        "## Experiment Identity",
        f"- Experiment ID: `{summary['experiment_id']}`",
        f"- Seed: {summary['seed']}",
        "",
        "## Configuration",
        f"- Clients (K): {summary['num_clients']}",
        f"- Participation rate (C): {summary['participation_rate']}",
        f"- Rounds: {summary['num_rounds']}",
        f"- Learning rate: {summary['learning_rate']}",
        f"- Batch size: {summary['batch_size']}",
        f"- Aggregator: {summary['aggregator']}",
        f"- Model parameters (d): {summary['num_params']}",
        f"- Objectives (m): {summary['num_objectives']}",
        "",
        "## Objective Values",
    ]
    for i in range(summary["num_objectives"]):
        init_v = summary["initial_objectives"][i]
        final_v = summary["final_objectives"][i]
        delta = summary["objective_deltas"][i]
        arrow = "↓" if delta < 0 else "↑"
        lines.append(f"- Objective {i}: initial={init_v:.6f}, final={final_v:.6f}, delta={delta:.6f} {arrow}")

    lines.extend([
        "",
        "## System Metrics",
        f"- Total training time: {summary['total_time']:.2f}s",
        f"- Avg round time: {summary['avg_round_time']:.4f}s",
        f"- Avg Jacobian Frobenius norm: {summary['avg_jacobian_norm']:.4f}",
        f"- Avg direction norm: {summary['avg_direction_norm']:.4f}",
        f"- Total upload bytes: {summary['total_upload_bytes']}",
        f"- Total download bytes: {summary['total_download_bytes']}",
        f"- Peak memory: {summary['peak_memory_mb']:.1f} MB",
        f"- Total NaN/Inf: {summary['total_nan_inf']}",
        "",
        "## Stage 1 Assessment",
        f"- No crash (NaN/Inf = 0): {'PASS' if summary['no_crash'] else 'FAIL'}",
        f"- Objectives decreasing: {'PASS' if summary['converged'] else 'FAIL'}",
        f"- Overall Stage 1: {'PASS' if summary['stage1_pass'] else 'FAIL'}",
    ])

    path = output_dir / "summary.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Summary saved to %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedJD Stage 1 Toy Example - Single Experiment")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--num-clients", type=int, default=8, help="Number of federated clients (K)")
    parser.add_argument("--samples-per-client", type=int, default=64, help="Samples per client")
    parser.add_argument("--input-dim", type=int, default=8, help="Input dimension")
    parser.add_argument("--hidden-dim", type=int, default=16, help="Hidden dimension")
    parser.add_argument("--num-rounds", type=int, default=30, help="Number of communication rounds")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--participation-rate", type=float, default=0.5, help="Client participation rate (C)")
    parser.add_argument("--aggregator", type=str, default="minnorm", choices=["minnorm", "mean", "random"], help="Aggregation strategy")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory (auto-generated if empty)")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save model checkpoints")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ExperimentConfig(
        seed=args.seed,
        num_clients=args.num_clients,
        samples_per_client=args.samples_per_client,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_rounds=args.num_rounds,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        participation_rate=args.participation_rate,
        aggregator=args.aggregator,
        output_dir=args.output_dir,
        save_checkpoints=args.save_checkpoints,
        device=args.device,
    )

    run_single_experiment(config)


if __name__ == "__main__":
    main()
