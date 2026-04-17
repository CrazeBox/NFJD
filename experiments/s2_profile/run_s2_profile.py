from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch

from fedjd import FedJDClient, FedJDServer, FedJDTrainer, MinNormAggregator
from fedjd.data import make_synthetic_federated_regression
from fedjd.models import MODEL_REGISTRY
from fedjd.problems import multi_objective_regression

logger = logging.getLogger("fedjd.s2_profile")

MODEL_SIZES = ["small", "medium", "large"]
OBJECTIVE_COUNTS = [2, 3, 5, 10]
PARTICIPATION_RATES = [0.1, 0.25, 0.5, 1.0]
SAMPLED_CLIENT_COUNTS = [4, 8, 16, 32]
SEEDS = [7, 42, 2024]
INPUT_DIM = 16


def _make_exp_id(model_size: str, num_objectives: int, num_clients: int, participation_rate: float, seed: int) -> str:
    c_str = f"C{participation_rate}".replace(".", "p")
    return f"S2-synth_m{num_objectives}-{model_size}-m{num_objectives}-K{num_clients}-{c_str}-E1-fulljac-seed{seed}"


def _get_model_hidden_dim(model_size: str) -> int:
    return {"small": 32, "medium": 128, "large": 512}[model_size]


def run_single_profile(
    model_size: str,
    num_objectives: int,
    num_clients: int,
    participation_rate: float,
    seed: int,
    num_rounds: int,
    device: torch.device,
    output_dir: str,
) -> dict:
    exp_id = _make_exp_id(model_size, num_objectives, num_clients, participation_rate, seed)
    exp_dir = Path(output_dir) / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)

    hidden_dim = _get_model_hidden_dim(model_size)
    model_cls = MODEL_REGISTRY[model_size]
    model = model_cls(input_dim=INPUT_DIM, hidden_dim=hidden_dim, output_dim=num_objectives)
    num_params = sum(p.numel() for p in model.parameters())

    data = make_synthetic_federated_regression(
        num_clients=num_clients,
        samples_per_client=64,
        input_dim=INPUT_DIM,
        num_objectives=num_objectives,
        noise_std=0.1,
        seed=seed,
    )

    clients = [
        FedJDClient(client_id=i, dataset=ds, batch_size=32, device=device)
        for i, ds in enumerate(data.client_datasets)
    ]

    server = FedJDServer(
        model=model,
        clients=clients,
        aggregator=MinNormAggregator(max_iters=250, lr=0.2),
        objective_fn=multi_objective_regression,
        participation_rate=participation_rate,
        learning_rate=0.05,
        device=device,
    )

    trainer = FedJDTrainer(
        server=server,
        num_rounds=num_rounds,
        output_dir=str(exp_dir),
    )

    start = time.time()
    history = trainer.fit()
    total_time = time.time() - start

    if not history:
        return {"experiment_id": exp_id, "error": "No history", "stage2_pass": False}

    avg_upload_per_client = history[-1].jacobian_upload_per_client
    avg_gradient_per_client = history[-1].gradient_upload_per_client
    jac_grad_ratio = history[-1].jacobian_vs_gradient_ratio

    avg_client_compute = sum(s.client_compute_time for s in history) / len(history)
    avg_direction_time = sum(s.direction_time for s in history) / len(history)
    avg_round_time = sum(s.round_time for s in history) / len(history)
    avg_upload_bytes = sum(s.upload_bytes for s in history) / len(history)
    avg_download_bytes = sum(s.download_bytes for s in history) / len(history)
    avg_client_mem = sum(s.client_peak_memory_mb for s in history) / len(history)
    avg_server_mem = sum(s.server_peak_memory_mb for s in history) / len(history)
    total_nan_inf = sum(s.nan_inf_count for s in history)

    initial_obj = history[0].objective_values
    final_obj = history[-1].objective_values
    obj_deltas = [final_obj[i] - initial_obj[i] for i in range(len(initial_obj))]

    return {
        "experiment_id": exp_id,
        "seed": seed,
        "model_size": model_size,
        "num_objectives": num_objectives,
        "num_clients": num_clients,
        "participation_rate": participation_rate,
        "num_params": num_params,
        "num_rounds": num_rounds,
        "total_time": total_time,
        "avg_round_time": avg_round_time,
        "avg_client_compute_time": avg_client_compute,
        "avg_direction_time": avg_direction_time,
        "avg_upload_bytes": avg_upload_bytes,
        "avg_download_bytes": avg_download_bytes,
        "avg_upload_per_client": avg_upload_per_client,
        "avg_gradient_per_client": avg_gradient_per_client,
        "jacobian_vs_gradient_ratio": jac_grad_ratio,
        "avg_client_peak_memory_mb": avg_client_mem,
        "avg_server_peak_memory_mb": avg_server_mem,
        "total_nan_inf": total_nan_inf,
        "initial_objectives": initial_obj,
        "final_objectives": final_obj,
        "objective_deltas": obj_deltas,
        "all_decrease": all(d < 0 for d in obj_deltas),
        "no_crash": total_nan_inf == 0,
        "stage2_pass": total_nan_inf == 0 and all(d < 0 for d in obj_deltas),
    }


def run_param_scale_sweep(output_dir: str, num_rounds: int, device: str, seeds: list[int]) -> list[dict]:
    logger.info("=" * 60)
    logger.info("Stage 2.1: Parameter Scale Sweep")
    logger.info("=" * 60)

    results = []
    total = len(MODEL_SIZES) * len(OBJECTIVE_COUNTS) * len(SAMPLED_CLIENT_COUNTS) * len(seeds)
    idx = 0
    for model_size in MODEL_SIZES:
        for m in OBJECTIVE_COUNTS:
            for k in SAMPLED_CLIENT_COUNTS:
                for seed in seeds:
                    idx += 1
                    c = min(8 / k, 1.0)
                    logger.info("[%d/%d] %s m=%d K=%d seed=%d", idx, total, model_size, m, k, seed)
                    try:
                        result = run_single_profile(
                            model_size, m, k, c, seed, num_rounds,
                            torch.device(device), output_dir,
                        )
                        results.append(result)
                    except Exception as exc:
                        logger.error("FAILED: %s", exc)
                        results.append({
                            "experiment_id": _make_exp_id(model_size, m, k, c, seed),
                            "error": str(exc), "stage2_pass": False,
                        })
    return results


def run_participation_sweep(output_dir: str, num_rounds: int, device: str, seeds: list[int]) -> list[dict]:
    logger.info("=" * 60)
    logger.info("Stage 2.2: Participation Rate Sweep")
    logger.info("=" * 60)

    results = []
    total = len(PARTICIPATION_RATES) * len(seeds)
    idx = 0
    for c in PARTICIPATION_RATES:
        for seed in seeds:
            idx += 1
            logger.info("[%d/%d] C=%.2f seed=%d", idx, total, c, seed)
            try:
                result = run_single_profile(
                    "medium", 5, 16, c, seed, num_rounds,
                    torch.device(device), output_dir,
                )
                results.append(result)
            except Exception as exc:
                logger.error("FAILED: %s", exc)
                results.append({
                    "experiment_id": _make_exp_id("medium", 5, 16, c, seed),
                    "error": str(exc), "stage2_pass": False,
                })
    return results


def save_results(results: list[dict], output_dir: str) -> None:
    out_path = Path(output_dir)

    csv_path = out_path / "s2_profile_results.csv"
    fieldnames = [
        "experiment_id", "seed", "model_size", "num_objectives", "num_clients",
        "participation_rate", "num_params", "num_rounds", "total_time",
        "avg_round_time", "avg_client_compute_time", "avg_direction_time",
        "avg_upload_bytes", "avg_download_bytes",
        "avg_upload_per_client", "avg_gradient_per_client",
        "jacobian_vs_gradient_ratio",
        "avg_client_peak_memory_mb", "avg_server_peak_memory_mb",
        "total_nan_inf", "all_decrease", "no_crash", "stage2_pass",
    ]
    for i in range(10):
        fieldnames.append(f"delta_obj_{i}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            if "objective_deltas" in row and row["objective_deltas"]:
                for i, d in enumerate(row["objective_deltas"]):
                    row[f"delta_obj_{i}"] = d
            writer.writerow(row)

    json_path = out_path / "s2_profile_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Results saved to %s and %s", csv_path, json_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedJD Stage 2 - Communication & Time Profiling")
    parser.add_argument("--output-dir", type=str, default="results/s2_profile", help="Output directory")
    parser.add_argument("--num-rounds", type=int, default=10, help="Rounds per experiment")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 2024], help="Random seeds")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--skip-param-sweep", action="store_true", help="Skip parameter scale sweep")
    parser.add_argument("--skip-participation-sweep", action="store_true", help="Skip participation rate sweep")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = parse_args()

    all_results = []

    if not args.skip_param_sweep:
        results = run_param_scale_sweep(args.output_dir, args.num_rounds, args.device, args.seeds)
        all_results.extend(results)

    if not args.skip_participation_sweep:
        results = run_participation_sweep(args.output_dir, args.num_rounds, args.device, args.seeds)
        all_results.extend(results)

    save_results(all_results, args.output_dir)

    passed = sum(1 for r in all_results if r.get("stage2_pass"))
    logger.info("Stage 2 complete: %d/%d experiments passed", passed, len(all_results))


if __name__ == "__main__":
    main()
