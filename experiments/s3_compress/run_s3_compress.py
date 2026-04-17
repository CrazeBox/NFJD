from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path

import torch

from fedjd import FedJDClient, FedJDServer, FedJDTrainer, MinNormAggregator
from fedjd.compressors import COMPRESSOR_REGISTRY, make_compressor
from fedjd.data import make_synthetic_federated_regression
from fedjd.models import MODEL_REGISTRY
from fedjd.problems import multi_objective_regression

logger = logging.getLogger("fedjd.s3_compress")

COMPRESSOR_NAMES = list(COMPRESSOR_REGISTRY.keys())
SYNC_INTERVALS = [1, 2, 5]
LOCAL_STEPS_LIST = [1, 2, 3]
SEEDS = [7, 42, 2024]


def _make_exp_id(compressor: str, model_size: str, num_objectives: int,
                 full_sync_interval: int, local_steps: int, seed: int) -> str:
    sync_str = f"sync{full_sync_interval}" if full_sync_interval > 1 else "sync1"
    step_str = f"ls{local_steps}" if local_steps > 1 else "ls1"
    return f"S3-{compressor}-{model_size}-m{num_objectives}-{sync_str}-{step_str}-seed{seed}"


def _get_model_hidden_dim(model_size: str) -> int:
    return {"small": 32, "medium": 128, "large": 512}[model_size]


def run_single(
    compressor_name: str,
    model_size: str,
    num_objectives: int,
    num_clients: int,
    participation_rate: float,
    full_sync_interval: int,
    local_steps: int,
    seed: int,
    num_rounds: int,
    device: torch.device,
    output_dir: str,
) -> dict:
    exp_id = _make_exp_id(compressor_name, model_size, num_objectives, full_sync_interval, local_steps, seed)
    exp_dir = Path(output_dir) / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)

    hidden_dim = _get_model_hidden_dim(model_size)
    model_cls = MODEL_REGISTRY[model_size]
    model = model_cls(input_dim=16, hidden_dim=hidden_dim, output_dim=num_objectives)
    num_params = sum(p.numel() for p in model.parameters())

    data = make_synthetic_federated_regression(
        num_clients=num_clients, samples_per_client=64,
        input_dim=16, num_objectives=num_objectives, noise_std=0.1, seed=seed,
    )

    clients = [
        FedJDClient(client_id=i, dataset=ds, batch_size=32, device=device)
        for i, ds in enumerate(data.client_datasets)
    ]

    compressor = make_compressor(compressor_name)

    server = FedJDServer(
        model=model, clients=clients, aggregator=MinNormAggregator(max_iters=250, lr=0.2),
        objective_fn=multi_objective_regression,
        participation_rate=participation_rate, learning_rate=0.05, device=device,
        compressor=compressor, full_sync_interval=full_sync_interval, local_steps=local_steps,
    )

    trainer = FedJDTrainer(server=server, num_rounds=num_rounds, output_dir=str(exp_dir))

    start = time.time()
    try:
        history = trainer.fit()
    except Exception as exc:
        logger.error("FAILED %s: %s", exp_id, exc)
        return {"experiment_id": exp_id, "error": str(exc), "stage3_pass": False}
    total_time = time.time() - start

    if not history:
        return {"experiment_id": exp_id, "error": "No history", "stage3_pass": False}

    initial_obj = history[0].objective_values
    final_obj = history[-1].objective_values
    obj_deltas = [final_obj[i] - initial_obj[i] for i in range(len(initial_obj))]

    full_sync_rounds = [s for s in history if s.is_full_sync_round]
    total_upload = sum(s.upload_bytes for s in history)
    total_nan_inf = sum(s.nan_inf_count for s in history)

    avg_compression_ratio = 0.0
    if full_sync_rounds:
        avg_compression_ratio = sum(s.compression_ratio for s in full_sync_rounds) / len(full_sync_rounds)

    baseline_upload = num_objectives * num_params * 4 * len(full_sync_rounds) * int(num_clients * participation_rate)

    return {
        "experiment_id": exp_id,
        "seed": seed,
        "compressor": compressor_name,
        "model_size": model_size,
        "num_objectives": num_objectives,
        "num_clients": num_clients,
        "participation_rate": participation_rate,
        "full_sync_interval": full_sync_interval,
        "local_steps": local_steps,
        "num_params": num_params,
        "num_rounds": num_rounds,
        "total_time": total_time,
        "total_upload_bytes": total_upload,
        "baseline_upload_bytes": baseline_upload,
        "upload_saving_ratio": 1.0 - (total_upload / max(baseline_upload, 1)),
        "avg_compression_ratio": avg_compression_ratio,
        "total_nan_inf": total_nan_inf,
        "initial_objectives": initial_obj,
        "final_objectives": final_obj,
        "objective_deltas": obj_deltas,
        "avg_obj_delta": sum(obj_deltas) / len(obj_deltas),
        "all_decrease": all(d < 0 for d in obj_deltas),
        "no_crash": total_nan_inf == 0,
        "stage3_pass": total_nan_inf == 0 and all(d < 0 for d in obj_deltas),
    }


def run_s3_sweep(output_dir: str, num_rounds: int, device: str, seeds: list[int]) -> list[dict]:
    all_results = []

    logger.info("=" * 60)
    logger.info("Stage 3.1: A/B Group - Compression vs Full Upload")
    logger.info("=" * 60)

    compressors_ab = ["none", "float16", "topk_0.1", "topk_0.3", "rowtopk_0.1", "rowtopk_0.3", "lowrank_r2", "lowrank_r4", "sketch_s2", "sketch_s4"]
    configs_ab = []
    for comp in compressors_ab:
        for m in [2, 3, 5]:
            for seed in seeds:
                configs_ab.append((comp, "medium", m, 8, 0.5, 1, 1, seed))

    total = len(configs_ab)
    for idx, (comp, ms, m, k, c, si, ls, seed) in enumerate(configs_ab):
        logger.info("[%d/%d] %s m=%d seed=%d", idx + 1, total, comp, m, seed)
        try:
            result = run_single(comp, ms, m, k, c, si, ls, seed, num_rounds, torch.device(device), output_dir)
            all_results.append(result)
        except Exception as exc:
            logger.error("FAILED: %s", exc)
            all_results.append({"experiment_id": _make_exp_id(comp, ms, m, si, ls, seed), "error": str(exc), "stage3_pass": False})

    logger.info("=" * 60)
    logger.info("Stage 3.2: C Group - Sync Frequency + Local Steps")
    logger.info("=" * 60)

    configs_c = []
    for si in SYNC_INTERVALS:
        for ls in LOCAL_STEPS_LIST:
            for seed in seeds:
                configs_c.append(("none", "medium", 3, 8, 0.5, si, ls, seed))

    total_c = len(configs_c)
    for idx, (comp, ms, m, k, c, si, ls, seed) in enumerate(configs_c):
        logger.info("[%d/%d] sync=%d local_steps=%d seed=%d", idx + 1, total_c, si, ls, seed)
        try:
            result = run_single(comp, ms, m, k, c, si, ls, seed, num_rounds, torch.device(device), output_dir)
            all_results.append(result)
        except Exception as exc:
            logger.error("FAILED: %s", exc)
            all_results.append({"experiment_id": _make_exp_id(comp, ms, m, si, ls, seed), "error": str(exc), "stage3_pass": False})

    return all_results


def save_results(results: list[dict], output_dir: str) -> None:
    out_path = Path(output_dir)
    fieldnames = [
        "experiment_id", "seed", "compressor", "model_size", "num_objectives",
        "num_clients", "participation_rate", "full_sync_interval", "local_steps",
        "num_params", "num_rounds", "total_time",
        "total_upload_bytes", "baseline_upload_bytes", "upload_saving_ratio",
        "avg_compression_ratio", "total_nan_inf",
        "avg_obj_delta", "all_decrease", "no_crash", "stage3_pass",
    ]
    for i in range(10):
        fieldnames.append(f"delta_obj_{i}")

    csv_path = out_path / "s3_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            if "objective_deltas" in row and row["objective_deltas"]:
                for i, d in enumerate(row["objective_deltas"]):
                    row[f"delta_obj_{i}"] = d
            writer.writerow(row)

    json_path = out_path / "s3_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Results saved to %s and %s", csv_path, json_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedJD Stage 3 - Communication Reduction")
    parser.add_argument("--output-dir", type=str, default="results/s3_compress", help="Output directory")
    parser.add_argument("--num-rounds", type=int, default=20, help="Rounds per experiment")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 2024], help="Random seeds")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = parse_args()

    results = run_s3_sweep(args.output_dir, args.num_rounds, args.device, args.seeds)
    save_results(results, args.output_dir)

    passed = sum(1 for r in results if r.get("stage3_pass"))
    logger.info("Stage 3 complete: %d/%d experiments passed", passed, len(results))


if __name__ == "__main__":
    main()
