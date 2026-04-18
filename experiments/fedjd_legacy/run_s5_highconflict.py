from __future__ import annotations

import csv
import logging
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from fedjd.aggregators import MinNormAggregator
from fedjd.compressors import Float16Compressor, NoCompressor
from fedjd.core import (
    DirectionAvgServer,
    FedJDClient,
    FedJDServer,
    FedJDTrainer,
    FMGDAServer,
    WeightedSumServer,
)
from fedjd.data import make_high_conflict_federated_regression
from fedjd.metrics import extract_pareto_front, hypervolume
from fedjd.models import MediumRegressor, SmallRegressor
from fedjd.problems import multi_objective_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("e:/AIProject/results/s5_highconflict")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_FIELDNAMES = [
    "exp_id", "method", "dataset", "m", "seed", "num_rounds", "num_clients",
    "participation_rate", "learning_rate", "conflict_strength", "use_float16",
    "model_size", "elapsed_time", "all_decreased", "any_nan", "total_nan_inf",
    "hypervolume", "pareto_gap", "num_pareto_points", "avg_upload_bytes",
    "avg_round_time", "upload_per_client", "avg_relative_improvement",
    "gradient_conflict_cos",
]
MAX_M = 10
for i in range(MAX_M):
    ALL_FIELDNAMES.extend([f"init_obj_{i}", f"final_obj_{i}", f"delta_obj_{i}"])


def _measure_gradient_conflict(model, clients, objective_fn, device, num_samples=3):
    model.train()
    cos_sums = []
    count = 0
    for client in clients[:num_samples]:
        loader = torch.utils.data.DataLoader(client.dataset, batch_size=32, shuffle=True)
        batch_inputs, batch_targets = next(iter(loader))
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        predictions = model(batch_inputs)
        objective_values = objective_fn(predictions, batch_targets, batch_inputs)
        model.zero_grad(set_to_none=True)
        grads = []
        for idx, obj in enumerate(objective_values):
            retain = idx < len(objective_values) - 1
            model.zero_grad(set_to_none=True)
            obj.backward(retain_graph=retain)
            g = torch.cat([p.grad.detach().reshape(-1).clone() if p.grad is not None else torch.zeros_like(p).reshape(-1) for p in model.parameters()])
            grads.append(g)
        model.zero_grad(set_to_none=True)
        for i in range(len(grads)):
            for j in range(i + 1, len(grads)):
                cos = torch.nn.functional.cosine_similarity(grads[i].unsqueeze(0), grads[j].unsqueeze(0)).item()
                cos_sums.append(cos)
                count += 1
    if not cos_sums:
        return 0.0
    return sum(cos_sums) / len(cos_sums)


def _run_single(
    method: str,
    dataset: str,
    m: int,
    seed: int,
    conflict_strength: float = 1.0,
    num_rounds: int = 100,
    num_clients: int = 10,
    participation_rate: float = 0.5,
    learning_rate: float = 0.01,
    use_float16: bool = False,
    model_size: str = "small",
):
    exp_id = f"S5-{method}-{dataset}-m{m}-cs{conflict_strength:.1f}-seed{seed}"
    device = torch.device("cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    fed_data = make_high_conflict_federated_regression(
        num_clients=num_clients,
        samples_per_client=100,
        input_dim=8,
        num_objectives=m,
        noise_std=0.1,
        conflict_strength=conflict_strength,
        seed=seed,
    )

    if model_size == "medium":
        model = MediumRegressor(input_dim=fed_data.input_dim, output_dim=m)
    else:
        model = SmallRegressor(input_dim=fed_data.input_dim, output_dim=m)

    clients = [
        FedJDClient(client_id=i, dataset=fed_data.client_datasets[i], batch_size=32, device=device)
        for i in range(num_clients)
    ]

    objective_fn = multi_objective_regression
    compressor = Float16Compressor() if use_float16 else NoCompressor()
    aggregator = MinNormAggregator(max_iters=250, lr=0.1, max_direction_norm=0.0)

    if method == "fedjd":
        server = FedJDServer(
            model=model, clients=clients, aggregator=aggregator,
            objective_fn=objective_fn, participation_rate=participation_rate,
            learning_rate=learning_rate, device=device, compressor=compressor,
        )
    elif method == "fmgda":
        server = FMGDAServer(
            model=model, clients=clients, objective_fn=objective_fn,
            participation_rate=participation_rate, learning_rate=learning_rate, device=device,
        )
    elif method == "weighted_sum":
        server = WeightedSumServer(
            model=model, clients=clients, objective_fn=objective_fn,
            participation_rate=participation_rate, learning_rate=learning_rate, device=device,
        )
    elif method == "direction_avg":
        server = DirectionAvgServer(
            model=model, clients=clients, objective_fn=objective_fn,
            participation_rate=participation_rate, learning_rate=learning_rate, device=device,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    trainer = FedJDTrainer(server=server, num_rounds=num_rounds)
    start = time.time()
    history = trainer.fit()
    elapsed = time.time() - start

    initial_obj = history[0].objective_values if history else [0.0] * m
    final_obj = history[-1].objective_values if history else [0.0] * m

    obj_history = [s.objective_values for s in history]
    all_decreased = all(final_obj[j] <= initial_obj[j] for j in range(m))
    any_nan = any(s.nan_inf_count > 0 for s in history)
    total_nan_inf = sum(s.nan_inf_count for s in history)

    min_vals = [min(h[j] for h in obj_history) for j in range(m)]
    max_vals = [max(h[j] for h in obj_history) for j in range(m)]
    ranges = [max_vals[j] - min_vals[j] for j in range(m)]
    for j in range(m):
        if ranges[j] < 1e-10:
            ranges[j] = 1.0

    normalized_history = []
    for h in obj_history:
        normalized_history.append([(h[j] - min_vals[j]) / ranges[j] for j in range(m)])

    ref_point = [1.1] * m
    pareto_front = extract_pareto_front(normalized_history)
    raw_hv = hypervolume(pareto_front, ref_point)
    max_possible_hv = 1.1 ** m
    hv = raw_hv / max_possible_hv if max_possible_hv > 0 else 0.0

    normalized_final = [(final_obj[j] - min_vals[j]) / ranges[j] for j in range(m)]
    pg = sum(normalized_final) / m

    ri_sum = 0.0
    for j in range(m):
        if abs(initial_obj[j]) > 1e-10:
            ri_sum += (initial_obj[j] - final_obj[j]) / abs(initial_obj[j])
        else:
            ri_sum += 1.0 if final_obj[j] < abs(initial_obj[j]) else 0.0
    avg_ri = ri_sum / m

    avg_upload = sum(s.upload_bytes for s in history if s.is_full_sync_round) / max(sum(1 for s in history if s.is_full_sync_round), 1)
    avg_round_time = sum(s.round_time for s in history) / max(len(history), 1)

    upload_per_client = 0
    for s in history:
        if s.is_full_sync_round and s.jacobian_upload_per_client > 0:
            upload_per_client = s.jacobian_upload_per_client
            break
        elif s.is_full_sync_round and s.gradient_upload_per_client > 0:
            upload_per_client = s.gradient_upload_per_client
            break

    avg_cos = _measure_gradient_conflict(server.model, clients, objective_fn, device)

    row = {
        "exp_id": exp_id,
        "method": method,
        "dataset": dataset,
        "m": m,
        "seed": seed,
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "participation_rate": participation_rate,
        "learning_rate": learning_rate,
        "conflict_strength": conflict_strength,
        "use_float16": use_float16,
        "model_size": model_size,
        "elapsed_time": round(elapsed, 2),
        "all_decreased": all_decreased,
        "any_nan": any_nan,
        "total_nan_inf": total_nan_inf,
        "hypervolume": round(hv, 6),
        "pareto_gap": round(pg, 6),
        "num_pareto_points": len(pareto_front),
        "avg_upload_bytes": round(avg_upload, 0),
        "avg_round_time": round(avg_round_time, 4),
        "upload_per_client": upload_per_client,
        "avg_relative_improvement": round(avg_ri, 6),
        "gradient_conflict_cos": round(avg_cos, 4),
    }
    for i in range(MAX_M):
        if i < m:
            row[f"init_obj_{i}"] = round(initial_obj[i], 6)
            row[f"final_obj_{i}"] = round(final_obj[i], 6)
            row[f"delta_obj_{i}"] = round(final_obj[i] - initial_obj[i], 6)
        else:
            row[f"init_obj_{i}"] = ""
            row[f"final_obj_{i}"] = ""
            row[f"delta_obj_{i}"] = ""

    logger.info("[%s] %s: NHV=%.4f RI=%.4f cos=%.4f upload=%d",
                exp_id, method, hv, avg_ri, avg_cos, upload_per_client)
    return row


def _write_csv(csv_path: Path, rows: list[dict]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    SEEDS = [7, 42, 123]
    METHODS = ["fedjd", "fmgda", "weighted_sum", "direction_avg"]
    all_rows = []
    exp_count = 0

    experiments = []

    for m in [2, 3, 5]:
        for cs in [0.5, 1.0, 2.0]:
            for method in METHODS:
                for seed in SEEDS:
                    experiments.append({
                        "method": method, "dataset": "highconflict_regression",
                        "m": m, "seed": seed, "conflict_strength": cs,
                        "num_rounds": 100, "model_size": "small",
                    })

    for m in [2, 3, 5]:
        for method in METHODS:
            for seed in SEEDS:
                experiments.append({
                    "method": method, "dataset": "highconflict_regression",
                    "m": m, "seed": seed, "conflict_strength": 1.0,
                    "num_rounds": 200, "model_size": "medium",
                })

    total = len(experiments)
    logger.info(f"Starting Stage 5 high-conflict experiments: {total} experiments")

    for exp in experiments:
        exp_count += 1
        logger.info(f"[{exp_count}/{total}] Running {exp}...")
        try:
            row = _run_single(**exp)
            all_rows.append(row)
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()

    csv_path = RESULTS_DIR / "s5_results.csv"
    _write_csv(csv_path, all_rows)
    logger.info(f"Stage 5 complete! {len(all_rows)}/{total} experiments, saved to {csv_path}")


if __name__ == "__main__":
    main()
