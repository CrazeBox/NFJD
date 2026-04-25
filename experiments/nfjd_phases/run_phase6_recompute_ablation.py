"""NFJD Phase 6: Recompute Interval Ablation Study

Compare recompute_interval = 1, 2, 4 on synthetic data to measure
quality vs speed tradeoff.
"""
from __future__ import annotations

import csv
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import torch

from fedjd.core import NFJDClient, NFJDServer, NFJDTrainer
from fedjd.data import make_synthetic_federated_regression, make_high_conflict_federated_regression
from fedjd.experiments.nfjd_phases.phase5_utils import evaluate_model, fill_regression_metrics
from fedjd.models import SmallRegressor
from fedjd.problems import multi_objective_regression

RESULTS_DIR = Path("results/nfjd_phase6_recompute")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "phase6_run.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

ALL_FIELDNAMES = [
    "exp_id", "recompute_interval", "dataset", "m", "seed", "num_rounds",
    "elapsed_time", "avg_round_time", "avg_mse", "max_mse", "mse_std", "avg_r2",
]


def _run_single(recompute_interval, dataset, m, seed, num_rounds=50,
                num_clients=10, participation_rate=0.5, learning_rate=0.01,
                conflict_strength=0.0):
    exp_id = f"P6-ri{recompute_interval}-{dataset}-m{m}-seed{seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    if dataset == "highconflict":
        fed_data = make_high_conflict_federated_regression(
            num_clients=num_clients, samples_per_client=100, input_dim=8,
            num_objectives=m, conflict_strength=conflict_strength, seed=seed)
    else:
        fed_data = make_synthetic_federated_regression(
            num_clients=num_clients, samples_per_client=100, input_dim=8,
            num_objectives=m, seed=seed)

    model = SmallRegressor(input_dim=fed_data.input_dim, output_dim=m)

    clients = [
        NFJDClient(
            client_id=i, dataset=fed_data.client_datasets[i], batch_size=32,
            device=device, local_epochs=3, learning_rate=learning_rate,
            local_momentum_beta=0.9, use_adaptive_rescaling=True,
            use_stochastic_gramian=True, stochastic_subset_size=4,
            stochastic_seed=seed + i, recompute_interval=recompute_interval,
        )
        for i in range(num_clients)
    ]
    server = NFJDServer(
        model=model, clients=clients, objective_fn=multi_objective_regression,
        participation_rate=participation_rate, learning_rate=learning_rate,
        device=device, global_momentum_beta=0.9,
        parallel_clients=False, eval_dataset=fed_data.val_dataset,
    )
    trainer = NFJDTrainer(server=server, num_rounds=num_rounds)

    start = time.time()
    history = trainer.fit()
    elapsed = time.time() - start

    avg_round_time = sum(s.round_time for s in history) / max(len(history), 1)
    predictions, targets = evaluate_model(trainer.server.model, fed_data.test_dataset, device, batch_size=256)

    row = {
        "exp_id": exp_id,
        "recompute_interval": recompute_interval,
        "dataset": dataset,
        "m": m,
        "seed": seed,
        "num_rounds": num_rounds,
        "elapsed_time": round(elapsed, 2),
        "avg_round_time": round(avg_round_time, 4),
        "avg_mse": "", "max_mse": "", "mse_std": "", "avg_r2": "",
    }
    row = fill_regression_metrics(row, predictions, targets, m)

    logger.info(
        "[%s] MSE=%.6f maxMSE=%.6f R2=%.6f time=%.1fs round_time=%.4fs",
        exp_id, float(row["avg_mse"]), float(row["max_mse"]), float(row["avg_r2"]), elapsed, avg_round_time,
    )
    return row


def _write_csv(csv_path, rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    SEEDS = [7, 42, 123]
    RECOMPUTE_INTERVALS = [1, 2, 4]
    M_VALUES = [2, 3, 5]
    CONFLICT_STRENGTHS = [0.0, 0.5, 1.0]

    experiments = []
    for m in M_VALUES:
        for ri in RECOMPUTE_INTERVALS:
            for seed in SEEDS:
                experiments.append(dict(
                    recompute_interval=ri, dataset="synthetic",
                    m=m, seed=seed, num_rounds=50))
            for cs in CONFLICT_STRENGTHS:
                for seed in SEEDS:
                    experiments.append(dict(
                        recompute_interval=ri, dataset="highconflict",
                        m=m, seed=seed, num_rounds=50,
                        conflict_strength=cs))

    all_rows = []
    total = len(experiments)
    logger.info(f"Starting Phase 6 Recompute Interval Ablation: {total} experiments")

    for idx, exp in enumerate(experiments):
        logger.info(f"[{idx+1}/{total}] Running {exp}")
        try:
            row = _run_single(**exp)
            all_rows.append(row)
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()

    csv_path = RESULTS_DIR / "phase6_results.csv"
    _write_csv(csv_path, all_rows)

    # Summary table
    logger.info("\n" + "=" * 100)
    logger.info("RECOMPUTE INTERVAL ABLATION SUMMARY")
    logger.info("=" * 100)
    for ri in RECOMPUTE_INTERVALS:
        ri_rows = [r for r in all_rows if r["recompute_interval"] == ri]
        if not ri_rows:
            continue
        avg_mse = sum(r["avg_mse"] for r in ri_rows) / len(ri_rows)
        max_mse = sum(r["max_mse"] for r in ri_rows) / len(ri_rows)
        avg_r2 = sum(r["avg_r2"] for r in ri_rows) / len(ri_rows)
        avg_time = sum(r["elapsed_time"] for r in ri_rows) / len(ri_rows)
        avg_rt = sum(r["avg_round_time"] for r in ri_rows) / len(ri_rows)
        speedup = avg_time / (sum(r["elapsed_time"] for r in all_rows if r["recompute_interval"] == 1) / len([r for r in all_rows if r["recompute_interval"] == 1])) if ri != 1 else 1.0
        logger.info(
            f"recompute={ri:2d}: avg_MSE={avg_mse:.6f}, max_MSE={max_mse:.6f}, avg_R2={avg_r2:.6f}, "
            f"avg_time={avg_time:.1f}s, avg_round={avg_rt:.4f}s, speedup={1.0 if ri == 1 else avg_time / (sum(r['elapsed_time'] for r in all_rows if r['recompute_interval'] == 1) / max(len([r for r in all_rows if r['recompute_interval'] == 1]), 1)):.2f}x"
        )

    logger.info(f"\nPhase 6 complete! {len(all_rows)}/{total} experiments, saved to {csv_path}")


if __name__ == "__main__":
    main()
