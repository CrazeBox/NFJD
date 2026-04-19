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
from fedjd.core import NFJDClient, NFJDServer, NFJDTrainer
from fedjd.data import make_synthetic_federated_regression
from fedjd.metrics import extract_pareto_front, hypervolume
from fedjd.models import SmallRegressor
from fedjd.problems import multi_objective_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/nfjd_phase2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_FIELDNAMES = [
    "exp_id", "method", "ablation_group", "dataset", "m", "seed",
    "num_rounds", "num_clients", "participation_rate", "learning_rate",
    "model_size", "local_epochs", "use_adaptive_rescaling",
    "use_stochastic_gramian", "local_momentum_beta", "global_momentum_beta",
    "elapsed_time", "all_decreased", "hypervolume", "pareto_gap",
    "avg_relative_improvement", "avg_upload_bytes", "avg_round_time",
    "upload_per_client", "avg_rescale_factor",
]
MAX_M = 10
for i in range(MAX_M):
    ALL_FIELDNAMES.extend([f"init_obj_{i}", f"final_obj_{i}", f"delta_obj_{i}"])


def _run_single(ablation_group, m, seed, num_rounds=50,
                num_clients=10, participation_rate=0.5, learning_rate=0.01,
                model_size="small", local_epochs=3, use_adaptive_rescaling=True,
                use_stochastic_gramian=True, local_momentum_beta=0.9,
                global_momentum_beta=0.9):
    exp_id = f"P2-{ablation_group}-m{m}-seed{seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    fed_data = make_synthetic_federated_regression(
        num_clients=num_clients, samples_per_client=100, input_dim=8,
        num_objectives=m, seed=seed)

    model = SmallRegressor(input_dim=fed_data.input_dim, output_dim=m)
    objective_fn = multi_objective_regression

    clients = [NFJDClient(
        client_id=i, dataset=fed_data.client_datasets[i], batch_size=32,
        device=device, local_epochs=local_epochs, learning_rate=learning_rate,
        local_momentum_beta=local_momentum_beta,
        use_adaptive_rescaling=use_adaptive_rescaling,
        use_stochastic_gramian=use_stochastic_gramian,
        stochastic_subset_size=4, stochastic_seed=seed
    ) for i in range(num_clients)]

    server = NFJDServer(model=model, clients=clients, objective_fn=objective_fn,
                        participation_rate=participation_rate, learning_rate=learning_rate,
                        device=device, global_momentum_beta=global_momentum_beta)

    trainer = NFJDTrainer(server=server, num_rounds=num_rounds)

    start = time.time()
    history = trainer.fit()
    elapsed = time.time() - start

    initial_obj = history[0].objective_values
    final_obj = history[-1].objective_values
    obj_history = [s.objective_values for s in history]
    all_decreased = all(final_obj[j] <= initial_obj[j] for j in range(m))

    min_vals = [min(h[j] for h in obj_history) for j in range(m)]
    max_vals = [max(h[j] for h in obj_history) for j in range(m)]
    ranges = [max_vals[j] - min_vals[j] for j in range(m)]
    for j in range(m):
        if ranges[j] < 1e-10:
            ranges[j] = 1.0

    normalized_history = [[(h[j] - min_vals[j]) / ranges[j] for j in range(m)] for h in obj_history]
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

    avg_upload = sum(s.upload_bytes for s in history) / max(len(history), 1)
    avg_round_time = sum(s.round_time for s in history) / max(len(history), 1)
    upload_per_client = avg_upload / max(participation_rate * num_clients, 1)

    rescale_vals = [s.avg_rescale_factor for s in history]
    avg_rescale = sum(rescale_vals) / len(rescale_vals) if rescale_vals else 1.0

    row = {
        "exp_id": exp_id, "method": "nfjd", "ablation_group": ablation_group,
        "dataset": "synthetic_regression", "m": m, "seed": seed,
        "num_rounds": num_rounds, "num_clients": num_clients,
        "participation_rate": participation_rate, "learning_rate": learning_rate,
        "model_size": model_size, "local_epochs": local_epochs,
        "use_adaptive_rescaling": use_adaptive_rescaling,
        "use_stochastic_gramian": use_stochastic_gramian,
        "local_momentum_beta": local_momentum_beta,
        "global_momentum_beta": global_momentum_beta,
        "elapsed_time": round(elapsed, 2), "all_decreased": all_decreased,
        "hypervolume": round(hv, 6), "pareto_gap": round(pg, 6),
        "avg_relative_improvement": round(avg_ri, 6),
        "avg_upload_bytes": round(avg_upload, 0),
        "avg_round_time": round(avg_round_time, 4),
        "upload_per_client": round(upload_per_client, 0),
        "avg_rescale_factor": round(avg_rescale, 4),
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

    logger.info("[%s] RI=%.4f NHV=%.4f rescale=%.2f time=%.1fs",
                exp_id, avg_ri, hv, avg_rescale, elapsed)
    return row


def _write_csv(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    SEEDS = [7, 42, 123]
    M_VALUES = [2, 5]
    all_rows = []
    experiments = []

    # Ablation A: AdaptiveRescaling (NFJD+AR vs NFJD no AR)
    for m in M_VALUES:
        for seed in SEEDS:
            experiments.append(dict(ablation_group="A_ar_on", m=m, seed=seed,
                use_adaptive_rescaling=True, use_stochastic_gramian=True,
                local_momentum_beta=0.9, global_momentum_beta=0.9, local_epochs=3))
            experiments.append(dict(ablation_group="A_ar_off", m=m, seed=seed,
                use_adaptive_rescaling=False, use_stochastic_gramian=True,
                local_momentum_beta=0.9, global_momentum_beta=0.9, local_epochs=3))

    # Ablation B: Global momentum (beta=0.9 vs beta=0.0)
    for m in M_VALUES:
        for seed in SEEDS:
            experiments.append(dict(ablation_group="B_gm_on", m=m, seed=seed,
                use_adaptive_rescaling=True, use_stochastic_gramian=True,
                local_momentum_beta=0.9, global_momentum_beta=0.9, local_epochs=3))
            experiments.append(dict(ablation_group="B_gm_off", m=m, seed=seed,
                use_adaptive_rescaling=True, use_stochastic_gramian=True,
                local_momentum_beta=0.9, global_momentum_beta=0.0, local_epochs=3))

    # Ablation C: Local momentum (beta=0.9 vs beta=0.0)
    for m in M_VALUES:
        for seed in SEEDS:
            experiments.append(dict(ablation_group="C_lm_on", m=m, seed=seed,
                use_adaptive_rescaling=True, use_stochastic_gramian=True,
                local_momentum_beta=0.9, global_momentum_beta=0.9, local_epochs=3))
            experiments.append(dict(ablation_group="C_lm_off", m=m, seed=seed,
                use_adaptive_rescaling=True, use_stochastic_gramian=True,
                local_momentum_beta=0.0, global_momentum_beta=0.9, local_epochs=3))

    # Ablation D: StochasticGramian (on vs off)
    for m in M_VALUES:
        for seed in SEEDS:
            experiments.append(dict(ablation_group="D_sg_on", m=m, seed=seed,
                use_adaptive_rescaling=True, use_stochastic_gramian=True,
                local_momentum_beta=0.9, global_momentum_beta=0.9, local_epochs=3))
            experiments.append(dict(ablation_group="D_sg_off", m=m, seed=seed,
                use_adaptive_rescaling=True, use_stochastic_gramian=False,
                local_momentum_beta=0.9, global_momentum_beta=0.9, local_epochs=3))

    # Ablation E: Local epochs (E=1,3,5)
    for m in M_VALUES:
        for seed in SEEDS:
            for e in [1, 3, 5]:
                experiments.append(dict(ablation_group=f"E_epoch{e}", m=m, seed=seed,
                    use_adaptive_rescaling=True, use_stochastic_gramian=True,
                    local_momentum_beta=0.9, global_momentum_beta=0.9, local_epochs=e))

    total = len(experiments)
    logger.info(f"Starting NFJD Phase 2 Ablation: {total} experiments")

    for idx, exp in enumerate(experiments):
        logger.info(f"[{idx+1}/{total}] Running {exp}...")
        try:
            row = _run_single(**exp)
            all_rows.append(row)
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()

    csv_path = RESULTS_DIR / "phase2_results.csv"
    _write_csv(csv_path, all_rows)
    logger.info(f"Phase 2 complete! {len(all_rows)}/{total} experiments, saved to {csv_path}")


if __name__ == "__main__":
    main()
