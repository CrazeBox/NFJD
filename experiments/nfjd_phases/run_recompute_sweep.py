"""Quick exploration: recompute_interval sweep with lightweight CLI controls."""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import torch

from fedjd.core import NFJDClient, NFJDServer, NFJDTrainer
from fedjd.data import make_high_conflict_federated_regression, make_synthetic_federated_regression
from fedjd.metrics import extract_pareto_front, hypervolume
from fedjd.models import SmallRegressor
from fedjd.problems import multi_objective_regression

OUT = Path("results/nfjd_tools/recompute_sweep_results.txt")


def log(msg: str) -> None:
    print(msg)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run_one(
    recompute_interval: int,
    conflict_strength: float,
    num_objectives: int,
    seed: int,
    rounds: int,
    num_clients: int,
    samples_per_client: int,
    local_epochs: int,
    participation_rate: float,
    learning_rate: float,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    if conflict_strength > 0:
        data = make_high_conflict_federated_regression(
            num_clients=num_clients,
            samples_per_client=samples_per_client,
            input_dim=8,
            num_objectives=num_objectives,
            conflict_strength=conflict_strength,
            seed=seed,
        )
    else:
        data = make_synthetic_federated_regression(
            num_clients=num_clients,
            samples_per_client=samples_per_client,
            input_dim=8,
            num_objectives=num_objectives,
            seed=seed,
        )

    model = SmallRegressor(input_dim=data.input_dim, output_dim=num_objectives)
    clients = [
        NFJDClient(
            client_id=i,
            dataset=data.client_datasets[i],
            batch_size=32,
            device=device,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            local_momentum_beta=0.9,
            use_adaptive_rescaling=True,
            use_stochastic_gramian=True,
            stochastic_subset_size=4,
            stochastic_seed=seed,
            recompute_interval=recompute_interval,
        )
        for i in range(num_clients)
    ]
    server = NFJDServer(
        model=model,
        clients=clients,
        objective_fn=multi_objective_regression,
        participation_rate=participation_rate,
        learning_rate=learning_rate,
        device=device,
        global_momentum_beta=0.9,
        parallel_clients=False,
    )
    trainer = NFJDTrainer(server=server, num_rounds=rounds)

    t0 = time.time()
    history = trainer.fit()
    elapsed = time.time() - t0

    init_obj = history[0].objective_values
    final_obj = history[-1].objective_values
    obj_history = [s.objective_values for s in history]

    min_vals = [min(h[j] for h in obj_history) for j in range(num_objectives)]
    max_vals = [max(h[j] for h in obj_history) for j in range(num_objectives)]
    ranges = [max(max_vals[j] - min_vals[j], 1e-10) for j in range(num_objectives)]

    normalized_history = [
        [(h[j] - min_vals[j]) / ranges[j] for j in range(num_objectives)]
        for h in obj_history
    ]
    pareto_front = extract_pareto_front(normalized_history)
    raw_hv = hypervolume(pareto_front, [1.1] * num_objectives)
    hv = raw_hv / (1.1 ** num_objectives)

    ri_sum = 0.0
    for j in range(num_objectives):
        if abs(init_obj[j]) > 1e-10:
            ri_sum += (init_obj[j] - final_obj[j]) / abs(init_obj[j])
        else:
            ri_sum += 1.0 if final_obj[j] < abs(init_obj[j]) else 0.0
    avg_ri = ri_sum / num_objectives

    all_decreased = all(final_obj[j] <= init_obj[j] for j in range(num_objectives))
    avg_round_time = sum(s.round_time for s in history) / max(len(history), 1)

    return {
        "ri": recompute_interval,
        "conflict": conflict_strength,
        "m": num_objectives,
        "seed": seed,
        "avg_ri": avg_ri,
        "hv": hv,
        "all_decr": all_decreased,
        "elapsed": elapsed,
        "avg_rt": avg_round_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick recompute-interval sweep for NFJD.")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--num-clients", type=int, default=6)
    parser.add_argument("--samples-per-client", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--participation-rate", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--m-values", type=int, nargs="+", default=[5])
    parser.add_argument("--recompute-intervals", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--conflicts", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42])
    args = parser.parse_args()

    if OUT.exists():
        OUT.unlink()

    log("=" * 100)
    log("RECOMPUTE INTERVAL SWEEP EXPLORATION")
    log("=" * 100)
    log(
        f"config: rounds={args.rounds}, num_clients={args.num_clients}, samples_per_client={args.samples_per_client}, "
        f"local_epochs={args.local_epochs}, participation_rate={args.participation_rate}, "
        f"m_values={args.m_values}, recompute_intervals={args.recompute_intervals}, conflicts={args.conflicts}, seeds={args.seeds}"
    )

    results = []
    total = len(args.recompute_intervals) * len(args.m_values) * len(args.conflicts) * len(args.seeds)
    idx = 0

    for num_objectives in args.m_values:
        for conflict_strength in args.conflicts:
            for recompute_interval in args.recompute_intervals:
                for seed in args.seeds:
                    idx += 1
                    tag = f"m={num_objectives} cs={conflict_strength} ri={recompute_interval} seed={seed}"
                    log(f"[{idx}/{total}] Running {tag}...")
                    try:
                        result = run_one(
                            recompute_interval=recompute_interval,
                            conflict_strength=conflict_strength,
                            num_objectives=num_objectives,
                            seed=seed,
                            rounds=args.rounds,
                            num_clients=args.num_clients,
                            samples_per_client=args.samples_per_client,
                            local_epochs=args.local_epochs,
                            participation_rate=args.participation_rate,
                            learning_rate=args.learning_rate,
                        )
                        results.append(result)
                        log(
                            f"  RI={result['avg_ri']:.4f} HV={result['hv']:.4f} "
                            f"time={result['elapsed']:.1f}s round={result['avg_rt']:.3f}s decr={result['all_decr']}"
                        )
                    except Exception as exc:
                        log(f"  FAILED: {exc}")

    log("\n" + "=" * 100)
    log("SUMMARY: Average metrics per recompute_interval")
    log("=" * 100)
    log(f"{'RI':>3s} | {'avg_RI':>8s} | {'HV':>8s} | {'avg_rt(s)':>10s} | {'elapsed(s)':>11s} | {'decr%':>6s} | {'speedup':>8s}")
    log("-" * 100)

    base_elapsed = None
    for recompute_interval in args.recompute_intervals:
        rows = [r for r in results if r["ri"] == recompute_interval]
        if not rows:
            continue
        avg_ri = sum(r["avg_ri"] for r in rows) / len(rows)
        avg_hv = sum(r["hv"] for r in rows) / len(rows)
        avg_rt = sum(r["avg_rt"] for r in rows) / len(rows)
        avg_elapsed = sum(r["elapsed"] for r in rows) / len(rows)
        decr_pct = sum(1 for r in rows if r["all_decr"]) / len(rows)
        if recompute_interval == args.recompute_intervals[0]:
            base_elapsed = avg_elapsed
            speedup = 1.0
        else:
            speedup = (base_elapsed / avg_elapsed) if base_elapsed and avg_elapsed > 0 else 0.0
        log(f"{recompute_interval:>3d} | {avg_ri:>8.4f} | {avg_hv:>8.4f} | {avg_rt:>10.4f} | {avg_elapsed:>11.1f} | {decr_pct:>5.0%} | {speedup:>7.2f}x")

    log("\n" + "=" * 100)
    log("PER-SCENARIO BREAKDOWN")
    log("=" * 100)
    for num_objectives in args.m_values:
        for conflict_strength in args.conflicts:
            log(f"\nm={num_objectives}, conflict_strength={conflict_strength}")
            log(f"{'RI':>3s} | {'avg_RI':>8s} | {'HV':>8s} | {'avg_rt(s)':>10s} | {'speedup':>8s}")
            log("-" * 70)
            base_rt = None
            for recompute_interval in args.recompute_intervals:
                rows = [r for r in results if r["ri"] == recompute_interval and r["m"] == num_objectives and r["conflict"] == conflict_strength]
                if not rows:
                    continue
                avg_ri = sum(r["avg_ri"] for r in rows) / len(rows)
                avg_hv = sum(r["hv"] for r in rows) / len(rows)
                avg_rt = sum(r["avg_rt"] for r in rows) / len(rows)
                if recompute_interval == args.recompute_intervals[0]:
                    base_rt = avg_rt
                    speedup = 1.0
                else:
                    speedup = (base_rt / avg_rt) if base_rt and avg_rt > 0 else 0.0
                log(f"{recompute_interval:>3d} | {avg_ri:>8.4f} | {avg_hv:>8.4f} | {avg_rt:>10.4f} | {speedup:>7.2f}x")

    log("\n" + "=" * 100)
    log("EXPLORATION COMPLETE")
    log("=" * 100)
    log(f"\nResults saved to: {OUT}")


if __name__ == "__main__":
    main()
