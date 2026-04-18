from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from itertools import product
from pathlib import Path

from fedjd import ExperimentConfig

from run_experiment import run_single_experiment

logger = logging.getLogger("fedjd.toy.sweep")

SEEDS = [7, 42, 2024]
PARTICIPATION_RATES = [0.25, 0.5, 1.0]
CLIENT_COUNTS = [8, 16]
AGGREGATORS = ["minnorm", "mean", "random"]


def _make_experiment_id(
    aggregator: str,
    num_clients: int,
    participation_rate: float,
    seed: int,
) -> str:
    c_str = f"C{participation_rate}".replace(".", "p")
    return f"S1-synth_m2-mlp-m2-K{num_clients}-{c_str}-E1-{aggregator}-seed{seed}"


def _build_configs(
    seeds: list[int],
    participation_rates: list[float],
    client_counts: list[int],
    aggregators: list[str],
    base_config: ExperimentConfig,
) -> list[ExperimentConfig]:
    configs = []
    for agg, k, c, seed in product(aggregators, client_counts, participation_rates, seeds):
        exp_id = _make_experiment_id(agg, k, c, seed)
        cfg = ExperimentConfig(
            experiment_id=exp_id,
            seed=seed,
            num_clients=k,
            participation_rate=c,
            aggregator=agg,
            samples_per_client=base_config.samples_per_client,
            input_dim=base_config.input_dim,
            hidden_dim=base_config.hidden_dim,
            output_dim=base_config.output_dim,
            num_rounds=base_config.num_rounds,
            batch_size=base_config.batch_size,
            learning_rate=base_config.learning_rate,
            aggregator_max_iters=base_config.aggregator_max_iters,
            aggregator_lr=base_config.aggregator_lr,
            save_checkpoints=base_config.save_checkpoints,
            checkpoint_interval=base_config.checkpoint_interval,
            device=base_config.device,
        )
        configs.append(cfg)
    return configs


def run_sweep(
    sweep_dir: str,
    seeds: list[int],
    participation_rates: list[float],
    client_counts: list[int],
    aggregators: list[str],
    base_config: ExperimentConfig,
) -> list[dict]:
    sweep_path = Path(sweep_dir)
    sweep_path.mkdir(parents=True, exist_ok=True)

    configs = _build_configs(seeds, participation_rates, client_counts, aggregators, base_config)
    logger.info("Sweep: %d total configurations", len(configs))

    all_summaries = []
    for idx, cfg in enumerate(configs):
        logger.info("[%d/%d] Running %s", idx + 1, len(configs), cfg.experiment_id)
        cfg.output_dir = str(sweep_path / cfg.experiment_id)
        try:
            summary = run_single_experiment(cfg)
            all_summaries.append(summary)
            logger.info(
                "[%d/%d] DONE | stage1_pass=%s | obj_deltas=%s",
                idx + 1,
                len(configs),
                summary["stage1_pass"],
                [f"{d:.6f}" for d in summary["objective_deltas"]],
            )
        except Exception as exc:
            logger.error("[%d/%d] FAILED | %s | Error: %s", idx + 1, len(configs), cfg.experiment_id, exc)
            all_summaries.append({
                "experiment_id": cfg.experiment_id,
                "seed": cfg.seed,
                "num_clients": cfg.num_clients,
                "participation_rate": cfg.participation_rate,
                "aggregator": cfg.aggregator,
                "error": str(exc),
                "stage1_pass": False,
            })

    _save_sweep_summary(sweep_path, all_summaries)
    _save_sweep_csv(sweep_path, all_summaries)
    _save_sweep_json(sweep_path, all_summaries)

    return all_summaries


def _save_sweep_csv(sweep_path: Path, summaries: list[dict]) -> None:
    if not summaries:
        return

    fieldnames = [
        "experiment_id", "seed", "num_clients", "participation_rate", "aggregator",
        "num_params", "num_objectives",
        "initial_obj_0", "initial_obj_1",
        "final_obj_0", "final_obj_1",
        "delta_obj_0", "delta_obj_1",
        "total_time", "avg_round_time",
        "avg_jacobian_norm", "avg_direction_norm",
        "total_upload_bytes", "total_download_bytes",
        "total_nan_inf", "peak_memory_mb",
        "converged", "no_crash", "stage1_pass",
    ]

    csv_path = sweep_path / "sweep_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for s in summaries:
            row = dict(s)
            if "initial_objectives" in row and row["initial_objectives"]:
                row["initial_obj_0"] = row["initial_objectives"][0]
                row["initial_obj_1"] = row["initial_objectives"][1] if len(row["initial_objectives"]) > 1 else ""
            if "final_objectives" in row and row["final_objectives"]:
                row["final_obj_0"] = row["final_objectives"][0]
                row["final_obj_1"] = row["final_objectives"][1] if len(row["final_objectives"]) > 1 else ""
            if "objective_deltas" in row and row["objective_deltas"]:
                row["delta_obj_0"] = row["objective_deltas"][0]
                row["delta_obj_1"] = row["objective_deltas"][1] if len(row["objective_deltas"]) > 1 else ""
            writer.writerow(row)

    logger.info("Sweep CSV saved to %s", csv_path)


def _save_sweep_json(sweep_path: Path, summaries: list[dict]) -> None:
    json_path = sweep_path / "sweep_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Sweep JSON saved to %s", json_path)


def _save_sweep_summary(sweep_path: Path, summaries: list[dict]) -> None:
    lines = [
        "# FedJD Stage 1 Sweep Summary",
        "",
        f"- Total experiments: {len(summaries)}",
        f"- Passed: {sum(1 for s in summaries if s.get('stage1_pass', False))}",
        f"- Failed: {sum(1 for s in summaries if not s.get('stage1_pass', False))}",
        "",
    ]

    valid = [s for s in summaries if "error" not in s]
    if valid:
        lines.append("## Per-Aggregator Results (mean ± std over seeds)")
        lines.append("")

        for agg in AGGREGATORS:
            agg_summaries = [s for s in valid if s.get("aggregator") == agg]
            if not agg_summaries:
                continue
            lines.append(f"### {agg}")
            lines.append("")

            for k in sorted(set(s["num_clients"] for s in agg_summaries)):
                for c in sorted(set(s["participation_rate"] for s in agg_summaries)):
                    group = [s for s in agg_summaries if s["num_clients"] == k and s["participation_rate"] == c]
                    if not group:
                        continue

                    deltas_0 = [s["objective_deltas"][0] for s in group if "objective_deltas" in s and len(s["objective_deltas"]) > 0]
                    deltas_1 = [s["objective_deltas"][1] for s in group if "objective_deltas" in s and len(s["objective_deltas"]) > 1]

                    if deltas_0:
                        mean_0 = sum(deltas_0) / len(deltas_0)
                        std_0 = (sum((d - mean_0) ** 2 for d in deltas_0) / len(deltas_0)) ** 0.5
                        best_0 = min(deltas_0)
                        worst_0 = max(deltas_0)
                    else:
                        mean_0 = std_0 = best_0 = worst_0 = float("nan")

                    if deltas_1:
                        mean_1 = sum(deltas_1) / len(deltas_1)
                        std_1 = (sum((d - mean_1) ** 2 for d in deltas_1) / len(deltas_1)) ** 0.5
                        best_1 = min(deltas_1)
                        worst_1 = max(deltas_1)
                    else:
                        mean_1 = std_1 = best_1 = worst_1 = float("nan")

                    lines.append(
                        f"- K={k}, C={c}: "
                        f"obj0_delta={mean_0:.6f}±{std_0:.6f} (best={best_0:.6f}, worst={worst_0:.6f}), "
                        f"obj1_delta={mean_1:.6f}±{std_1:.6f} (best={best_1:.6f}, worst={worst_1:.6f})"
                    )
            lines.append("")

    errors = [s for s in summaries if "error" in s]
    if errors:
        lines.append("## Failed Experiments")
        lines.append("")
        for s in errors:
            lines.append(f"- `{s['experiment_id']}`: {s['error']}")
        lines.append("")

    lines.extend([
        "## Stage 1 Gate Check",
        "",
        "- [ ] All experiments completed without crash",
        "- [ ] At least one configuration shows objective decrease across all seeds",
        "- [ ] MinNorm aggregator outperforms random baseline",
        "- [ ] No NaN/Inf in any run",
        "- [ ] Results are reproducible (same seed → same result)",
    ])

    path = sweep_path / "sweep_summary.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Sweep summary saved to %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedJD Stage 1 - Multi-config Sweep")
    parser.add_argument("--sweep-dir", type=str, default="results/s1_sweep", help="Root directory for sweep results")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="Random seeds")
    parser.add_argument("--participation-rates", type=float, nargs="+", default=PARTICIPATION_RATES, help="Participation rates to test")
    parser.add_argument("--client-counts", type=int, nargs="+", default=CLIENT_COUNTS, help="Client counts to test")
    parser.add_argument("--aggregators", type=str, nargs="+", default=AGGREGATORS, help="Aggregators to compare")
    parser.add_argument("--num-rounds", type=int, default=30, help="Number of rounds per experiment")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

    args = parse_args()
    base_config = ExperimentConfig(
        num_rounds=args.num_rounds,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )

    run_sweep(
        sweep_dir=args.sweep_dir,
        seeds=args.seeds,
        participation_rates=args.participation_rates,
        client_counts=args.client_counts,
        aggregators=args.aggregators,
        base_config=base_config,
    )


if __name__ == "__main__":
    main()
