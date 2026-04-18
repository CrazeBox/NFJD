from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

import numpy as np


def load_sweep_csv(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            typed_row = {}
            for k, v in row.items():
                try:
                    typed_row[k] = float(v)
                except (ValueError, TypeError):
                    typed_row[k] = v
            rows.append(typed_row)
    return rows


def load_sweep_json(json_path: Path) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_single_metrics(metrics_path: Path) -> list[dict]:
    rows = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            typed_row = {}
            for k, v in row.items():
                try:
                    typed_row[k] = float(v)
                except (ValueError, TypeError):
                    typed_row[k] = v
            rows.append(typed_row)
    return rows


def analyze_sweep_results(sweep_dir: str, output_dir: str | None = None) -> None:
    sweep_path = Path(sweep_dir)
    out_path = Path(output_dir) if output_dir else sweep_path / "analysis"
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = sweep_path / "sweep_results.csv"
    json_path = sweep_path / "sweep_results.json"

    if csv_path.exists():
        summaries = load_sweep_csv(csv_path)
    elif json_path.exists():
        summaries = load_sweep_json(json_path)
    else:
        print(f"No sweep results found in {sweep_dir}")
        return

    valid = [s for s in summaries if "error" not in s and s.get("stage1_pass") is not None]
    print(f"Loaded {len(valid)} valid experiment summaries (out of {len(summaries)} total)")

    _print_aggregator_comparison(valid)
    _print_participation_rate_analysis(valid)
    _print_client_count_analysis(valid)
    _print_seed_stability_analysis(valid)

    if HAS_MPL:
        _plot_aggregator_comparison(valid, out_path)
        _plot_participation_rate_impact(valid, out_path)
        _plot_seed_stability(valid, out_path)
        _plot_objective_trajectories(sweep_path, out_path)
        print(f"\nPlots saved to {out_path}")
    else:
        print("\n[WARNING] matplotlib not available, skipping plots.")

    _save_analysis_report(valid, out_path)
    print(f"Analysis report saved to {out_path / 'analysis_report.md'}")


def _print_aggregator_comparison(summaries: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("Aggregator Comparison (all configs, mean ± std)")
    print("=" * 60)
    for agg in ["minnorm", "mean", "random"]:
        group = [s for s in summaries if s.get("aggregator") == agg]
        if not group:
            continue
        deltas_0 = [s["delta_obj_0"] for s in group if "delta_obj_0" in s and s["delta_obj_0"] != ""]
        deltas_1 = [s["delta_obj_1"] for s in group if "delta_obj_1" in s and s["delta_obj_1"] != ""]
        pass_rate = sum(1 for s in group if s.get("stage1_pass")) / len(group) * 100

        def _stats(vals):
            if not vals:
                return "N/A"
            arr = np.array(vals, dtype=float)
            return f"{arr.mean():.6f} ± {arr.std():.6f}"

        print(f"  {agg:8s}: obj0_delta={_stats(deltas_0)}, obj1_delta={_stats(deltas_1)}, pass_rate={pass_rate:.0f}%")


def _print_participation_rate_analysis(summaries: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("Participation Rate Impact")
    print("=" * 60)
    for c in sorted(set(s.get("participation_rate", 0) for s in summaries)):
        group = [s for s in summaries if s.get("participation_rate") == c]
        if not group:
            continue
        deltas_0 = [s["delta_obj_0"] for s in group if "delta_obj_0" in s and s["delta_obj_0"] != ""]
        if deltas_0:
            arr = np.array(deltas_0, dtype=float)
            print(f"  C={c:.2f}: obj0_delta={arr.mean():.6f} ± {arr.std():.6f} (n={len(group)})")


def _print_client_count_analysis(summaries: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("Client Count Impact")
    print("=" * 60)
    for k in sorted(set(int(s.get("num_clients", 0)) for s in summaries)):
        group = [s for s in summaries if int(s.get("num_clients", 0)) == k]
        if not group:
            continue
        deltas_0 = [s["delta_obj_0"] for s in group if "delta_obj_0" in s and s["delta_obj_0"] != ""]
        if deltas_0:
            arr = np.array(deltas_0, dtype=float)
            print(f"  K={k}: obj0_delta={arr.mean():.6f} ± {arr.std():.6f} (n={len(group)})")


def _print_seed_stability_analysis(summaries: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("Seed Stability (same config, different seeds)")
    print("=" * 60)
    configs = {}
    for s in summaries:
        key = f"{s.get('aggregator')}_K{int(s.get('num_clients', 0))}_C{s.get('participation_rate', 0)}"
        configs.setdefault(key, []).append(s)

    for key, group in sorted(configs.items()):
        deltas_0 = [s["delta_obj_0"] for s in group if "delta_obj_0" in s and s["delta_obj_0"] != ""]
        if len(deltas_0) >= 2:
            arr = np.array(deltas_0, dtype=float)
            spread = arr.max() - arr.min()
            print(f"  {key}: spread={spread:.6f} (min={arr.min():.6f}, max={arr.max():.6f})")


def _plot_aggregator_comparison(summaries: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for obj_idx, ax in enumerate(axes):
        key = f"delta_obj_{obj_idx}"
        agg_names = []
        agg_means = []
        agg_stds = []
        for agg in ["minnorm", "mean", "random"]:
            vals = [s[key] for s in summaries if s.get("aggregator") == agg and key in s and s[key] != ""]
            if vals:
                arr = np.array(vals, dtype=float)
                agg_names.append(agg)
                agg_means.append(arr.mean())
                agg_stds.append(arr.std())

        if agg_names:
            x = range(len(agg_names))
            ax.bar(x, agg_means, yerr=agg_stds, capsize=5, alpha=0.7, color=["tab:blue", "tab:orange", "tab:green"])
            ax.set_xticks(x)
            ax.set_xticklabels(agg_names)
            ax.set_ylabel(f"Objective {obj_idx} Delta")
            ax.set_title(f"Objective {obj_idx} - Aggregator Comparison")
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path / "aggregator_comparison.png", dpi=150)
    plt.close(fig)


def _plot_participation_rate_impact(summaries: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for agg in ["minnorm", "mean", "random"]:
        c_vals = []
        mean_deltas = []
        std_deltas = []
        for c in sorted(set(s.get("participation_rate", 0) for s in summaries)):
            group = [s for s in summaries if s.get("aggregator") == agg and s.get("participation_rate") == c]
            deltas = [s["delta_obj_0"] for s in group if "delta_obj_0" in s and s["delta_obj_0"] != ""]
            if deltas:
                arr = np.array(deltas, dtype=float)
                c_vals.append(c)
                mean_deltas.append(arr.mean())
                std_deltas.append(arr.std())

        if c_vals:
            ax.errorbar(c_vals, mean_deltas, yerr=std_deltas, marker="o", label=agg, capsize=5, linewidth=1.5)

    ax.set_xlabel("Participation Rate (C)")
    ax.set_ylabel("Objective 0 Delta")
    ax.set_title("Participation Rate Impact on Objective 0")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path / "participation_rate_impact.png", dpi=150)
    plt.close(fig)


def _plot_seed_stability(summaries: list[dict], out_path: Path) -> None:
    configs = {}
    for s in summaries:
        key = f"{s.get('aggregator')}_K{int(s.get('num_clients', 0))}_C{s.get('participation_rate', 0)}"
        configs.setdefault(key, []).append(s)

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = []
    spreads = []
    for key, group in sorted(configs.items()):
        deltas_0 = [s["delta_obj_0"] for s in group if "delta_obj_0" in s and s["delta_obj_0"] != ""]
        if len(deltas_0) >= 2:
            arr = np.array(deltas_0, dtype=float)
            labels.append(key)
            spreads.append(arr.max() - arr.min())

    if labels:
        y_pos = range(len(labels))
        ax.barh(y_pos, spreads, alpha=0.7, color="tab:purple")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Objective 0 Delta Spread (max - min)")
        ax.set_title("Seed Stability: Spread Across Seeds")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path / "seed_stability.png", dpi=150)
    plt.close(fig)


def _plot_objective_trajectories(sweep_path: Path, out_path: Path) -> None:
    exp_dirs = sorted([d for d in sweep_path.iterdir() if d.is_dir() and (d / "metrics.csv").exists()])
    if not exp_dirs:
        return

    minnorm_dirs = [d for d in exp_dirs if "minnorm" in d.name]
    if not minnorm_dirs:
        minnorm_dirs = exp_dirs[:6]

    n = min(len(minnorm_dirs), 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, exp_dir in enumerate(minnorm_dirs[:n]):
        ax = axes[idx]
        metrics = load_single_metrics(exp_dir / "metrics.csv")
        if not metrics:
            continue
        rounds = [m["round"] for m in metrics]
        obj_0 = [m.get("objective_0", float("nan")) for m in metrics]
        obj_1 = [m.get("objective_1", float("nan")) for m in metrics]
        ax.plot(rounds, obj_0, label="Obj 0", linewidth=1.5)
        ax.plot(rounds, obj_1, label="Obj 1", linewidth=1.5)
        ax.set_title(exp_dir.name, fontsize=8)
        ax.set_xlabel("Round")
        ax.set_ylabel("Objective")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Objective Trajectories (MinNorm)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path / "objective_trajectories.png", dpi=150)
    plt.close(fig)


def _save_analysis_report(summaries: list[dict], out_path: Path) -> None:
    lines = [
        "# FedJD Stage 1 Analysis Report",
        "",
        f"- Total valid experiments: {len(summaries)}",
        f"- Overall pass rate: {sum(1 for s in summaries if s.get('stage1_pass')) / max(len(summaries), 1) * 100:.1f}%",
        "",
        "## Aggregator Comparison",
        "",
        "| Aggregator | Obj0 Delta (mean±std) | Obj1 Delta (mean±std) | Pass Rate |",
        "|------------|----------------------|----------------------|-----------|",
    ]

    for agg in ["minnorm", "mean", "random"]:
        group = [s for s in summaries if s.get("aggregator") == agg]
        if not group:
            continue
        deltas_0 = [s["delta_obj_0"] for s in group if "delta_obj_0" in s and s["delta_obj_0"] != ""]
        deltas_1 = [s["delta_obj_1"] for s in group if "delta_obj_1" in s and s["delta_obj_1"] != ""]
        pass_rate = sum(1 for s in group if s.get("stage1_pass")) / len(group) * 100

        def _fmt(vals):
            if not vals:
                return "N/A"
            arr = np.array(vals, dtype=float)
            return f"{arr.mean():.6f}±{arr.std():.6f}"

        lines.append(f"| {agg} | {_fmt(deltas_0)} | {_fmt(deltas_1)} | {pass_rate:.0f}% |")

    lines.extend([
        "",
        "## Key Findings",
        "",
    ])

    minnorm_group = [s for s in summaries if s.get("aggregator") == "minnorm"]
    random_group = [s for s in summaries if s.get("aggregator") == "random"]

    if minnorm_group and random_group:
        mn_deltas = [s["delta_obj_0"] for s in minnorm_group if "delta_obj_0" in s and s["delta_obj_0"] != ""]
        rd_deltas = [s["delta_obj_0"] for s in random_group if "delta_obj_0" in s and s["delta_obj_0"] != ""]
        if mn_deltas and rd_deltas:
            mn_mean = np.mean(mn_deltas)
            rd_mean = np.mean(rd_deltas)
            if mn_mean < rd_mean:
                lines.append("- MinNorm aggregator achieves larger objective decrease than Random baseline, confirming FedJD is not running empty.")
            else:
                lines.append("- WARNING: MinNorm does NOT outperform Random baseline. Further investigation needed.")

    no_crash_count = sum(1 for s in summaries if s.get("no_crash"))
    lines.append(f"- {no_crash_count}/{len(summaries)} experiments completed without NaN/Inf.")

    converged_count = sum(1 for s in summaries if s.get("converged"))
    lines.append(f"- {converged_count}/{len(summaries)} experiments show objective decrease.")

    lines.extend([
        "",
        "## Stage 1 Gate Decision",
        "",
    ])

    all_pass = all(s.get("stage1_pass") for s in summaries)
    if all_pass:
        lines.append("**PASS** — All experiments passed Stage 1 criteria.")
    else:
        fail_count = sum(1 for s in summaries if not s.get("stage1_pass"))
        lines.append(f"**CONDITIONAL** — {fail_count} experiment(s) did not pass. Review before proceeding to Stage 2.")

    path = out_path / "analysis_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedJD Stage 1 - Analyze Sweep Results")
    parser.add_argument("--sweep-dir", type=str, default="results/s1_sweep", help="Sweep results directory")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory for analysis (default: sweep_dir/analysis)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_sweep_results(args.sweep_dir, args.output_dir or None)


if __name__ == "__main__":
    main()
