from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            typed = {}
            for k, v in row.items():
                if v == "" or v is None:
                    typed[k] = float("nan")
                else:
                    try:
                        typed[k] = float(v)
                    except (ValueError, TypeError):
                        typed[k] = v
            rows.append(typed)
    return rows


def _safe_int(val, default=0):
    if val is None:
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def analyze_s2(sweep_dir: str, output_dir: str | None = None) -> None:
    sweep_path = Path(sweep_dir)
    out_path = Path(output_dir) if output_dir else sweep_path / "analysis"
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = sweep_path / "s2_profile_results.csv"
    if not csv_path.exists():
        print(f"No results found at {csv_path}")
        return

    results = load_csv(csv_path)
    valid = [r for r in results if "error" not in r]
    print(f"Loaded {len(valid)} valid results (out of {len(results)} total)")

    _print_comm_vs_m(valid)
    _print_comm_vs_d(valid)
    _print_time_breakdown(valid)
    _print_participation_impact(valid)
    _print_jacobian_vs_gradient(valid)
    _print_bottleneck_analysis(valid)

    if HAS_MPL:
        _plot_comm_vs_m(valid, out_path)
        _plot_comm_vs_d(valid, out_path)
        _plot_time_breakdown(valid, out_path)
        _plot_jacobian_vs_gradient(valid, out_path)
        _plot_participation_impact(valid, out_path)
        print(f"\nPlots saved to {out_path}")

    _save_report(valid, out_path)


def _print_comm_vs_m(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("Communication vs Number of Objectives (m)")
    print("=" * 60)
    for m in sorted(set(_safe_int(r.get("num_objectives", 0)) for r in results)):
        group = [r for r in results if _safe_int(r.get("num_objectives", 0)) == m]
        uploads = [r["avg_upload_per_client"] for r in group if "avg_upload_per_client" in r and not np.isnan(r["avg_upload_per_client"])]
        if uploads:
            print(f"  m={m}: avg upload/client = {np.mean(uploads):.0f} bytes ({np.mean(uploads)/1024:.1f} KB)")


def _print_comm_vs_d(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("Communication vs Model Size (d)")
    print("=" * 60)
    for size in ["small", "medium", "large"]:
        group = [r for r in results if r.get("model_size") == size]
        if not group:
            continue
        d_vals = [_safe_int(r.get("num_params", 0)) for r in group]
        uploads = [r["avg_upload_per_client"] for r in group if "avg_upload_per_client" in r]
        if uploads:
            print(f"  {size} (d≈{np.mean(d_vals):.0f}): avg upload/client = {np.mean(uploads):.0f} bytes ({np.mean(uploads)/1024:.1f} KB)")


def _print_time_breakdown(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("Time Breakdown (averaged over all experiments)")
    print("=" * 60)
    client_times = [r["avg_client_compute_time"] for r in results if "avg_client_compute_time" in r]
    dir_times = [r["avg_direction_time"] for r in results if "avg_direction_time" in r]
    round_times = [r["avg_round_time"] for r in results if "avg_round_time" in r]

    if client_times and round_times:
        avg_client = np.mean(client_times)
        avg_dir = np.mean(dir_times)
        avg_round = np.mean(round_times)
        other = max(0, avg_round - avg_client - avg_dir)
        print(f"  Client Jacobian compute: {avg_client:.4f}s ({avg_client/avg_round*100:.1f}%)")
        print(f"  Direction computation:   {avg_dir:.4f}s ({avg_dir/avg_round*100:.1f}%)")
        print(f"  Other (agg+update+eval): {other:.4f}s ({other/avg_round*100:.1f}%)")
        print(f"  Total round time:        {avg_round:.4f}s")


def _print_participation_impact(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("Participation Rate Impact")
    print("=" * 60)
    for c in sorted(set(r.get("participation_rate", 0) for r in results)):
        group = [r for r in results if r.get("participation_rate") == c]
        if not group:
            continue
        uploads = [r["avg_upload_bytes"] for r in group if "avg_upload_bytes" in r]
        times = [r["avg_round_time"] for r in group if "avg_round_time" in r]
        if uploads:
            print(f"  C={c:.2f}: avg upload/round = {np.mean(uploads):.0f} B, avg round time = {np.mean(times):.4f}s")


def _print_jacobian_vs_gradient(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("Jacobian vs Gradient Upload Ratio")
    print("=" * 60)
    for m in sorted(set(_safe_int(r.get("num_objectives", 0)) for r in results)):
        group = [r for r in results if _safe_int(r.get("num_objectives", 0)) == m]
        ratios = [r["jacobian_vs_gradient_ratio"] for r in group if "jacobian_vs_gradient_ratio" in r]
        if ratios:
            print(f"  m={m}: Jacobian/Gradient ratio = {np.mean(ratios):.2f}x (theoretical = {m:.1f}x)")


def _print_bottleneck_analysis(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("Bottleneck Analysis")
    print("=" * 60)

    client_times = [r["avg_client_compute_time"] for r in results if "avg_client_compute_time" in r]
    dir_times = [r["avg_direction_time"] for r in results if "avg_direction_time" in r]
    round_times = [r["avg_round_time"] for r in results if "avg_round_time" in r]

    if client_times and round_times:
        avg_client = np.mean(client_times)
        avg_dir = np.mean(dir_times)
        avg_round = np.mean(round_times)

        if avg_client > avg_dir:
            print(f"  TIME BOTTLENECK: Client Jacobian compute ({avg_client/avg_round*100:.1f}% of round time)")
        else:
            print(f"  TIME BOTTLENECK: Direction computation ({avg_dir/avg_round*100:.1f}% of round time)")

    ratios = [r["jacobian_vs_gradient_ratio"] for r in results if "jacobian_vs_gradient_ratio" in r]
    if ratios:
        avg_ratio = np.mean(ratios)
        print(f"  COMMUNICATION: Jacobian upload is {avg_ratio:.1f}x larger than gradient upload")

    large_m = [r for r in results if _safe_int(r.get("num_objectives", 0)) >= 5]
    large_d = [r for r in results if r.get("model_size") == "large"]
    if large_m and large_d:
        m_ratios = [r["jacobian_vs_gradient_ratio"] for r in large_m if "jacobian_vs_gradient_ratio" in r]
        d_ratios = [r["jacobian_vs_gradient_ratio"] for r in large_d if "jacobian_vs_gradient_ratio" in r]
        if m_ratios and d_ratios:
            m_sensitivity = np.std(m_ratios) / max(np.mean(m_ratios), 1e-8)
            d_sensitivity = np.std(d_ratios) / max(np.mean(d_ratios), 1e-8)
            if m_sensitivity > d_sensitivity:
                print(f"  COST SENSITIVITY: More sensitive to m (CV={m_sensitivity:.3f}) than d (CV={d_sensitivity:.3f})")
            else:
                print(f"  COST SENSITIVITY: More sensitive to d (CV={d_sensitivity:.3f}) than m (CV={m_sensitivity:.3f})")


def _plot_comm_vs_m(results: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    m_vals = sorted(set(_safe_int(r.get("num_objectives", 0)) for r in results))
    for size in ["small", "medium", "large"]:
        uploads = []
        for m in m_vals:
            group = [r for r in results if _safe_int(r.get("num_objectives", 0)) == m and r.get("model_size") == size]
            if group:
                uploads.append(np.mean([r["avg_upload_per_client"] for r in group if "avg_upload_per_client" in r]) / 1024)
            else:
                uploads.append(None)
        valid_m = [m for m, u in zip(m_vals, uploads) if u is not None]
        valid_u = [u for u in uploads if u is not None]
        if valid_m:
            ax.plot(valid_m, valid_u, marker="o", label=size, linewidth=1.5)

    ax.set_xlabel("Number of Objectives (m)")
    ax.set_ylabel("Upload per Client (KB)")
    ax.set_title("Communication Cost vs Number of Objectives")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path / "comm_vs_m.png", dpi=150)
    plt.close(fig)


def _plot_comm_vs_d(results: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in sorted(set(_safe_int(r.get("num_objectives", 0)) for r in results)):
        group = [r for r in results if _safe_int(r.get("num_objectives", 0)) == m]
        d_vals = [_safe_int(r.get("num_params", 0)) for r in group]
        uploads = [r["avg_upload_per_client"] / 1024 for r in group if "avg_upload_per_client" in r]
        if d_vals and uploads:
            ax.scatter(d_vals[:len(uploads)], uploads, label=f"m={m}", alpha=0.6)

    ax.set_xlabel("Model Parameters (d)")
    ax.set_ylabel("Upload per Client (KB)")
    ax.set_title("Communication Cost vs Model Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path / "comm_vs_d.png", dpi=150)
    plt.close(fig)


def _plot_time_breakdown(results: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sizes = ["small", "medium", "large"]
    client_fracs = []
    dir_fracs = []
    other_fracs = []
    for size in sizes:
        group = [r for r in results if r.get("model_size") == size]
        if not group:
            client_fracs.append(0)
            dir_fracs.append(0)
            other_fracs.append(0)
            continue
        avg_client = np.mean([r["avg_client_compute_time"] for r in group])
        avg_dir = np.mean([r["avg_direction_time"] for r in group])
        avg_round = np.mean([r["avg_round_time"] for r in group])
        other = max(0, avg_round - avg_client - avg_dir)
        client_fracs.append(avg_client / max(avg_round, 1e-8) * 100)
        dir_fracs.append(avg_dir / max(avg_round, 1e-8) * 100)
        other_fracs.append(other / max(avg_round, 1e-8) * 100)

    x = range(len(sizes))
    axes[0].bar(x, client_fracs, label="Client Compute", alpha=0.8)
    axes[0].bar(x, dir_fracs, bottom=client_fracs, label="Direction", alpha=0.8)
    axes[0].bar(x, other_fracs, bottom=[c+d for c,d in zip(client_fracs, dir_fracs)], label="Other", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sizes)
    axes[0].set_ylabel("Time Fraction (%)")
    axes[0].set_title("Time Breakdown by Model Size")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    m_vals = sorted(set(_safe_int(r.get("num_objectives", 0)) for r in results))
    client_fracs_m = []
    dir_fracs_m = []
    other_fracs_m = []
    for m in m_vals:
        group = [r for r in results if _safe_int(r.get("num_objectives", 0)) == m]
        if not group:
            client_fracs_m.append(0)
            dir_fracs_m.append(0)
            other_fracs_m.append(0)
            continue
        avg_client = np.mean([r["avg_client_compute_time"] for r in group])
        avg_dir = np.mean([r["avg_direction_time"] for r in group])
        avg_round = np.mean([r["avg_round_time"] for r in group])
        other = max(0, avg_round - avg_client - avg_dir)
        client_fracs_m.append(avg_client / max(avg_round, 1e-8) * 100)
        dir_fracs_m.append(avg_dir / max(avg_round, 1e-8) * 100)
        other_fracs_m.append(other / max(avg_round, 1e-8) * 100)

    x2 = range(len(m_vals))
    axes[1].bar(x2, client_fracs_m, label="Client Compute", alpha=0.8)
    axes[1].bar(x2, dir_fracs_m, bottom=client_fracs_m, label="Direction", alpha=0.8)
    axes[1].bar(x2, other_fracs_m, bottom=[c+d for c,d in zip(client_fracs_m, dir_fracs_m)], label="Other", alpha=0.8)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([str(m) for m in m_vals])
    axes[1].set_ylabel("Time Fraction (%)")
    axes[1].set_title("Time Breakdown by Number of Objectives")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path / "time_breakdown.png", dpi=150)
    plt.close(fig)


def _plot_jacobian_vs_gradient(results: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    m_vals = sorted(set(_safe_int(r.get("num_objectives", 0)) for r in results))
    actual_ratios = []
    theoretical_ratios = []
    for m in m_vals:
        group = [r for r in results if _safe_int(r.get("num_objectives", 0)) == m]
        ratios = [r["jacobian_vs_gradient_ratio"] for r in group if "jacobian_vs_gradient_ratio" in r]
        if ratios:
            actual_ratios.append(np.mean(ratios))
            theoretical_ratios.append(float(m))

    if actual_ratios:
        ax.plot(m_vals, actual_ratios, marker="o", label="Measured", linewidth=2)
        ax.plot(m_vals, theoretical_ratios, marker="s", linestyle="--", label="Theoretical (m)", linewidth=1.5)
        ax.set_xlabel("Number of Objectives (m)")
        ax.set_ylabel("Jacobian / Gradient Upload Ratio")
        ax.set_title("Jacobian vs Gradient Communication Overhead")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path / "jacobian_vs_gradient.png", dpi=150)
    plt.close(fig)


def _plot_participation_impact(results: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    c_vals = sorted(set(r.get("participation_rate", 0) for r in results))
    uploads = []
    times = []
    for c in c_vals:
        group = [r for r in results if r.get("participation_rate") == c]
        u = [r["avg_upload_bytes"] for r in group if "avg_upload_bytes" in r]
        t = [r["avg_round_time"] for r in group if "avg_round_time" in r]
        if u:
            uploads.append(np.mean(u) / 1024)
            times.append(np.mean(t))
        else:
            uploads.append(0)
            times.append(0)

    if uploads:
        axes[0].plot(c_vals, uploads, marker="o", linewidth=1.5)
        axes[0].set_xlabel("Participation Rate (C)")
        axes[0].set_ylabel("Avg Upload per Round (KB)")
        axes[0].set_title("Communication vs Participation Rate")
        axes[0].grid(True, alpha=0.3)

    if times:
        axes[1].plot(c_vals, times, marker="o", color="tab:orange", linewidth=1.5)
        axes[1].set_xlabel("Participation Rate (C)")
        axes[1].set_ylabel("Avg Round Time (s)")
        axes[1].set_title("Time vs Participation Rate")
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path / "participation_impact.png", dpi=150)
    plt.close(fig)


def _save_report(results: list[dict], out_path: Path) -> None:
    valid = [r for r in results if "error" not in r]
    lines = [
        "# FedJD Stage 2 Communication & Time Profiling Report",
        "",
        f"- Total valid experiments: {len(valid)}",
        f"- Pass rate: {sum(1 for r in valid if r.get('stage2_pass')) / max(len(valid), 1) * 100:.1f}%",
        "",
        "## Communication Cost Summary",
        "",
        "### Upload per Client vs Number of Objectives",
        "",
        "| m | Upload/Client (KB) | Jacobian/Gradient Ratio | Theoretical Ratio |",
        "|---|-------------------|------------------------|-------------------|",
    ]
    for m in sorted(set(_safe_int(r.get("num_objectives", 0)) for r in valid)):
        group = [r for r in valid if _safe_int(r.get("num_objectives", 0)) == m]
        uploads = [r["avg_upload_per_client"] / 1024 for r in group if "avg_upload_per_client" in r]
        ratios = [r["jacobian_vs_gradient_ratio"] for r in group if "jacobian_vs_gradient_ratio" in r]
        if uploads:
            lines.append(f"| {m} | {np.mean(uploads):.1f} | {np.mean(ratios):.2f}x | {m:.1f}x |")

    lines.extend([
        "",
        "### Upload per Client vs Model Size",
        "",
        "| Model | d (params) | Upload/Client (KB) |",
        "|-------|-----------|-------------------|",
    ])
    for size in ["small", "medium", "large"]:
        group = [r for r in valid if r.get("model_size") == size]
        if group:
            d = np.mean([_safe_int(r.get("num_params", 0)) for r in group])
            u = np.mean([r["avg_upload_per_client"] / 1024 for r in group if "avg_upload_per_client" in r])
            lines.append(f"| {size} | {d:.0f} | {u:.1f} |")

    lines.extend([
        "",
        "## Time Breakdown",
        "",
    ])
    client_vals = [r["avg_client_compute_time"] for r in valid if "avg_client_compute_time" in r and not np.isnan(r.get("avg_client_compute_time", float("nan")))]
    dir_vals = [r["avg_direction_time"] for r in valid if "avg_direction_time" in r and not np.isnan(r.get("avg_direction_time", float("nan")))]
    round_vals = [r["avg_round_time"] for r in valid if "avg_round_time" in r and not np.isnan(r.get("avg_round_time", float("nan")))]
    if client_vals and round_vals:
        client_t = np.mean(client_vals)
        dir_t = np.mean(dir_vals)
        round_t = np.mean(round_vals)
        other_t = max(0, round_t - client_t - dir_t)
        lines.append(f"- Client Jacobian compute: {client_t:.4f}s ({client_t/round_t*100:.1f}%)")
        lines.append(f"- Direction computation: {dir_t:.4f}s ({dir_t/round_t*100:.1f}%)")
        lines.append(f"- Other (agg+update+eval): {other_t:.4f}s ({other_t/round_t*100:.1f}%)")
    else:
        client_t = dir_t = round_t = other_t = 0
        lines.append("- Time data not available")

    lines.extend([
        "",
        "## Bottleneck Conclusion",
        "",
    ])
    if client_vals and round_vals:
        if client_t > dir_t:
            lines.append("**Primary bottleneck: Client Jacobian computation**")
            lines.append("- Jacobian compute requires m backward passes per client per round")
            lines.append("- Cost scales linearly with m (number of objectives)")
        else:
            lines.append("**Primary bottleneck: Server direction computation**")
            lines.append("- MinNorm direction requires iterative optimization on the simplex")
            lines.append("- Cost scales with m^2 (gramian computation)")
    else:
        lines.append("**Insufficient time data for bottleneck analysis**")

    ratio_vals = [r["jacobian_vs_gradient_ratio"] for r in valid if "jacobian_vs_gradient_ratio" in r and not np.isnan(r.get("jacobian_vs_gradient_ratio", float("nan")))]
    avg_ratio = np.mean(ratio_vals) if ratio_vals else 0
    lines.extend([
        "",
        "## Communication Overhead",
        "",
        f"- Average Jacobian/Gradient upload ratio: **{avg_ratio:.1f}x**",
        "- This ratio equals m (number of objectives) in theory",
        "- Jacobian upload is the dominant communication cost",
        "",
        "## Compression Priority Ranking",
        "",
        "1. **Jacobian row compression** (reduce m rows -> fewer rows)",
        "2. **Jacobian quantization** (float32 -> float16/int8)",
        "3. **Jacobian sparsification** (top-k gradient per row)",
        "4. **Local multi-step** (reduce communication rounds at cost of staleness)",
    ])

    path = out_path / "s2_profiling_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report saved to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedJD Stage 2 - Analyze Profiling Results")
    parser.add_argument("--sweep-dir", type=str, default="results/s2_profile", help="Sweep results directory")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory for analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_s2(args.sweep_dir, args.output_dir or None)


if __name__ == "__main__":
    main()
