from __future__ import annotations

import argparse
import csv
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
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def analyze_s3(sweep_dir: str, output_dir: str | None = None) -> None:
    sweep_path = Path(sweep_dir)
    out_path = Path(output_dir) if output_dir else sweep_path / "analysis"
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = sweep_path / "s3_results.csv"
    if not csv_path.exists():
        print(f"No results found at {csv_path}")
        return

    results = load_csv(csv_path)
    valid = [r for r in results if "error" not in r]
    print(f"Loaded {len(valid)} valid results (out of {len(results)} total)")

    _print_compressor_comparison(valid)
    _print_sync_frequency_analysis(valid)
    _print_pareto_analysis(valid)

    if HAS_MPL:
        _plot_pareto(valid, out_path)
        _plot_compressor_comparison(valid, out_path)
        _plot_sync_frequency(valid, out_path)
        print(f"\nPlots saved to {out_path}")

    _save_report(valid, out_path)


def _print_compressor_comparison(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("Compression vs Full Upload (A/B Group)")
    print("=" * 60)

    compressors = sorted(set(r.get("compressor") for r in results if isinstance(r.get("compressor"), str)))
    for comp in compressors:
        group = [r for r in results if r.get("compressor") == comp]
        if not group:
            continue
        deltas = [r["avg_obj_delta"] for r in group if not np.isnan(r.get("avg_obj_delta", float("nan")))]
        savings = [r["upload_saving_ratio"] for r in group if not np.isnan(r.get("upload_saving_ratio", float("nan")))]
        pass_rate = sum(1 for r in group if r.get("stage3_pass")) / len(group) * 100

        if deltas:
            print(f"  {comp:16s}: avg_delta={np.mean(deltas):.4f}, upload_saving={np.mean(savings)*100:.1f}%, pass_rate={pass_rate:.0f}%")


def _print_sync_frequency_analysis(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("Sync Frequency + Local Steps (C Group)")
    print("=" * 60)

    for si in sorted(set(_safe_int(r.get("full_sync_interval", 0)) for r in results)):
        for ls in sorted(set(_safe_int(r.get("local_steps", 0)) for r in results)):
            group = [r for r in results if _safe_int(r.get("full_sync_interval", 0)) == si and _safe_int(r.get("local_steps", 0)) == ls]
            if not group:
                continue
            deltas = [r["avg_obj_delta"] for r in group if not np.isnan(r.get("avg_obj_delta", float("nan")))]
            savings = [r["upload_saving_ratio"] for r in group if not np.isnan(r.get("upload_saving_ratio", float("nan")))]
            if deltas:
                print(f"  sync={si} ls={ls}: avg_delta={np.mean(deltas):.4f}, upload_saving={np.mean(savings)*100:.1f}%")


def _print_pareto_analysis(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("Pareto Analysis: Upload Saving vs Objective Quality")
    print("=" * 60)

    baseline = [r for r in results if r.get("compressor") == "none" and _safe_int(r.get("full_sync_interval", 0)) == 1 and _safe_int(r.get("local_steps", 0)) == 1]
    if not baseline:
        return
    baseline_delta = np.mean([r["avg_obj_delta"] for r in baseline if not np.isnan(r.get("avg_obj_delta", float("nan")))])

    configs = []
    for comp in sorted(set(r.get("compressor") for r in results if isinstance(r.get("compressor"), str))):
        if comp == "none":
            continue
        group = [r for r in results if r.get("compressor") == comp and _safe_int(r.get("full_sync_interval", 0)) == 1 and _safe_int(r.get("local_steps", 0)) == 1]
        if not group:
            continue
        deltas = [r["avg_obj_delta"] for r in group if not np.isnan(r.get("avg_obj_delta", float("nan")))]
        savings = [r["upload_saving_ratio"] for r in group if not np.isnan(r.get("upload_saving_ratio", float("nan")))]
        if deltas:
            avg_delta = np.mean(deltas)
            avg_saving = np.mean(savings)
            quality_ratio = avg_delta / baseline_delta if baseline_delta != 0 else float("inf")
            configs.append((comp, avg_delta, avg_saving, quality_ratio))

    configs.sort(key=lambda x: x[2], reverse=True)
    for comp, delta, saving, qr in configs:
        verdict = "RECOMMEND" if qr >= 0.9 else ("CAUTION" if qr >= 0.7 else "ELIMINATE")
        print(f"  {comp:16s}: saving={saving*100:.1f}%, quality={qr:.2f}x, delta={delta:.4f} -> {verdict}")


def _plot_pareto(results: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    baseline = [r for r in results if r.get("compressor") == "none" and _safe_int(r.get("full_sync_interval", 0)) == 1 and _safe_int(r.get("local_steps", 0)) == 1]
    if baseline:
        bl_delta = np.mean([r["avg_obj_delta"] for r in baseline if not np.isnan(r.get("avg_obj_delta", float("nan")))])
        ax.scatter([0], [bl_delta], marker="*", s=200, c="red", zorder=5, label="Full Upload (baseline)")

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    compressors = sorted(set(r.get("compressor") for r in results if isinstance(r.get("compressor"), str) and r.get("compressor") != "none"))
    for idx, comp in enumerate(compressors):
        group = [r for r in results if r.get("compressor") == comp and _safe_int(r.get("full_sync_interval", 0)) == 1 and _safe_int(r.get("local_steps", 0)) == 1]
        if not group:
            continue
        savings = [r["upload_saving_ratio"] * 100 for r in group if not np.isnan(r.get("upload_saving_ratio", float("nan")))]
        deltas = [r["avg_obj_delta"] for r in group if not np.isnan(r.get("avg_obj_delta", float("nan")))]
        if savings and deltas:
            ax.scatter(savings, deltas, label=comp, alpha=0.7, color=colors[idx % 10])

    ax.set_xlabel("Upload Saving (%)")
    ax.set_ylabel("Avg Objective Delta (more negative = better)")
    ax.set_title("Pareto: Communication Saving vs Optimization Quality")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path / "pareto.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_compressor_comparison(results: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    compressors = sorted(set(r.get("compressor") for r in results if isinstance(r.get("compressor"), str)))
    comp_names = []
    comp_deltas = []
    comp_savings = []
    for comp in compressors:
        group = [r for r in results if r.get("compressor") == comp]
        deltas = [r["avg_obj_delta"] for r in group if not np.isnan(r.get("avg_obj_delta", float("nan")))]
        savings = [r["upload_saving_ratio"] * 100 for r in group if not np.isnan(r.get("upload_saving_ratio", float("nan")))]
        if deltas:
            comp_names.append(comp)
            comp_deltas.append(np.mean(deltas))
            comp_savings.append(np.mean(savings))

    if comp_names:
        x = range(len(comp_names))
        axes[0].bar(x, comp_deltas, alpha=0.7)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(comp_names, rotation=45, ha="right", fontsize=7)
        axes[0].set_ylabel("Avg Objective Delta")
        axes[0].set_title("Optimization Quality by Compressor")
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(x, comp_savings, alpha=0.7, color="tab:orange")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(comp_names, rotation=45, ha="right", fontsize=7)
        axes[1].set_ylabel("Upload Saving (%)")
        axes[1].set_title("Communication Saving by Compressor")
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path / "compressor_comparison.png", dpi=150)
    plt.close(fig)


def _plot_sync_frequency(results: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sync_intervals = sorted(set(_safe_int(r.get("full_sync_interval", 0)) for r in results))
    for ls in sorted(set(_safe_int(r.get("local_steps", 0)) for r in results)):
        deltas = []
        savings = []
        valid_si = []
        for si in sync_intervals:
            group = [r for r in results if _safe_int(r.get("full_sync_interval", 0)) == si and _safe_int(r.get("local_steps", 0)) == ls]
            d = [r["avg_obj_delta"] for r in group if not np.isnan(r.get("avg_obj_delta", float("nan")))]
            s = [r["upload_saving_ratio"] * 100 for r in group if not np.isnan(r.get("upload_saving_ratio", float("nan")))]
            if d:
                valid_si.append(si)
                deltas.append(np.mean(d))
                savings.append(np.mean(s))
        if valid_si:
            axes[0].plot(valid_si, deltas, marker="o", label=f"local_steps={ls}", linewidth=1.5)
            axes[1].plot(valid_si, savings, marker="o", label=f"local_steps={ls}", linewidth=1.5)

    axes[0].set_xlabel("Full Sync Interval")
    axes[0].set_ylabel("Avg Objective Delta")
    axes[0].set_title("Quality vs Sync Interval")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Full Sync Interval")
    axes[1].set_ylabel("Upload Saving (%)")
    axes[1].set_title("Saving vs Sync Interval")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path / "sync_frequency.png", dpi=150)
    plt.close(fig)


def _save_report(results: list[dict], out_path: Path) -> None:
    valid = [r for r in results if "error" not in r]

    baseline = [r for r in valid if r.get("compressor") == "none" and _safe_int(r.get("full_sync_interval", 0)) == 1 and _safe_int(r.get("local_steps", 0)) == 1]
    baseline_delta = np.mean([r["avg_obj_delta"] for r in baseline if not np.isnan(r.get("avg_obj_delta", float("nan")))]) if baseline else 0

    lines = [
        "# FedJD Stage 3 Communication Reduction Report",
        "",
        f"- Total valid experiments: {len(valid)}",
        f"- Baseline avg objective delta: {baseline_delta:.6f}",
        "",
        "## Compression Scheme Table (A/B Group)",
        "",
        "| Compressor | Avg Delta | Quality Ratio | Upload Saving | Pass Rate | Verdict |",
        "|-----------|-----------|--------------|--------------|-----------|---------|",
    ]

    configs = []
    for comp in sorted(set(r.get("compressor") for r in valid if isinstance(r.get("compressor"), str))):
        group = [r for r in valid if r.get("compressor") == comp and _safe_int(r.get("full_sync_interval", 0)) == 1 and _safe_int(r.get("local_steps", 0)) == 1]
        if not group:
            continue
        deltas = [r["avg_obj_delta"] for r in group if not np.isnan(r.get("avg_obj_delta", float("nan")))]
        savings = [r["upload_saving_ratio"] for r in group if not np.isnan(r.get("upload_saving_ratio", float("nan")))]
        pass_rate = sum(1 for r in group if r.get("stage3_pass")) / len(group) * 100
        if deltas:
            avg_delta = np.mean(deltas)
            avg_saving = np.mean(savings)
            qr = avg_delta / baseline_delta if baseline_delta != 0 else float("inf")
            verdict = "RECOMMEND" if qr >= 0.9 else ("CAUTION" if qr >= 0.7 else "ELIMINATE")
            configs.append((comp, avg_delta, qr, avg_saving, pass_rate, verdict))
            lines.append(f"| {comp} | {avg_delta:.4f} | {qr:.2f}x | {avg_saving*100:.1f}% | {pass_rate:.0f}% | {verdict} |")

    lines.extend([
        "",
        "## Sync Frequency Table (C Group)",
        "",
        "| Sync Interval | Local Steps | Avg Delta | Upload Saving |",
        "|--------------|------------|-----------|--------------|",
    ])

    for si in sorted(set(_safe_int(r.get("full_sync_interval", 0)) for r in valid)):
        for ls in sorted(set(_safe_int(r.get("local_steps", 0)) for r in valid)):
            group = [r for r in valid if _safe_int(r.get("full_sync_interval", 0)) == si and _safe_int(r.get("local_steps", 0)) == ls]
            if not group:
                continue
            deltas = [r["avg_obj_delta"] for r in group if not np.isnan(r.get("avg_obj_delta", float("nan")))]
            savings = [r["upload_saving_ratio"] for r in group if not np.isnan(r.get("upload_saving_ratio", float("nan")))]
            if deltas:
                lines.append(f"| {si} | {ls} | {np.mean(deltas):.4f} | {np.mean(savings)*100:.1f}% |")

    recommend = [c for c in configs if c[5] == "RECOMMEND"]
    caution = [c for c in configs if c[5] == "CAUTION"]
    eliminate = [c for c in configs if c[5] == "ELIMINATE"]

    lines.extend([
        "",
        "## Recommendation Summary",
        "",
        "### RECOMMEND (quality >= 90% of baseline)",
    ])
    for comp, delta, qr, saving, pr, verdict in recommend:
        lines.append(f"- **{comp}**: saving={saving*100:.1f}%, quality={qr:.2f}x")

    lines.extend(["", "### CAUTION (quality 70-90% of baseline)"])
    for comp, delta, qr, saving, pr, verdict in caution:
        lines.append(f"- **{comp}**: saving={saving*100:.1f}%, quality={qr:.2f}x")

    lines.extend(["", "### ELIMINATE (quality < 70% of baseline)"])
    for comp, delta, qr, saving, pr, verdict in eliminate:
        lines.append(f"- **{comp}**: saving={saving*100:.1f}%, quality={qr:.2f}x")

    lines.extend([
        "",
        "## Stage 3 Gate Decision",
        "",
    ])

    if recommend:
        lines.append(f"**PASS** - {len(recommend)} compression scheme(s) achieve >= 90% quality with communication savings.")
    elif caution:
        lines.append(f"**CONDITIONAL** - No scheme achieves >= 90% quality, but {len(caution)} achieve >= 70%.")
    else:
        lines.append("**FAIL** - No viable compression scheme found.")

    path = out_path / "s3_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report saved to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedJD Stage 3 - Analyze Compression Results")
    parser.add_argument("--sweep-dir", type=str, default="results/s3_compress", help="Sweep results directory")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory for analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_s3(args.sweep_dir, args.output_dir or None)


if __name__ == "__main__":
    main()

