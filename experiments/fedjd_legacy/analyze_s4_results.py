from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

RESULTS_DIR = Path("e:/AIProject/results/s4_benchmark")
ANALYSIS_DIR = RESULTS_DIR / "analysis"


def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def _safe_float(v, default=0.0):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, str):
        if v.strip() == "":
            return default
        if v.lower() in ("true", "1"):
            return 1.0
        if v.lower() in ("false", "0"):
            return 0.0
        try:
            return float(v)
        except ValueError:
            return default
    return float(v)


def _safe_int(v, default=0):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return int(v)


def _safe_str(v, default=""):
    if v is None:
        return default
    return str(v)


def _mean(vals):
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def _std(vals):
    vals = [v for v in vals if not math.isnan(v)]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def generate_main_results_table(rows: list[dict]) -> str:
    lines = ["# Stage 4 Main Results Table\n"]
    lines.append("## Group A: Method Comparison (averaged over 3 seeds)\n")

    groups = defaultdict(list)
    for r in rows:
        task = _safe_str(r.get("task"))
        m = _safe_int(r.get("m"))
        method = _safe_str(r.get("method"))
        noniid = _safe_float(r.get("noniid_strength"))
        use_f16 = r.get("use_float16", False)
        if use_f16 is True or use_f16 == "True" or use_f16 == 1.0:
            continue
        if noniid > 0:
            continue
        key = (task, m, method)
        groups[key].append(r)

    table_data = defaultdict(lambda: {"hv": [], "pg": [], "ri": [], "all_dec": [], "upload": [], "time": []})
    for (task, m, method), rs in groups.items():
        for r in rs:
            table_data[(task, m, method)]["hv"].append(_safe_float(r.get("hypervolume")))
            table_data[(task, m, method)]["pg"].append(_safe_float(r.get("pareto_gap")))
            table_data[(task, m, method)]["ri"].append(_safe_float(r.get("avg_relative_improvement")))
            table_data[(task, m, method)]["all_dec"].append(1 if r.get("all_decreased") in (True, "True", 1, 1.0) else 0)
            table_data[(task, m, method)]["upload"].append(_safe_float(r.get("avg_upload_bytes")))
            table_data[(task, m, method)]["time"].append(_safe_float(r.get("avg_round_time")))

    all_keys = sorted(table_data.keys(), key=lambda x: (x[0], x[1], x[2]))
    lines.append("| Task | m | Method | NHV (mean±std) | NPG (mean±std) | Avg RI (mean±std) | All Dec. | Upload (B) | Time (s) |")
    lines.append("|------|---|--------|----------------|----------------|-------------------|----------|------------|----------|")
    for task, m, method in all_keys:
        d = table_data[(task, m, method)]
        hv_m = _mean(d["hv"])
        hv_s = _std(d["hv"])
        pg_m = _mean(d["pg"])
        pg_s = _std(d["pg"])
        ri_m = _mean(d["ri"])
        ri_s = _std(d["ri"])
        dec_rate = _mean(d["all_dec"])
        up_m = _mean(d["upload"])
        t_m = _mean(d["time"])
        lines.append(f"| {task} | {m} | {method} | {hv_m:.4f}±{hv_s:.4f} | {pg_m:.4f}±{pg_s:.4f} | {ri_m:.4f}±{ri_s:.4f} | {dec_rate:.0%} | {up_m:.0f} | {t_m:.4f} |")

    lines.append("")
    return "\n".join(lines)


def generate_noniid_ablation(rows: list[dict]) -> str:
    lines = ["## Group B: Non-IID Ablation\n"]

    groups = defaultdict(list)
    for r in rows:
        task = _safe_str(r.get("task"))
        method = _safe_str(r.get("method"))
        noniid = _safe_float(r.get("noniid_strength"))
        m = _safe_int(r.get("m"))
        use_f16 = r.get("use_float16", False)
        if use_f16 is True or use_f16 == "True" or use_f16 == 1.0:
            continue
        if task != "classification" or m != 2:
            continue
        key = (method, noniid)
        groups[key].append(r)

    table_data = defaultdict(lambda: {"hv": [], "pg": [], "ri": [], "all_dec": []})
    for (method, noniid), rs in groups.items():
        for r in rs:
            table_data[(method, noniid)]["hv"].append(_safe_float(r.get("hypervolume")))
            table_data[(method, noniid)]["pg"].append(_safe_float(r.get("pareto_gap")))
            table_data[(method, noniid)]["ri"].append(_safe_float(r.get("avg_relative_improvement")))
            table_data[(method, noniid)]["all_dec"].append(1 if r.get("all_decreased") in (True, "True", 1, 1.0) else 0)

    all_keys = sorted(table_data.keys(), key=lambda x: (x[0], x[1]))
    lines.append("| Method | Non-IID | NHV (mean±std) | NPG (mean±std) | Avg RI (mean±std) | All Dec. |")
    lines.append("|--------|---------|----------------|----------------|-------------------|----------|")
    for method, noniid in all_keys:
        d = table_data[(method, noniid)]
        hv_m = _mean(d["hv"])
        hv_s = _std(d["hv"])
        pg_m = _mean(d["pg"])
        pg_s = _std(d["pg"])
        ri_m = _mean(d["ri"])
        ri_s = _std(d["ri"])
        dec_rate = _mean(d["all_dec"])
        lines.append(f"| {method} | {noniid:.1f} | {hv_m:.4f}±{hv_s:.4f} | {pg_m:.4f}±{pg_s:.4f} | {ri_m:.4f}±{ri_s:.4f} | {dec_rate:.0%} |")

    lines.append("")
    return "\n".join(lines)


def generate_m_scaling(rows: list[dict]) -> str:
    lines = ["## Group C: Objective Scaling\n"]

    groups = defaultdict(list)
    for r in rows:
        task = _safe_str(r.get("task"))
        method = _safe_str(r.get("method"))
        m = _safe_int(r.get("m"))
        noniid = _safe_float(r.get("noniid_strength"))
        use_f16 = r.get("use_float16", False)
        if use_f16 is True or use_f16 == "True" or use_f16 == 1.0:
            continue
        if task != "regression" or noniid > 0:
            continue
        key = (method, m)
        groups[key].append(r)

    table_data = defaultdict(lambda: {"hv": [], "pg": [], "ri": [], "all_dec": [], "upload": []})
    for (method, m), rs in groups.items():
        for r in rs:
            table_data[(method, m)]["hv"].append(_safe_float(r.get("hypervolume")))
            table_data[(method, m)]["pg"].append(_safe_float(r.get("pareto_gap")))
            table_data[(method, m)]["ri"].append(_safe_float(r.get("avg_relative_improvement")))
            table_data[(method, m)]["all_dec"].append(1 if r.get("all_decreased") in (True, "True", 1, 1.0) else 0)
            table_data[(method, m)]["upload"].append(_safe_float(r.get("avg_upload_bytes")))

    all_keys = sorted(table_data.keys(), key=lambda x: (x[1], x[0]))
    lines.append("| Method | m | NHV (mean±std) | NPG (mean±std) | Avg RI (mean±std) | All Dec. | Upload (B) |")
    lines.append("|--------|---|----------------|----------------|-------------------|----------|------------|")
    for method, m in all_keys:
        d = table_data[(method, m)]
        hv_m = _mean(d["hv"])
        hv_s = _std(d["hv"])
        pg_m = _mean(d["pg"])
        pg_s = _std(d["pg"])
        ri_m = _mean(d["ri"])
        ri_s = _std(d["ri"])
        dec_rate = _mean(d["all_dec"])
        up_m = _mean(d["upload"])
        lines.append(f"| {method} | {m} | {hv_m:.4f}±{hv_s:.4f} | {pg_m:.4f}±{pg_s:.4f} | {ri_m:.4f}±{ri_s:.4f} | {dec_rate:.0%} | {up_m:.0f} |")

    lines.append("")
    return "\n".join(lines)


def generate_comm_efficiency(rows: list[dict]) -> str:
    lines = ["## Group D: Communication Efficiency\n"]

    groups = defaultdict(list)
    for r in rows:
        task = _safe_str(r.get("task"))
        method = _safe_str(r.get("method"))
        m = _safe_int(r.get("m"))
        use_f16 = r.get("use_float16", False)
        if use_f16 is True or use_f16 == "True" or use_f16 == 1.0:
            f16 = True
        else:
            f16 = False
        if task != "regression":
            continue
        key = (method, m, f16)
        groups[key].append(r)

    table_data = defaultdict(lambda: {"hv": [], "pg": [], "ri": [], "upload": []})
    for (method, m, f16), rs in groups.items():
        for r in rs:
            table_data[(method, m, f16)]["hv"].append(_safe_float(r.get("hypervolume")))
            table_data[(method, m, f16)]["pg"].append(_safe_float(r.get("pareto_gap")))
            table_data[(method, m, f16)]["ri"].append(_safe_float(r.get("avg_relative_improvement")))
            upload = _safe_float(r.get("upload_per_client"))
            if upload == 0:
                upload = _safe_float(r.get("avg_upload_bytes"))
            table_data[(method, m, f16)]["upload"].append(upload)

    all_keys = sorted(table_data.keys(), key=lambda x: (x[1], x[0], x[2]))
    lines.append("| Method | m | float16 | NHV (mean±std) | NPG (mean±std) | Avg RI (mean±std) | Upload (B) |")
    lines.append("|--------|---|---------|----------------|----------------|-------------------|------------|")
    for method, m, f16 in all_keys:
        d = table_data[(method, m, f16)]
        hv_m = _mean(d["hv"])
        hv_s = _std(d["hv"])
        pg_m = _mean(d["pg"])
        pg_s = _std(d["pg"])
        ri_m = _mean(d["ri"])
        ri_s = _std(d["ri"])
        up_m = _mean(d["upload"])
        lines.append(f"| {method} | {m} | {f16} | {hv_m:.4f}±{hv_s:.4f} | {pg_m:.4f}±{pg_s:.4f} | {ri_m:.4f}±{ri_s:.4f} | {up_m:.0f} |")

    lines.append("")
    return "\n".join(lines)


def generate_plots(rows: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARNING] matplotlib not installed, skipping plots.")
        return

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    _plot_method_comparison(rows, plt, np)
    _plot_noniid_ablation(rows, plt, np)
    _plot_m_scaling(rows, plt, np)
    _plot_comm_efficiency(rows, plt, np)

    print(f"Plots saved to {ANALYSIS_DIR}")


def _plot_method_comparison(rows, plt, np):
    groups = defaultdict(list)
    for r in rows:
        task = _safe_str(r.get("task"))
        m = _safe_int(r.get("m"))
        method = _safe_str(r.get("method"))
        noniid = _safe_float(r.get("noniid_strength"))
        use_f16 = r.get("use_float16", False)
        if use_f16 is True or use_f16 == "True" or use_f16 == 1.0:
            continue
        if noniid > 0:
            continue
        key = (task, m)
        groups[key].append((method, _safe_float(r.get("hypervolume")), _safe_float(r.get("pareto_gap"))))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    method_colors = {"fedjd": "#2196F3", "fmgda": "#4CAF50", "weighted_sum": "#FF9800", "direction_avg": "#9C27B0"}

    for ax_idx, metric_idx in enumerate([1, 2]):
        ax = axes[ax_idx]
        metric_name = "Norm. Hypervolume" if metric_idx == 1 else "Norm. Pareto Gap"
        x_labels = []
        x_pos = []
        pos = 0
        for (task, m) in sorted(groups.keys(), key=lambda x: (x[0], x[1])):
            x_labels.append(f"{task[:3]}\nm={m}")
            entries = groups[(task, m)]
            method_vals = defaultdict(list)
            for method, hv, pg in entries:
                val = hv if metric_idx == 1 else pg
                method_vals[method].append(val)
            for method in ["fedjd", "fmgda", "weighted_sum", "direction_avg"]:
                if method in method_vals:
                    vals = method_vals[method]
                    mean_v = np.mean(vals)
                    std_v = np.std(vals) if len(vals) > 1 else 0
                    ax.bar(pos, mean_v, yerr=std_v, width=0.2,
                           color=method_colors.get(method, "gray"), alpha=0.8,
                           label=method if pos == 0 else "")
                    pos += 0.22
            pos += 0.4

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel(metric_name)
        ax.set_title(f"Method Comparison - {metric_name}")
        ax.grid(True, alpha=0.3, axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "method_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_noniid_ablation(rows, plt, np):
    groups = defaultdict(list)
    for r in rows:
        task = _safe_str(r.get("task"))
        m = _safe_int(r.get("m"))
        method = _safe_str(r.get("method"))
        noniid = _safe_float(r.get("noniid_strength"))
        use_f16 = r.get("use_float16", False)
        if use_f16 is True or use_f16 == "True" or use_f16 == 1.0:
            continue
        if task != "classification" or m != 2:
            continue
        groups[(method, noniid)].append(_safe_float(r.get("hypervolume")))

    fig, ax = plt.subplots(figsize=(8, 5))
    method_colors = {"fedjd": "#2196F3", "fmgda": "#4CAF50", "weighted_sum": "#FF9800"}
    for method in ["fedjd", "fmgda", "weighted_sum"]:
        noniid_vals = sorted(set(k[1] for k in groups if k[0] == method))
        means = [_mean(groups[(method, ni)]) for ni in noniid_vals]
        stds = [_std(groups[(method, ni)]) for ni in noniid_vals]
        ax.errorbar(noniid_vals, means, yerr=stds, marker="o",
                    color=method_colors.get(method, "gray"), label=method, linewidth=2, capsize=4)

    ax.set_xlabel("Non-IID Strength")
    ax.set_ylabel("Normalized Hypervolume")
    ax.set_title("Non-IID Ablation (Classification, m=2)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "noniid_ablation.png", dpi=150)
    plt.close(fig)


def _plot_m_scaling(rows, plt, np):
    groups = defaultdict(list)
    for r in rows:
        task = _safe_str(r.get("task"))
        m = _safe_int(r.get("m"))
        method = _safe_str(r.get("method"))
        noniid = _safe_float(r.get("noniid_strength"))
        use_f16 = r.get("use_float16", False)
        if use_f16 is True or use_f16 == "True" or use_f16 == 1.0:
            continue
        if task != "regression" or noniid > 0:
            continue
        groups[(method, m)].append(_safe_float(r.get("hypervolume")))

    fig, ax = plt.subplots(figsize=(8, 5))
    method_colors = {"fedjd": "#2196F3", "fmgda": "#4CAF50", "weighted_sum": "#FF9800"}
    for method in ["fedjd", "fmgda", "weighted_sum"]:
        m_vals = sorted(set(k[1] for k in groups if k[0] == method))
        means = [_mean(groups[(method, mv)]) for mv in m_vals]
        stds = [_std(groups[(method, mv)]) for mv in m_vals]
        ax.errorbar(m_vals, means, yerr=stds, marker="s",
                    color=method_colors.get(method, "gray"), label=method, linewidth=2, capsize=4)

    ax.set_xlabel("Number of Objectives (m)")
    ax.set_ylabel("Normalized Hypervolume")
    ax.set_title("Objective Scaling (Synthetic Regression)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "m_scaling.png", dpi=150)
    plt.close(fig)


def _plot_comm_efficiency(rows, plt, np):
    groups = defaultdict(list)
    for r in rows:
        task = _safe_str(r.get("task"))
        m = _safe_int(r.get("m"))
        method = _safe_str(r.get("method"))
        use_f16 = r.get("use_float16", False)
        if use_f16 is True or use_f16 == "True" or use_f16 == 1.0:
            f16 = True
        else:
            f16 = False
        if task != "regression":
            continue
        groups[(method, m, f16)].append({
            "hv": _safe_float(r.get("hypervolume")),
            "upload": _safe_float(r.get("avg_upload_bytes")),
        })

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {("fedjd", False): "#2196F3", ("fedjd", True): "#03A9F4"}
    labels_map = {("fedjd", False): "FedJD (fp32)", ("fedjd", True): "FedJD (fp16)"}

    for key in [("fedjd", False), ("fedjd", True)]:
        m_vals = sorted(set(k[1] for k in groups if k[0] == key[0] and k[2] == key[1]))
        hv_means = [_mean([v["hv"] for v in groups[(key[0], mv, key[1])]]) for mv in m_vals]
        up_means = [_mean([v["upload"] for v in groups[(key[0], mv, key[1])]]) for mv in m_vals]
        ax.plot(up_means, hv_means, marker="o", color=colors[key], label=labels_map[key], linewidth=2)
        for i, mv in enumerate(m_vals):
            ax.annotate(f"m={mv}", (up_means[i], hv_means[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Avg Upload Bytes per Round")
    ax.set_ylabel("Normalized Hypervolume")
    ax.set_title("Communication vs Quality Trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "comm_efficiency.png", dpi=150)
    plt.close(fig)


def generate_conclusions(rows: list[dict]) -> str:
    lines = ["## FedJD Applicability Conclusions\n"]

    groups_a = defaultdict(list)
    for r in rows:
        task = _safe_str(r.get("task"))
        m = _safe_int(r.get("m"))
        method = _safe_str(r.get("method"))
        noniid = _safe_float(r.get("noniid_strength"))
        use_f16 = r.get("use_float16", False)
        if use_f16 is True or use_f16 == "True" or use_f16 == 1.0:
            continue
        if noniid > 0:
            continue
        groups_a[(task, m, method)].append(_safe_float(r.get("hypervolume")))

    fedjd_wins = 0
    total_comparisons = 0
    for (task, m, method), hvs in groups_a.items():
        if method == "fedjd":
            fedjd_hv = _mean(hvs)
            for (t2, m2, m2_name), hvs2 in groups_a.items():
                if t2 == task and m2 == m and m2_name != "fedjd":
                    other_hv = _mean(hvs2)
                    total_comparisons += 1
                    if fedjd_hv >= other_hv:
                        fedjd_wins += 1

    lines.append(f"### Overall Performance")
    lines.append(f"- FedJD achieves best or tied-best normalized hypervolume in **{fedjd_wins}/{total_comparisons}** comparisons\n")
    lines.append("- NHV = Normalized Hypervolume: objectives scaled to [0,1], ref=[1.1,...,1.1], divided by 1.1^m")
    lines.append("- NPG = Normalized Pareto Gap: average distance from ideal in normalized space\n")

    groups_b = defaultdict(list)
    for r in rows:
        task = _safe_str(r.get("task"))
        m = _safe_int(r.get("m"))
        method = _safe_str(r.get("method"))
        noniid = _safe_float(r.get("noniid_strength"))
        use_f16 = r.get("use_float16", False)
        if use_f16 is True or use_f16 == "True" or use_f16 == 1.0:
            continue
        if task != "classification" or m != 2:
            continue
        groups_b[(method, noniid)].append(_safe_float(r.get("hypervolume")))

    lines.append("### Non-IID Robustness")
    for method in ["fedjd", "fmgda", "weighted_sum"]:
        hvs_low = groups_b.get((method, 0.0), [])
        hvs_high = groups_b.get((method, 0.9), [])
        if hvs_low and hvs_high:
            drop = (_mean(hvs_low) - _mean(hvs_high)) / max(abs(_mean(hvs_low)), 1e-8) * 100
            lines.append(f"- {method}: HV drop from noniid=0.0 to 0.9 = **{drop:.1f}%**")
    lines.append("")

    groups_c = defaultdict(list)
    for r in rows:
        task = _safe_str(r.get("task"))
        m = _safe_int(r.get("m"))
        method = _safe_str(r.get("method"))
        noniid = _safe_float(r.get("noniid_strength"))
        use_f16 = r.get("use_float16", False)
        if use_f16 is True or use_f16 == "True" or use_f16 == 1.0:
            continue
        if task != "regression" or noniid > 0:
            continue
        groups_c[(method, m)].append(_safe_float(r.get("all_decreased")))

    lines.append("### Objective Scaling Stability")
    for method in ["fedjd", "fmgda", "weighted_sum"]:
        line_parts = [f"- {method}:"]
        for m in [2, 3, 5, 10]:
            dec_rate = _mean(groups_c.get((method, m), []))
            line_parts.append(f" m={m}: {dec_rate:.0%} all-dec")
        lines.append(" ".join(line_parts))
    lines.append("")

    lines.append("### Communication Efficiency")
    lines.append("- float16 compression provides ~50% upload reduction with quality preserved (per Stage 3)")
    lines.append("- FedJD's Jacobian upload cost scales as O(m×d), acceptable when m is moderate (≤10)")
    lines.append("")

    lines.append("### Applicable Scenarios")
    lines.append("- ✅ Multi-objective problems with **conflicting objectives** (m ≥ 2)")
    lines.append("- ✅ Settings where Pareto-optimal trade-offs are needed")
    lines.append("- ✅ Non-IID federated settings with moderate skew")
    lines.append("- ✅ Communication-constrained environments (with float16)")
    lines.append("")
    lines.append("### Not Applicable Scenarios")
    lines.append("- ❌ Single-objective problems (no benefit over FedAvg)")
    lines.append("- ❌ Very high m (>10) where Jacobian upload becomes prohibitive")
    lines.append("- ❌ Extremely low bandwidth where even fp16 Jacobian is too large")
    lines.append("- ❌ Low-conflict objectives where simple weighted sum suffices")

    return "\n".join(lines)


def main():
    csv_path = RESULTS_DIR / "s4_results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run run_s4_benchmark.py first.")
        return

    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} experiment results")

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    report_parts = []
    report_parts.append(generate_main_results_table(rows))
    report_parts.append(generate_noniid_ablation(rows))
    report_parts.append(generate_m_scaling(rows))
    report_parts.append(generate_comm_efficiency(rows))
    report_parts.append(generate_conclusions(rows))

    report = "\n".join(report_parts)
    report_path = ANALYSIS_DIR / "s4_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    generate_plots(rows)
    print("Analysis complete!")


if __name__ == "__main__":
    main()
