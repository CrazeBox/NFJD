from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

RESULTS_DIR = Path("results/s5_highconflict")


def _safe_float(v, default=0.0):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    if isinstance(v, str):
        if v.strip() == "":
            return default
        try:
            return float(v)
        except ValueError:
            return default
    return float(v)


def _safe_int(v, default=0):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


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


def load_results():
    csv_path = RESULTS_DIR / "s5_results.csv"
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def generate_main_results(rows):
    lines = ["# Stage 5: High-Conflict Dataset Results\n"]

    lines.append("## Group A: Method Comparison on High-Conflict Data (conflict_strength=1.0, 100 rounds)\n")
    groups = defaultdict(list)
    for r in rows:
        m = _safe_int(r.get("m"))
        method = _safe_str(r.get("method"))
        cs = _safe_float(r.get("conflict_strength"))
        num_rounds = _safe_int(r.get("num_rounds"))
        model_size = _safe_str(r.get("model_size"))
        if cs != 1.0 or num_rounds != 100 or model_size != "small":
            continue
        key = (m, method)
        groups[key].append(r)

    table_data = defaultdict(lambda: {"hv": [], "pg": [], "ri": [], "cos": [], "all_dec": [], "upload": [], "time": []})
    for (m, method), rs in groups.items():
        for r in rs:
            table_data[(m, method)]["hv"].append(_safe_float(r.get("hypervolume")))
            table_data[(m, method)]["pg"].append(_safe_float(r.get("pareto_gap")))
            table_data[(m, method)]["ri"].append(_safe_float(r.get("avg_relative_improvement")))
            table_data[(m, method)]["cos"].append(_safe_float(r.get("gradient_conflict_cos")))
            table_data[(m, method)]["all_dec"].append(1 if r.get("all_decreased") in (True, "True", 1, 1.0) else 0)
            table_data[(m, method)]["upload"].append(_safe_float(r.get("upload_per_client")))
            table_data[(m, method)]["time"].append(_safe_float(r.get("elapsed_time")))

    all_keys = sorted(table_data.keys(), key=lambda x: (x[0], x[1]))
    lines.append("| m | Method | NHV (mean±std) | NPG (mean±std) | Avg RI (mean±std) | Avg Cos | All Dec. | Upload/Client | Time (s) |")
    lines.append("|---|--------|----------------|----------------|-------------------|---------|----------|---------------|----------|")
    for m, method in all_keys:
        d = table_data[(m, method)]
        hv_m = _mean(d["hv"])
        hv_s = _std(d["hv"])
        pg_m = _mean(d["pg"])
        pg_s = _std(d["pg"])
        ri_m = _mean(d["ri"])
        ri_s = _std(d["ri"])
        cos_m = _mean(d["cos"])
        dec_rate = _mean(d["all_dec"])
        up_m = _mean(d["upload"])
        t_m = _mean(d["time"])
        lines.append(f"| {m} | {method} | {hv_m:.4f}±{hv_s:.4f} | {pg_m:.4f}±{pg_s:.4f} | {ri_m:.4f}±{ri_s:.4f} | {cos_m:.4f} | {dec_rate:.0%} | {up_m:.0f} | {t_m:.2f} |")

    lines.append("")

    lines.append("## Group B: Conflict Strength Ablation (m=2, 100 rounds)\n")
    cs_groups = defaultdict(list)
    for r in rows:
        m = _safe_int(r.get("m"))
        method = _safe_str(r.get("method"))
        cs = _safe_float(r.get("conflict_strength"))
        num_rounds = _safe_int(r.get("num_rounds"))
        model_size = _safe_str(r.get("model_size"))
        if m != 2 or num_rounds != 100 or model_size != "small":
            continue
        key = (method, cs)
        cs_groups[key].append(r)

    cs_table = defaultdict(lambda: {"hv": [], "ri": [], "cos": []})
    for (method, cs), rs in cs_groups.items():
        for r in rs:
            cs_table[(method, cs)]["hv"].append(_safe_float(r.get("hypervolume")))
            cs_table[(method, cs)]["ri"].append(_safe_float(r.get("avg_relative_improvement")))
            cs_table[(method, cs)]["cos"].append(_safe_float(r.get("gradient_conflict_cos")))

    all_cs_keys = sorted(cs_table.keys(), key=lambda x: (x[1], x[0]))
    lines.append("| Method | Conflict | NHV (mean±std) | Avg RI (mean±std) | Avg Cos |")
    lines.append("|--------|----------|----------------|-------------------|---------|")
    for method, cs in all_cs_keys:
        d = cs_table[(method, cs)]
        hv_m = _mean(d["hv"])
        hv_s = _std(d["hv"])
        ri_m = _mean(d["ri"])
        ri_s = _std(d["ri"])
        cos_m = _mean(d["cos"])
        lines.append(f"| {method} | {cs:.1f} | {hv_m:.4f}±{hv_s:.4f} | {ri_m:.4f}±{ri_s:.4f} | {cos_m:.4f} |")

    lines.append("")

    lines.append("## Group C: Extended Training (200 rounds, medium model)\n")
    ext_groups = defaultdict(list)
    for r in rows:
        m = _safe_int(r.get("m"))
        method = _safe_str(r.get("method"))
        num_rounds = _safe_int(r.get("num_rounds"))
        model_size = _safe_str(r.get("model_size"))
        if num_rounds != 200 or model_size != "medium":
            continue
        key = (m, method)
        ext_groups[key].append(r)

    ext_table = defaultdict(lambda: {"hv": [], "ri": [], "upload": []})
    for (m, method), rs in ext_groups.items():
        for r in rs:
            ext_table[(m, method)]["hv"].append(_safe_float(r.get("hypervolume")))
            ext_table[(m, method)]["ri"].append(_safe_float(r.get("avg_relative_improvement")))
            ext_table[(m, method)]["upload"].append(_safe_float(r.get("upload_per_client")))

    all_ext_keys = sorted(ext_table.keys(), key=lambda x: (x[0], x[1]))
    lines.append("| m | Method | NHV (mean±std) | Avg RI (mean±std) | Upload/Client |")
    lines.append("|---|--------|----------------|-------------------|---------------|")
    for m, method in all_ext_keys:
        d = ext_table[(m, method)]
        hv_m = _mean(d["hv"])
        hv_s = _std(d["hv"])
        ri_m = _mean(d["ri"])
        ri_s = _std(d["ri"])
        up_m = _mean(d["upload"])
        lines.append(f"| {m} | {method} | {hv_m:.4f}±{hv_s:.4f} | {ri_m:.4f}±{ri_s:.4f} | {up_m:.0f} |")

    lines.append("")

    lines.append("## Key Findings\n")
    lines.append("- NHV = Normalized Hypervolume (objectives scaled to [0,1], ref=[1.1,...,1.1], divided by 1.1^m)")
    lines.append("- NPG = Normalized Pareto Gap (average distance from ideal in normalized space)")
    lines.append("- Avg RI = Average Relative Improvement across objectives")
    lines.append("- Avg Cos = Average pairwise cosine similarity of objective gradients (negative = high conflict)")

    fedjd_wins = 0
    total_comp = 0
    for m_val in [2, 3, 5]:
        best_ri = -float("inf")
        best_method = ""
        for method in METHODS:
            key = (m_val, method)
            if key in table_data:
                ri = _mean(table_data[key]["ri"])
                if ri > best_ri:
                    best_ri = ri
                    best_method = method
        if best_method == "fedjd":
            fedjd_wins += 1
        total_comp += 1

    lines.append(f"\n### FedJD Performance on High-Conflict Data")
    lines.append(f"- FedJD achieves best Avg RI in **{fedjd_wins}/{total_comp}** m-value comparisons")
    lines.append(f"- Gradient conflict (Avg Cos): more negative values indicate stronger objective conflict")
    lines.append(f"- On high-conflict data, MinNorm direction finding should provide better Pareto trade-offs")

    return "\n".join(lines)


METHODS = ["fedjd", "fmgda", "weighted_sum", "direction_avg"]


def main():
    rows = load_results()
    if not rows:
        print("No results found!")
        return

    report = generate_main_results(rows)

    report_dir = RESULTS_DIR / "analysis"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "s5_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved to {report_path}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
