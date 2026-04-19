from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("results/nfjd_benchmark")


def _sf(v, default=0.0):
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


def _si(v, default=0):
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def _mean(vals):
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def _std(vals):
    vals = [v for v in vals if not math.isnan(v)]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def main():
    rows = []
    with open(RESULTS_DIR / "nfjd_results.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    lines = ["# NFJD Benchmark Results\n"]

    # Group A: Main comparison on synthetic data
    lines.append("## Group A: Method Comparison (Synthetic Regression, low conflict)\n")
    groups = defaultdict(list)
    for r in rows:
        m = _si(r.get("m"))
        method = r.get("method", "")
        cs = _sf(r.get("conflict_strength"))
        if cs != 0.0:
            continue
        groups[(m, method)].append(r)

    lines.append("| m | Method | NHV (mean±std) | Avg RI (mean±std) | Upload/Client | Time (s) | Rescale |")
    lines.append("|---|--------|----------------|-------------------|---------------|----------|---------|")
    for key in sorted(groups.keys()):
        m, method = key
        rs = groups[key]
        hv_vals = [_sf(r.get("hypervolume")) for r in rs]
        ri_vals = [_sf(r.get("avg_relative_improvement")) for r in rs]
        up_vals = [_sf(r.get("upload_per_client")) for r in rs]
        time_vals = [_sf(r.get("elapsed_time")) for r in rs]
        rescale_vals = [_sf(r.get("avg_rescale_factor")) for r in rs]
        lines.append(f"| {m} | {method} | {_mean(hv_vals):.4f}±{_std(hv_vals):.4f} | "
                     f"{_mean(ri_vals):.4f}±{_std(ri_vals):.4f} | {_mean(up_vals):.0f} | "
                     f"{_mean(time_vals):.1f} | {_mean(rescale_vals):.2f} |")

    # Group B: High conflict comparison
    lines.append("\n## Group B: Method Comparison (High Conflict Regression, cs=1.0)\n")
    hc_groups = defaultdict(list)
    for r in rows:
        m = _si(r.get("m"))
        method = r.get("method", "")
        cs = _sf(r.get("conflict_strength"))
        ar = r.get("use_adaptive_rescaling", "True")
        sg = r.get("use_stochastic_gramian", "True")
        if cs != 1.0 or ar != "True" or sg != "True":
            continue
        hc_groups[(m, method)].append(r)

    lines.append("| m | Method | NHV (mean±std) | Avg RI (mean±std) | Upload/Client | Time (s) | Rescale |")
    lines.append("|---|--------|----------------|-------------------|---------------|----------|---------|")
    for key in sorted(hc_groups.keys()):
        m, method = key
        rs = hc_groups[key]
        hv_vals = [_sf(r.get("hypervolume")) for r in rs]
        ri_vals = [_sf(r.get("avg_relative_improvement")) for r in rs]
        up_vals = [_sf(r.get("upload_per_client")) for r in rs]
        time_vals = [_sf(r.get("elapsed_time")) for r in rs]
        rescale_vals = [_sf(r.get("avg_rescale_factor")) for r in rs]
        lines.append(f"| {m} | {method} | {_mean(hv_vals):.4f}±{_std(hv_vals):.4f} | "
                     f"{_mean(ri_vals):.4f}±{_std(ri_vals):.4f} | {_mean(up_vals):.0f} | "
                     f"{_mean(time_vals):.1f} | {_mean(rescale_vals):.2f} |")

    # Group C: Ablation - AdaptiveRescaling
    lines.append("\n## Group C: Ablation - AdaptiveRescaling (High Conflict, m=2,5)\n")
    ablation_groups = defaultdict(list)
    for r in rows:
        method = r.get("method", "")
        cs = _sf(r.get("conflict_strength"))
        m = _si(r.get("m"))
        ar = r.get("use_adaptive_rescaling", "True")
        if method != "nfjd" or cs != 1.0 or m not in (2, 5):
            continue
        label = "NFJD+AR" if ar == "True" else "NFJD (no AR)"
        ablation_groups[(m, label)].append(r)

    lines.append("| m | Config | NHV (mean±std) | Avg RI (mean±std) | Rescale |")
    lines.append("|---|--------|----------------|-------------------|---------|")
    for key in sorted(ablation_groups.keys()):
        m, label = key
        rs = ablation_groups[key]
        hv_vals = [_sf(r.get("hypervolume")) for r in rs]
        ri_vals = [_sf(r.get("avg_relative_improvement")) for r in rs]
        rescale_vals = [_sf(r.get("avg_rescale_factor")) for r in rs]
        lines.append(f"| {m} | {label} | {_mean(hv_vals):.4f}±{_std(hv_vals):.4f} | "
                     f"{_mean(ri_vals):.4f}±{_std(ri_vals):.4f} | {_mean(rescale_vals):.2f} |")

    # Key findings
    lines.append("\n## Key Findings\n")

    nfjd_ri_syn = []
    fedjd_ri_syn = []
    for r in rows:
        if _sf(r.get("conflict_strength")) == 0.0 and r.get("method") == "nfjd":
            nfjd_ri_syn.append(_sf(r.get("avg_relative_improvement")))
        elif _sf(r.get("conflict_strength")) == 0.0 and r.get("method") == "fedjd":
            fedjd_ri_syn.append(_sf(r.get("avg_relative_improvement")))

    nfjd_ri_hc = []
    fedjd_ri_hc = []
    for r in rows:
        if _sf(r.get("conflict_strength")) == 1.0 and r.get("method") == "nfjd" and r.get("use_adaptive_rescaling") == "True":
            nfjd_ri_hc.append(_sf(r.get("avg_relative_improvement")))
        elif _sf(r.get("conflict_strength")) == 1.0 and r.get("method") == "fedjd":
            fedjd_ri_hc.append(_sf(r.get("avg_relative_improvement")))

    lines.append(f"### NFJD vs FedJD\n")
    if nfjd_ri_syn and fedjd_ri_syn:
        lines.append(f"- **Synthetic (low conflict)**: NFJD Avg RI = {_mean(nfjd_ri_syn):.4f} vs FedJD = {_mean(fedjd_ri_syn):.4f} "
                     f"(**{((_mean(nfjd_ri_syn) - _mean(fedjd_ri_syn)) / _mean(fedjd_ri_syn) * 100):+.1f}%**)")
    if nfjd_ri_hc and fedjd_ri_hc:
        lines.append(f"- **High Conflict**: NFJD Avg RI = {_mean(nfjd_ri_hc):.4f} vs FedJD = {_mean(fedjd_ri_hc):.4f} "
                     f"(**{((_mean(nfjd_ri_hc) - _mean(fedjd_ri_hc)) / _mean(fedjd_ri_hc) * 100):+.1f}%**)")

    nfjd_upload = []
    fedjd_upload = []
    for r in rows:
        if _sf(r.get("conflict_strength")) == 1.0 and r.get("method") == "nfjd" and _si(r.get("m")) == 5:
            nfjd_upload.append(_sf(r.get("upload_per_client")))
        elif _sf(r.get("conflict_strength")) == 1.0 and r.get("method") == "fedjd" and _si(r.get("m")) == 5:
            fedjd_upload.append(_sf(r.get("upload_per_client")))
    if nfjd_upload and fedjd_upload:
        lines.append(f"- **Communication (m=5)**: NFJD = {_mean(nfjd_upload):.0f} B vs FedJD = {_mean(fedjd_upload):.0f} B "
                     f"(**{(_mean(nfjd_upload) / _mean(fedjd_upload) * 100):.0f}% of FedJD**)")

    lines.append(f"\n### Architecture Improvements\n")
    lines.append(f"- **AdaptiveRescaling**: Average scale factor = {_mean([_sf(r.get('avg_rescale_factor')) for r in rows if r.get('method') == 'nfjd' and _sf(r.get('conflict_strength')) == 1.0]):.2f}")
    lines.append(f"- **Δθ Upload**: Communication completely decoupled from m (upload = d×4 B always)")
    lines.append(f"- **Dual Momentum**: Local β=0.9 + Global β=0.9 smooth training trajectory")

    report_dir = RESULTS_DIR / "analysis"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "nfjd_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {report_dir / 'nfjd_report.md'}")


if __name__ == "__main__":
    main()
