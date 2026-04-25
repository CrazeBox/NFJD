from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def _to_float(value: str) -> float:
    if value == "" or value is None:
        return 0.0
    return float(value)


def main():
    parser = argparse.ArgumentParser(description="Summarize cone prototype benchmark CSV.")
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    with args.csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("No rows found.")

    grouped = defaultdict(list)
    for row in rows:
        grouped[(
            row["dataset"],
            row["method"],
            row.get("cone_align_alpha", ""),
            row.get("cone_reference_mode", ""),
            row.get("cone_basis_size", ""),
        )].append(row)

    summary = []
    for (dataset, method, alpha, reference_mode, basis_size), group in grouped.items():
        ri = [_to_float(r["avg_ri"]) for r in group]
        jfi = [_to_float(r["task_jfi"]) for r in group]
        time_s = [_to_float(r["elapsed_time"]) for r in group]
        cone_margin = [_to_float(r.get("avg_cone_margin", "0")) for r in group]
        cone_cosine = [_to_float(r.get("avg_cone_cosine", "0")) for r in group]
        summary.append({
            "dataset": dataset,
            "method": method,
            "alpha": alpha,
            "reference_mode": reference_mode,
            "basis_size": basis_size,
            "ri_mean": mean(ri),
            "ri_std": pstdev(ri) if len(ri) > 1 else 0.0,
            "jfi_mean": mean(jfi),
            "time_mean": mean(time_s),
            "cone_margin_mean": mean(cone_margin),
            "cone_cosine_mean": mean(cone_cosine),
            "runs": len(group),
        })

    columns = ["dataset", "method", "alpha", "reference_mode", "basis_size", "ri_mean", "ri_std", "jfi_mean", "time_mean", "cone_margin_mean", "cone_cosine_mean", "runs"]
    widths = {col: len(col) for col in columns}
    formatted = []
    for row in summary:
        rendered = {}
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                text = f"{value:.4f}"
            else:
                text = str(value)
            widths[col] = max(widths[col], len(text))
            rendered[col] = text
        formatted.append(rendered)

    print("  ".join(col.ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in sorted(formatted, key=lambda r: (r["dataset"], r["method"], r["alpha"])):
        print("  ".join(row[col].ljust(widths[col]) for col in columns))


if __name__ == "__main__":
    main()
