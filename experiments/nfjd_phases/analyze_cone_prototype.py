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
            row.get("public_preprocess_alpha", ""),
            row.get("public_preprocess_mode", ""),
            row.get("public_preprocess_center_mode", ""),
            row.get("public_preprocess_trim_k", ""),
            row.get("public_preprocess_adaptive_mode", ""),
        )].append(row)

    summary = []
    for (dataset, method, alpha, reference_mode, basis_size, public_alpha, public_mode, public_center_mode, public_trim_k, public_adaptive_mode), group in grouped.items():
        mse = [_to_float(r["avg_mse"]) for r in group]
        max_mse = [_to_float(r["max_mse"]) for r in group]
        mse_std = [_to_float(r["mse_std"]) for r in group]
        r2 = [_to_float(r["avg_r2"]) for r in group]
        time_s = [_to_float(r["elapsed_time"]) for r in group]
        summary.append({
            "dataset": dataset,
            "method": method,
            "alpha": alpha,
            "reference_mode": reference_mode,
            "basis_size": basis_size,
            "public_alpha": public_alpha,
            "public_mode": public_mode,
            "public_center_mode": public_center_mode,
            "public_trim_k": public_trim_k,
            "public_adaptive_mode": public_adaptive_mode,
            "avg_mse_mean": mean(mse),
            "avg_mse_std": pstdev(mse) if len(mse) > 1 else 0.0,
            "max_mse_mean": mean(max_mse),
            "mse_std_mean": mean(mse_std),
            "avg_r2_mean": mean(r2),
            "time_mean": mean(time_s),
            "runs": len(group),
        })

    columns = ["dataset", "method", "alpha", "reference_mode", "basis_size", "public_alpha", "public_mode", "public_center_mode", "public_trim_k", "public_adaptive_mode", "avg_mse_mean", "avg_mse_std", "max_mse_mean", "mse_std_mean", "avg_r2_mean", "time_mean", "runs"]
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
