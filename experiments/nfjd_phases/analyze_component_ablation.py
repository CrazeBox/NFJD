from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def _to_float(value: str, default: float = 0.0) -> float:
    if value == "" or value is None:
        return default
    return float(value)


def _group_rows(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def _summarize_classification(rows):
    grouped = _group_rows(rows, ["data_split", "method"])
    out = []
    for (split, method), group in grouped.items():
        acc = [_to_float(row["avg_accuracy"]) for row in group]
        f1 = [_to_float(row["avg_f1"]) for row in group]
        min_acc = [_to_float(row["min_task_acc"]) for row in group]
        min_f1 = [_to_float(row["min_task_f1"]) for row in group]
        time_s = [_to_float(row["elapsed_time"]) for row in group]
        out.append({
            "split": split,
            "method": method,
            "acc_mean": mean(acc),
            "acc_std": pstdev(acc) if len(acc) > 1 else 0.0,
            "f1_mean": mean(f1),
            "min_acc_mean": mean(min_acc),
            "min_f1_mean": mean(min_f1),
            "time_mean": mean(time_s),
            "runs": len(group),
        })
    return sorted(out, key=lambda x: (x["split"], -x["acc_mean"]))


def _summarize_regression(rows):
    grouped = _group_rows(rows, ["m", "data_split", "method"])
    out = []
    for (m, split, method), group in grouped.items():
        mse = [_to_float(row["avg_mse"]) for row in group]
        max_mse = [_to_float(row["max_mse"]) for row in group]
        mse_std = [_to_float(row["mse_std"]) for row in group]
        r2 = [_to_float(row["avg_r2"]) for row in group]
        time_s = [_to_float(row["elapsed_time"]) for row in group]
        out.append({
            "m": int(m),
            "split": split,
            "method": method,
            "mse_mean": mean(mse),
            "mse_seed_std": pstdev(mse) if len(mse) > 1 else 0.0,
            "max_mse_mean": mean(max_mse),
            "mse_std_mean": mean(mse_std),
            "r2_mean": mean(r2),
            "time_mean": mean(time_s),
            "runs": len(group),
        })
    return sorted(out, key=lambda x: (x["m"], x["split"], x["mse_mean"]))


def _print_table(rows, columns):
    widths = {col: len(col) for col in columns}
    rendered_rows = []
    for row in rows:
        rendered = {}
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                value = f"{value:.4f}"
            else:
                value = str(value)
            rendered[col] = value
            widths[col] = max(widths[col], len(value))
        rendered_rows.append(rendered)

    print("  ".join(col.ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in rendered_rows:
        print("  ".join(row[col].ljust(widths[col]) for col in columns))


def main():
    parser = argparse.ArgumentParser(description="Summarize Phase 5 component ablation CSV files.")
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    with args.csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("No rows found.")

    dataset = rows[0].get("dataset", "")
    if dataset == "multimnist":
        summary = _summarize_classification(rows)
        _print_table(summary, ["split", "method", "acc_mean", "acc_std", "f1_mean", "min_acc_mean", "min_f1_mean", "time_mean", "runs"])
    elif dataset == "riverflow":
        summary = _summarize_regression(rows)
        _print_table(summary, ["m", "split", "method", "mse_mean", "mse_seed_std", "max_mse_mean", "mse_std_mean", "r2_mean", "time_mean", "runs"])
    else:
        raise SystemExit(f"Unsupported dataset: {dataset}")


if __name__ == "__main__":
    main()
