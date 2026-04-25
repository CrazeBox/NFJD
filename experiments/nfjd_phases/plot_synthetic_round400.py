from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: str):
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def plot_mse_ranking(rows, output_dir: Path):
    methods = [row["method"] for row in rows]
    vals = [float(row["avg_mse"]) for row in rows]
    order = sorted(range(len(rows)), key=lambda i: vals[i])
    methods = [methods[i] for i in order]
    vals = [vals[i] for i in order]

    plt.figure(figsize=(12, 7))
    plt.barh(methods, vals, color="#4C78A8")
    plt.gca().invert_yaxis()
    plt.xlabel("Average MSE")
    plt.title("Synthetic Regression 400 Rounds (seed=42): MSE Ranking")
    plt.tight_layout()
    out = output_dir / "round400_seed42_mse_ranking.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_time_vs_mse(rows, output_dir: Path):
    plt.figure(figsize=(10, 7))
    for row in rows:
        x = float(row["elapsed_time"])
        y = float(row["avg_mse"])
        label = row["method"]
        plt.scatter(x, y, s=70)
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Average MSE")
    plt.title("Synthetic Regression 400 Rounds (seed=42): MSE vs Time")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = output_dir / "round400_seed42_time_vs_mse.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_r2_ranking(rows, output_dir: Path):
    methods = [row["method"] for row in rows]
    vals = [float(row["avg_r2"]) for row in rows]
    order = sorted(range(len(rows)), key=lambda i: vals[i], reverse=True)
    methods = [methods[i] for i in order]
    vals = [vals[i] for i in order]

    plt.figure(figsize=(12, 7))
    plt.barh(methods, vals, color="#59A14F")
    plt.gca().invert_yaxis()
    plt.xlabel("Average R2")
    plt.title("Synthetic Regression 400 Rounds (seed=42): R2 Ranking")
    plt.tight_layout()
    out = output_dir / "round400_seed42_r2_ranking.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_metric_bars(rows, output_dir: Path):
    excluded = {
        "exp_id", "method", "dataset", "seed",
        "cone_reference_mode", "cone_align_positive_only", "public_preprocess_mode",
        "public_preprocess_positive_only", "public_preprocess_center_mode",
        "public_preprocess_adaptive_mode",
    }
    generated = []
    sample = rows[0]
    methods = [row["method"] for row in rows]
    for key in sample.keys():
        if key in excluded:
            continue
        values = [_to_float(row.get(key, "")) for row in rows]
        if all(v is None for v in values):
            continue
        if key.startswith("init_obj_") or key.startswith("final_obj_") or key.startswith("delta_obj_"):
            # keep these too, but skip empty trailing objective slots
            if all(v is None for v in values):
                continue
        cleaned = [float("nan") if v is None else v for v in values]
        if np.all(np.isnan(cleaned)):
            continue
        order = sorted(range(len(rows)), key=lambda i: cleaned[i] if not np.isnan(cleaned[i]) else float("inf"), reverse=True)
        ordered_methods = [methods[i] for i in order]
        ordered_values = [cleaned[i] for i in order]

        plt.figure(figsize=(12, 7))
        bars = plt.barh(ordered_methods, ordered_values, color="#4C78A8")
        plt.gca().invert_yaxis()
        plt.xlabel(key)
        plt.title(f"Synthetic Regression 400 Rounds (seed=42): {key}")
        plt.tight_layout()
        out = output_dir / f"metric_{key}.png"
        plt.savefig(out, dpi=180)
        plt.close()
        generated.append(out)
    return generated


def plot_metric_heatmap(rows, output_dir: Path):
    metrics = [
        "avg_mse",
        "max_mse",
        "mse_std",
        "avg_r2",
        "elapsed_time",
        "avg_round_time",
        "avg_upload_bytes",
    ]
    methods = [row["method"] for row in rows]
    matrix = []
    for metric in metrics:
        vals = np.array([_to_float(row.get(metric, "")) for row in rows], dtype=float)
        if np.isnan(vals).all():
            continue
        if metric in {"elapsed_time", "avg_round_time", "avg_upload_bytes", "avg_mse", "max_mse", "mse_std"}:
            vals = -vals
        min_v = np.nanmin(vals)
        max_v = np.nanmax(vals)
        if abs(max_v - min_v) < 1e-12:
            norm = np.zeros_like(vals)
        else:
            norm = (vals - min_v) / (max_v - min_v)
        matrix.append(norm)
    matrix = np.array(matrix)
    labels = [m for m in metrics if not np.isnan(np.array([_to_float(row.get(m, "")) for row in rows], dtype=float)).all()]

    plt.figure(figsize=(16, max(6, 0.45 * len(labels))))
    im = plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Normalized better-is-higher score")
    plt.yticks(range(len(labels)), labels)
    plt.xticks(range(len(methods)), methods, rotation=45, ha="right")
    plt.title("Synthetic Regression 400 Rounds (seed=42): Metric Heatmap")
    plt.tight_layout()
    out = output_dir / "round400_seed42_metric_heatmap.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot charts for 400-round synthetic benchmark.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/nfjd_cone_prototype/charts_round400_seed42"))
    args = parser.parse_args()

    rows = load_rows(args.csv_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        plot_mse_ranking(rows, args.output_dir),
        plot_r2_ranking(rows, args.output_dir),
        plot_time_vs_mse(rows, args.output_dir),
        plot_metric_heatmap(rows, args.output_dir),
    ]
    outputs.extend(plot_metric_bars(rows, args.output_dir))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
