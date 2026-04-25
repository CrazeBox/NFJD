from __future__ import annotations

import argparse
import ast
import csv
import re
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


def parse_log(log_path: Path):
    run_re = re.compile(r"Running (\{.*\})")
    init_re = re.compile(r"Initial objectives: (.*)")
    round_re = re.compile(r"Round (\d+) \| .*? obj=\[(.*?)\]")
    completed_re = re.compile(r"\[(.*?)\] (\w+) RI=")

    current = None
    runs = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            run_match = run_re.search(line)
            if run_match:
                payload = ast.literal_eval(run_match.group(1))
                current = {
                    "method": payload["method"],
                    "seed": payload["seed"],
                    "dataset": payload["dataset"],
                    "num_rounds": payload["num_rounds"],
                    "initial": None,
                    "rounds": [],
                }
                runs.append(current)
                continue

            if current is None:
                continue

            init_match = init_re.search(line)
            if init_match and current["initial"] is None:
                current["initial"] = [float(x.strip()) for x in init_match.group(1).split(",")]
                continue

            round_match = round_re.search(line)
            if round_match:
                round_idx = int(round_match.group(1))
                values = [float(x.strip()) for x in round_match.group(2).split(",")]
                current["rounds"].append((round_idx, values))
                continue

            if completed_re.search(line):
                current = None

    return runs


def avg_ri(initial, current):
    vals = []
    for init, cur in zip(initial, current):
        denom = abs(init) if abs(init) > 1e-8 else 1e-8
        vals.append((init - cur) / denom)
    return sum(vals) / len(vals)


def plot_ri_ranking(rows, output_dir: Path):
    methods = [row["method"] for row in rows]
    ris = [float(row["avg_ri"]) for row in rows]
    order = sorted(range(len(rows)), key=lambda i: ris[i], reverse=True)
    methods = [methods[i] for i in order]
    ris = [ris[i] for i in order]

    plt.figure(figsize=(12, 7))
    plt.barh(methods, ris, color="#4C78A8")
    plt.gca().invert_yaxis()
    plt.xlabel("Average Relative Improvement (RI)")
    plt.title("Synthetic Regression 400 Rounds (seed=42): RI Ranking")
    plt.tight_layout()
    out = output_dir / "round400_seed42_ri_ranking.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_time_vs_ri(rows, output_dir: Path):
    plt.figure(figsize=(10, 7))
    for row in rows:
        x = float(row["elapsed_time"])
        y = float(row["avg_ri"])
        label = row["method"]
        plt.scatter(x, y, s=70)
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Average Relative Improvement (RI)")
    plt.title("Synthetic Regression 400 Rounds (seed=42): RI vs Time")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = output_dir / "round400_seed42_time_vs_ri.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_round_ri_trends(runs, output_dir: Path):
    selected = [
        "fedavg_ls",
        "fedavg_pcgrad",
        "fedavg_cagrad",
        "nfjd",
        "nfjd_fast",
        "nfjd_cached",
        "nfjd_common_safe",
        "nfjd_cone_basis",
    ]
    color_map = {
        "fedavg_ls": "#1f77b4",
        "fedavg_pcgrad": "#ff7f0e",
        "fedavg_cagrad": "#2ca02c",
        "nfjd": "#d62728",
        "nfjd_fast": "#9467bd",
        "nfjd_cached": "#8c564b",
        "nfjd_common_safe": "#e377c2",
        "nfjd_cone_basis": "#7f7f7f",
    }
    plt.figure(figsize=(12, 7))
    for run in runs:
        if run["dataset"] != "synthetic_regression" or run["seed"] != 42 or run["method"] not in selected:
            continue
        initial = run["initial"]
        if not initial:
            continue
        xs = []
        ys = []
        for round_idx, values in run["rounds"]:
            xs.append(round_idx)
            ys.append(avg_ri(initial, values))
        plt.plot(xs, ys, label=run["method"], linewidth=2, color=color_map.get(run["method"]))
    plt.xlabel("Round")
    plt.ylabel("Average Relative Improvement (RI)")
    plt.title("Synthetic Regression 400 Rounds (seed=42): RI Trajectories")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = output_dir / "round400_seed42_ri_trends.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_objective_endpoints(rows, output_dir: Path):
    selected = ["fedavg_ls", "fedavg_pcgrad", "fedavg_cagrad", "nfjd", "nfjd_fast", "nfjd_cached", "nfjd_common_safe", "nfjd_cone_basis"]
    rows = [row for row in rows if row["method"] in selected]
    rows.sort(key=lambda r: selected.index(r["method"]))
    methods = [row["method"] for row in rows]
    obj0 = [float(row["final_obj_0"]) for row in rows]
    obj1 = [float(row["final_obj_1"]) for row in rows]
    obj2 = [float(row["final_obj_2"]) for row in rows]
    x = range(len(methods))
    width = 0.25
    plt.figure(figsize=(13, 7))
    plt.bar([i - width for i in x], obj0, width=width, label="final_obj_0")
    plt.bar(list(x), obj1, width=width, label="final_obj_1")
    plt.bar([i + width for i in x], obj2, width=width, label="final_obj_2")
    plt.xticks(list(x), methods, rotation=30, ha="right")
    plt.ylabel("Final Objective Value")
    plt.title("Synthetic Regression 400 Rounds (seed=42): Final Objectives")
    plt.legend()
    plt.tight_layout()
    out = output_dir / "round400_seed42_final_objectives.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_metric_bars(rows, output_dir: Path):
    excluded = {
        "exp_id", "method", "dataset", "seed", "all_decreased",
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
        "avg_ri",
        "task_jfi",
        "task_mmag",
        "hypervolume",
        "pareto_gap",
        "elapsed_time",
        "avg_round_time",
        "avg_upload_bytes",
        "avg_rescale_factor",
        "avg_cosine_sim",
        "avg_prox_ratio",
        "avg_preprocess_alpha",
        "avg_cone_margin",
        "avg_cone_cosine",
        "delta_obj_0",
        "delta_obj_1",
        "delta_obj_2",
    ]
    methods = [row["method"] for row in rows]
    matrix = []
    for metric in metrics:
        vals = np.array([_to_float(row.get(metric, "")) for row in rows], dtype=float)
        if np.isnan(vals).all():
            continue
        if metric in {"elapsed_time", "avg_round_time", "avg_upload_bytes", "pareto_gap", "task_mmag", "final_obj_0", "final_obj_1", "final_obj_2"}:
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
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/nfjd_cone_prototype/charts_round400_seed42"))
    args = parser.parse_args()

    rows = load_rows(args.csv_path)
    runs = parse_log(args.log_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        plot_ri_ranking(rows, args.output_dir),
        plot_time_vs_ri(rows, args.output_dir),
        plot_round_ri_trends(runs, args.output_dir),
        plot_objective_endpoints(rows, args.output_dir),
        plot_metric_heatmap(rows, args.output_dir),
    ]
    outputs.extend(plot_metric_bars(rows, args.output_dir))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
