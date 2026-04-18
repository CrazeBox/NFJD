from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def load_sweep_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            typed = {}
            for k, v in row.items():
                try:
                    typed[k] = float(v)
                except (ValueError, TypeError):
                    typed[k] = v
            rows.append(typed)
    return rows


def load_metrics(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            typed = {}
            for k, v in row.items():
                try:
                    typed[k] = float(v)
                except (ValueError, TypeError):
                    typed[k] = v
            rows.append(typed)
    return rows


def check_D_optimization_trend(sweep_dir: Path):
    print("=" * 60)
    print("D. 优化趋势")
    print("=" * 60)

    summaries = load_sweep_csv(sweep_dir / "sweep_results.csv")

    # D1: At least one config with objective decrease
    all_decrease = [s for s in summaries if s.get("delta_obj_0", 0) < 0]
    d1_pass = len(all_decrease) > 0
    print(f"  D1 Configs with obj0 decrease: {len(all_decrease)}/{len(summaries)}: {'PASS' if d1_pass else 'FAIL'}")

    # D2: Both objectives decrease
    both_decrease = [s for s in summaries if s.get("delta_obj_0", 0) < 0 and s.get("delta_obj_1", 0) < 0]
    d2_pass = len(both_decrease) > 0
    print(f"  D2 Configs with both objs decrease: {len(both_decrease)}/{len(summaries)}: {'PASS' if d2_pass else 'FAIL'}")

    # D3: Training curves no divergence - check if any objective increases significantly
    diverged = 0
    total_exp = 0
    for exp_dir in sorted(sweep_dir.iterdir()):
        if not exp_dir.is_dir() or not (exp_dir / "metrics.csv").exists():
            continue
        total_exp += 1
        metrics = load_metrics(exp_dir / "metrics.csv")
        for m in metrics:
            if m.get("objective_0", 0) > 1e6 or m.get("objective_1", 0) > 1e6:
                diverged += 1
                break
    d3_pass = diverged == 0
    print(f"  D3 Diverged experiments: {diverged}/{total_exp}: {'PASS' if d3_pass else 'FAIL'}")

    # D4: Jacobian norm not exploding
    max_jac_norm = 0
    for exp_dir in sorted(sweep_dir.iterdir()):
        if not exp_dir.is_dir() or not (exp_dir / "metrics.csv").exists():
            continue
        metrics = load_metrics(exp_dir / "metrics.csv")
        for m in metrics:
            jn = m.get("jacobian_norm", 0)
            if jn > max_jac_norm:
                max_jac_norm = jn
    d4_pass = max_jac_norm < 1000
    print(f"  D4 Max Jacobian norm across all rounds: {max_jac_norm:.4f} (< 1000): {'PASS' if d4_pass else 'FAIL'}")

    # D5: Direction norm not exploding
    max_dir_norm = 0
    for exp_dir in sorted(sweep_dir.iterdir()):
        if not exp_dir.is_dir() or not (exp_dir / "metrics.csv").exists():
            continue
        metrics = load_metrics(exp_dir / "metrics.csv")
        for m in metrics:
            dn = m.get("direction_norm", 0)
            if dn > max_dir_norm:
                max_dir_norm = dn
    d5_pass = max_dir_norm < 1000
    print(f"  D5 Max direction norm across all rounds: {max_dir_norm:.4f} (< 1000): {'PASS' if d5_pass else 'FAIL'}")

    return {"D1": d1_pass, "D2": d2_pass, "D3": d3_pass, "D4": d4_pass, "D5": d5_pass}


def check_E_baseline_comparison(sweep_dir: Path):
    print("\n" + "=" * 60)
    print("E. 基线对比")
    print("=" * 60)

    summaries = load_sweep_csv(sweep_dir / "sweep_results.csv")

    # Per-config comparison: MinNorm vs Random
    minnorm = [s for s in summaries if s.get("aggregator") == "minnorm"]
    random_agg = [s for s in summaries if s.get("aggregator") == "random"]
    mean_agg = [s for s in summaries if s.get("aggregator") == "mean"]

    # E1: MinNorm vs Random - per config pair
    mn_better_count = 0
    total_pairs = 0
    for mn in minnorm:
        for rd in random_agg:
            if (mn.get("num_clients") == rd.get("num_clients") and
                mn.get("participation_rate") == rd.get("participation_rate") and
                mn.get("seed") == rd.get("seed")):
                total_pairs += 1
                mn_avg = (mn.get("delta_obj_0", 0) + mn.get("delta_obj_1", 0)) / 2
                rd_avg = (rd.get("delta_obj_0", 0) + rd.get("delta_obj_1", 0)) / 2
                if mn_avg < rd_avg:
                    mn_better_count += 1

    e1_pass = mn_better_count > total_pairs / 2
    print(f"  E1 MinNorm better than Random in {mn_better_count}/{total_pairs} config pairs: {'PASS' if e1_pass else 'FAIL'}")

    # E2: Mean vs Random
    mean_better_count = 0
    total_pairs_e2 = 0
    for mn in mean_agg:
        for rd in random_agg:
            if (mn.get("num_clients") == rd.get("num_clients") and
                mn.get("participation_rate") == rd.get("participation_rate") and
                mn.get("seed") == rd.get("seed")):
                total_pairs_e2 += 1
                mn_avg = (mn.get("delta_obj_0", 0) + mn.get("delta_obj_1", 0)) / 2
                rd_avg = (rd.get("delta_obj_0", 0) + rd.get("delta_obj_1", 0)) / 2
                if mn_avg < rd_avg:
                    mean_better_count += 1

    e2_pass = mean_better_count > total_pairs_e2 / 2
    print(f"  E2 Mean better than Random in {mean_better_count}/{total_pairs_e2} config pairs: {'PASS' if e2_pass else 'FAIL'}")

    # E3: MinNorm weights change across rounds (not degenerate to fixed MGDA)
    # Check a specific experiment
    minnorm_dirs = sorted([
        d for d in sweep_dir.iterdir()
        if d.is_dir() and "minnorm" in d.name and (d / "metrics.csv").exists()
    ])

    e3_pass = False
    if minnorm_dirs:
        exp_dir = minnorm_dirs[0]
        metrics = load_metrics(exp_dir / "metrics.csv")
        jac_norms = [m.get("jacobian_norm", 0) for m in metrics]
        dir_norms = [m.get("direction_norm", 0) for m in metrics]

        # If direction norms vary, the weights are changing
        if len(dir_norms) > 1:
            norm_std = np.std(dir_norms)
            norm_mean = np.mean(dir_norms)
            cv = norm_std / max(abs(norm_mean), 1e-8)
            e3_pass = cv > 0.01
            print(f"  E3 MinNorm direction norm CV = {cv:.4f} (>{0.01} means dynamic): {'PASS' if e3_pass else 'FAIL'}")
            print(f"       Direction norms: {[f'{n:.4f}' for n in dir_norms[:10]]}...")
        else:
            print("  E3 Not enough rounds to check: FAIL")
    else:
        print("  E3 No MinNorm experiments found: FAIL")

    # Additional analysis: Why MinNorm underperforms
    print("\n  --- MinNorm 行为深入分析 ---")
    mn_deltas_0 = [s["delta_obj_0"] for s in minnorm if "delta_obj_0" in s]
    rd_deltas_0 = [s["delta_obj_0"] for s in random_agg if "delta_obj_0" in s]
    mn_deltas_1 = [s["delta_obj_1"] for s in minnorm if "delta_obj_1" in s]
    rd_deltas_1 = [s["delta_obj_1"] for s in random_agg if "delta_obj_1" in s]

    print(f"  MinNorm obj0 delta: mean={np.mean(mn_deltas_0):.4f}, std={np.std(mn_deltas_0):.4f}")
    print(f"  Random  obj0 delta: mean={np.mean(rd_deltas_0):.4f}, std={np.std(rd_deltas_0):.4f}")
    print(f"  MinNorm obj1 delta: mean={np.mean(mn_deltas_1):.4f}, std={np.std(mn_deltas_1):.4f}")
    print(f"  Random  obj1 delta: mean={np.mean(rd_deltas_1):.4f}, std={np.std(rd_deltas_1):.4f}")

    # Check K=8 specifically where MinNorm might degenerate
    mn_k8 = [s for s in minnorm if int(s.get("num_clients", 0)) == 8]
    rd_k8 = [s for s in random_agg if int(s.get("num_clients", 0)) == 8]
    if mn_k8 and rd_k8:
        mn_k8_avg = np.mean([(s["delta_obj_0"] + s["delta_obj_1"]) / 2 for s in mn_k8])
        rd_k8_avg = np.mean([(s["delta_obj_0"] + s["delta_obj_1"]) / 2 for s in rd_k8])
        print(f"  K=8: MinNorm avg delta = {mn_k8_avg:.4f}, Random avg delta = {rd_k8_avg:.4f}")

    return {"E1": e1_pass, "E2": e2_pass, "E3": e3_pass}


def check_F_seed_stability(sweep_dir: Path):
    print("\n" + "=" * 60)
    print("F. 随机种子稳定性")
    print("=" * 60)

    summaries = load_sweep_csv(sweep_dir / "sweep_results.csv")

    # F1: Same seed reproducibility - check by running same config twice
    # We can check if same seed gives same result by looking at experiment IDs
    # For now, we note this requires a separate run
    print("  F1 同种子可复现性: 需要单独验证（见下方专项测试）")
    f1_pass = None

    # F2: Different seeds show consistent trend
    configs = {}
    for s in summaries:
        key = f"{s.get('aggregator')}_K{int(s.get('num_clients', 0))}_C{s.get('participation_rate', 0)}"
        configs.setdefault(key, []).append(s)

    consistent_configs = 0
    total_configs = 0
    for key, group in configs.items():
        if len(group) >= 2:
            total_configs += 1
            all_decrease = all(
                s.get("delta_obj_0", 0) < 0 and s.get("delta_obj_1", 0) < 0
                for s in group
            )
            if all_decrease:
                consistent_configs += 1

    f2_pass = consistent_configs >= total_configs * 2 / 3
    print(f"  F2 Configs where all seeds show decrease: {consistent_configs}/{total_configs} (≥2/3): {'PASS' if f2_pass else 'FAIL'}")

    # F3: Spread reasonable
    unreasonable = 0
    for key, group in configs.items():
        deltas_0 = [s["delta_obj_0"] for s in group if "delta_obj_0" in s]
        if len(deltas_0) >= 2:
            arr = np.array(deltas_0)
            spread = arr.max() - arr.min()
            mean_abs = abs(arr.mean())
            if mean_abs > 0 and spread > mean_abs:
                unreasonable += 1

    f3_pass = unreasonable <= len(configs) / 2
    print(f"  F3 Configs with spread > |mean|: {unreasonable}/{len(configs)} (≤ half): {'PASS' if f3_pass else 'FAIL'}")

    return {"F1": f1_pass, "F2": f2_pass, "F3": f3_pass}


def check_G_metric_completeness(sweep_dir: Path):
    print("\n" + "=" * 60)
    print("G. 指标采集完整性")
    print("=" * 60)

    exp_dirs = [d for d in sorted(sweep_dir.iterdir()) if d.is_dir() and (d / "metrics.csv").exists()]

    # G1: metrics.csv exists and has correct rows
    g1_pass = True
    for d in exp_dirs[:5]:
        metrics = load_metrics(d / "metrics.csv")
        config = load_config(d / "config.yaml")
        expected_rounds = int(config.get("num_rounds", 30))
        if len(metrics) != expected_rounds:
            g1_pass = False
            print(f"  G1 {d.name}: {len(metrics)} rows, expected {expected_rounds}: FAIL")
            break
    if g1_pass:
        print(f"  G1 metrics.csv row count correct in all checked experiments: PASS")

    # G2: config.yaml exists and loadable
    g2_pass = all((d / "config.yaml").exists() for d in exp_dirs)
    print(f"  G2 config.yaml exists in all {len(exp_dirs)} experiments: {'PASS' if g2_pass else 'FAIL'}")

    # G3: summary.md exists
    g3_pass = all((d / "summary.md").exists() for d in exp_dirs)
    print(f"  G3 summary.md exists in all experiments: {'PASS' if g3_pass else 'FAIL'}")

    # G4: stdout.log exists
    g4_pass = all((d / "stdout.log").exists() for d in exp_dirs)
    print(f"  G4 stdout.log exists in all experiments: {'PASS' if g4_pass else 'FAIL'}")

    # G5: plots directory with 4 images
    g5_pass = True
    for d in exp_dirs[:5]:
        plots_dir = d / "plots"
        if not plots_dir.exists():
            g5_pass = False
            break
        expected = ["objectives.png", "norms.png", "communication.png", "time_breakdown.png"]
        for p in expected:
            if not (plots_dir / p).exists():
                g5_pass = False
                break
    print(f"  G5 plots/ with 4 images in checked experiments: {'PASS' if g5_pass else 'FAIL'}")

    # G6: All required fields in metrics.csv
    required_fields = [
        "round", "objective_0", "objective_1", "direction_norm", "jacobian_norm",
        "round_time", "upload_bytes", "download_bytes", "nan_inf_count",
    ]
    g6_pass = True
    if exp_dirs:
        metrics = load_metrics(exp_dirs[0] / "metrics.csv")
        missing = [f for f in required_fields if f not in metrics[0]]
        if missing:
            g6_pass = False
            print(f"  G6 Missing fields: {missing}: FAIL")
        else:
            print(f"  G6 All required fields present in metrics.csv: PASS")

    return {"G1": g1_pass, "G2": g2_pass, "G3": g3_pass, "G4": g4_pass, "G5": g5_pass, "G6": g6_pass}


def check_H_communication(sweep_dir: Path):
    print("\n" + "=" * 60)
    print("H. 通信量验证")
    print("=" * 60)

    summaries = load_sweep_csv(sweep_dir / "sweep_results.csv")

    # H1: Upload bytes = m * d * element_size * num_sampled_clients per round
    # For default config: m=2, d=178, element_size=4 (float32), so per-client upload = 2*178*4 = 1424 bytes
    # Total upload per round = 1424 * num_sampled_clients
    # Total upload over 30 rounds
    m = 2
    d = 178
    elem_size = 4
    per_client_upload = m * d * elem_size
    print(f"  Expected per-client Jacobian upload: {per_client_upload} bytes")

    # Check a specific experiment
    minnorm_dirs = [
        d for d in sorted(sweep_dir.iterdir())
        if d.is_dir() and "minnorm" in d.name and (d / "metrics.csv").exists()
    ]

    h1_pass = False
    if minnorm_dirs:
        exp_dir = minnorm_dirs[0]
        metrics = load_metrics(exp_dir / "metrics.csv")
        if metrics:
            first_round = metrics[0]
            upload = first_round.get("upload_bytes", 0)
            num_sampled = int(first_round.get("num_sampled", 0))
            expected_upload = per_client_upload * num_sampled
            h1_pass = abs(upload - expected_upload) < per_client_upload
            print(f"  H1 Round 0 upload: {upload} bytes, expected ≈ {expected_upload} ({num_sampled} clients × {per_client_upload}): {'PASS' if h1_pass else 'FAIL'}")

    # H2: Download bytes = d * element_size per round
    per_round_download = d * elem_size
    h2_pass = False
    if minnorm_dirs:
        exp_dir = minnorm_dirs[0]
        metrics = load_metrics(exp_dir / "metrics.csv")
        if metrics:
            download = metrics[0].get("download_bytes", 0)
            h2_pass = download == per_round_download
            print(f"  H2 Round 0 download: {download} bytes, expected {per_round_download}: {'PASS' if h2_pass else 'FAIL'}")

    # H3: Upload increases with participation rate
    c025_upload = [s for s in summaries if s.get("participation_rate") == 0.25]
    c050_upload = [s for s in summaries if s.get("participation_rate") == 0.5]
    c100_upload = [s for s in summaries if s.get("participation_rate") == 1.0]

    avg_c025 = np.mean([s["total_upload_bytes"] for s in c025_upload]) if c025_upload else 0
    avg_c050 = np.mean([s["total_upload_bytes"] for s in c050_upload]) if c050_upload else 0
    avg_c100 = np.mean([s["total_upload_bytes"] for s in c100_upload]) if c100_upload else 0

    h3_pass = avg_c025 < avg_c050 < avg_c100
    print(f"  H3 Avg upload: C=0.25={avg_c025:.0f}, C=0.5={avg_c050:.0f}, C=1.0={avg_c100:.0f}: {'PASS' if h3_pass else 'FAIL'}")

    return {"H1": h1_pass, "H2": h2_pass, "H3": h3_pass}


def load_config(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    sweep_dir = Path("results/s1_sweep")

    results_D = check_D_optimization_trend(sweep_dir)
    results_E = check_E_baseline_comparison(sweep_dir)
    results_F = check_F_seed_stability(sweep_dir)
    results_G = check_G_metric_completeness(sweep_dir)
    results_H = check_H_communication(sweep_dir)

    print("\n" + "=" * 60)
    print("D/E/F/G/H SUMMARY")
    print("=" * 60)
    all_results = {**results_D, **results_E, **results_F, **results_G, **results_H}
    for k, v in all_results.items():
        status = "PASS" if v else ("N/A" if v is None else "FAIL")
        print(f"  {k}: {status}")
    total = len([v for v in all_results.values() if v is not None])
    passed = len([v for v in all_results.values() if v is True])
    print(f"\n  Total: {passed}/{total} passed (excluding N/A)")
