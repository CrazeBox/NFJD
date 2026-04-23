from __future__ import annotations

from fedjd.metrics import extract_pareto_front, hypervolume, jain_fairness_index, min_max_gap


def compute_relative_improvements(initial_obj: list[float], final_obj: list[float]) -> list[float]:
    ri_values: list[float] = []
    for init_v, final_v in zip(initial_obj, final_obj):
        if abs(init_v) > 1e-10:
            ri_values.append((init_v - final_v) / abs(init_v))
        else:
            ri_values.append(1.0 if final_v < 0.0 else 0.0)
    return ri_values


def compute_improvement_fairness(ri_values: list[float]) -> tuple[float, float]:
    if not ri_values:
        return 0.0, 0.0

    positive_scores = [max(v, 0.0) for v in ri_values]
    if sum(positive_scores) <= 1e-12:
        jfi = 0.0
    else:
        jfi = jain_fairness_index(positive_scores)
    mmag = min_max_gap(ri_values)
    return jfi, mmag


def compute_normalized_pareto_metrics(objective_history: list[list[float]]) -> tuple[float, float]:
    if not objective_history:
        return 0.0, 0.0

    m = len(objective_history[0])
    min_vals = [min(point[j] for point in objective_history) for j in range(m)]
    max_vals = [max(point[j] for point in objective_history) for j in range(m)]
    ranges = []
    for j in range(m):
        diff = max_vals[j] - min_vals[j]
        ranges.append(1.0 if abs(diff) < 1e-10 else diff)

    normalized_history: list[list[float]] = []
    for point in objective_history:
        normalized_history.append([(point[j] - min_vals[j]) / ranges[j] for j in range(m)])

    ref_point = [1.1] * m
    pareto_front = extract_pareto_front(normalized_history)
    raw_hv = hypervolume(pareto_front, ref_point)
    max_possible_hv = 1.1 ** m
    hv = raw_hv / max_possible_hv if max_possible_hv > 0 else 0.0

    final_point = normalized_history[-1]
    pareto_gap = sum(final_point) / m if m > 0 else 0.0
    return hv, pareto_gap


def summarize_objective_history(initial_obj: list[float], history_objectives: list[list[float]]) -> dict[str, float | bool | list[float]]:
    if not history_objectives:
        raise ValueError("history_objectives must not be empty")

    final_obj = history_objectives[-1]
    ri_values = compute_relative_improvements(initial_obj, final_obj)
    task_jfi, task_mmag = compute_improvement_fairness(ri_values)
    hv, pareto_gap = compute_normalized_pareto_metrics([initial_obj] + history_objectives)

    return {
        "final_obj": final_obj,
        "all_decreased": all(final_obj[j] <= initial_obj[j] for j in range(len(initial_obj))),
        "ri_values": ri_values,
        "avg_ri": sum(ri_values) / max(len(ri_values), 1),
        "task_jfi": task_jfi,
        "task_mmag": task_mmag,
        "hypervolume": hv,
        "pareto_gap": pareto_gap,
    }


def summarize_round_history(history: list) -> dict[str, float]:
    rounds = max(len(history), 1)
    total_upload = float(sum(getattr(s, "upload_bytes", 0) for s in history))
    total_download = float(sum(getattr(s, "download_bytes", 0) for s in history))
    total_sampled_clients = float(sum(max(int(getattr(s, "num_sampled_clients", 0)), 0) for s in history))

    avg_upload = total_upload / rounds
    avg_download = total_download / rounds
    avg_round_time = sum(getattr(s, "round_time", 0.0) for s in history) / rounds
    avg_total_comm = (total_upload + total_download) / rounds
    upload_per_client = total_upload / total_sampled_clients if total_sampled_clients > 0 else 0.0
    download_per_client = total_download / total_sampled_clients if total_sampled_clients > 0 else 0.0
    total_comm_per_client = (total_upload + total_download) / total_sampled_clients if total_sampled_clients > 0 else 0.0

    return {
        "avg_upload_bytes": avg_upload,
        "avg_download_bytes": avg_download,
        "avg_total_comm_bytes": avg_total_comm,
        "avg_round_time": avg_round_time,
        "upload_per_client": upload_per_client,
        "download_per_client": download_per_client,
        "total_comm_per_client": total_comm_per_client,
    }
