from __future__ import annotations

import random

import torch


def hypervolume_2d(points: list[tuple[float, float]], reference: tuple[float, float]) -> float:
    if not points:
        return 0.0
    dominated = [
        p for p in points
        if p[0] < reference[0] and p[1] < reference[1]
    ]
    if not dominated:
        return 0.0
    dominated.sort(key=lambda p: p[0])
    hv = 0.0
    prev_y = reference[1]
    for p in dominated:
        if p[1] < prev_y:
            hv += (reference[0] - p[0]) * (prev_y - p[1])
            prev_y = p[1]
    return hv


def hypervolume(points: list[list[float]], reference: list[float]) -> float:
    if not points:
        return 0.0
    m = len(reference)
    if m == 2:
        return hypervolume_2d(
            [(p[0], p[1]) for p in points],
            (reference[0], reference[1]),
        )
    dominated = [
        p for p in points
        if all(p[j] < reference[j] for j in range(m))
    ]
    if not dominated:
        return 0.0
    return _hypervolume_monte_carlo(dominated, reference, m, num_samples=100000)


def _hypervolume_monte_carlo(
    points: list[list[float]], reference: list[float], m: int, num_samples: int = 100000
) -> float:
    mins = [min(p[j] for p in points) for j in range(m)]
    volume = 1.0
    for j in range(m):
        dim_range = reference[j] - mins[j]
        if dim_range <= 0:
            return 0.0
        volume *= dim_range
    if volume <= 0:
        return 0.0
    count = 0
    for _ in range(num_samples):
        sample = [random.uniform(mins[j], reference[j]) for j in range(m)]
        if any(all(sample[j] >= p[j] for j in range(m)) for p in points):
            count += 1
    return volume * count / num_samples


def pareto_gap(objective_values: list[float], ideal_point: list[float]) -> float:
    if not objective_values or not ideal_point:
        return float("inf")
    m = len(objective_values)
    total = 0.0
    for j in range(m):
        if ideal_point[j] != 0:
            total += abs(objective_values[j] - ideal_point[j]) / abs(ideal_point[j])
        else:
            total += abs(objective_values[j])
    return total / m


def is_pareto_dominated(a: list[float], b: list[float]) -> bool:
    at_least_one_strictly_better = False
    for aj, bj in zip(a, b):
        if bj > aj:
            return False
        if aj > bj:
            at_least_one_strictly_better = True
    return at_least_one_strictly_better


def extract_pareto_front(points: list[list[float]]) -> list[list[float]]:
    front = []
    for p in points:
        dominated = False
        for q in points:
            if p is not q and is_pareto_dominated(p, q):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


def compute_metrics_from_history(
    objective_histories: list[list[float]],
    reference_point: list[float] | None = None,
    ideal_point: list[float] | None = None,
) -> dict:
    if not objective_histories:
        return {"hypervolume": 0.0, "pareto_gap": float("inf"), "num_pareto_points": 0}

    final_values = objective_histories[-1]
    m = len(final_values)

    if reference_point is None:
        max_vals = [max(h[j] for h in objective_histories) for j in range(m)]
        min_vals = [min(h[j] for h in objective_histories) for j in range(m)]
        ranges = [max_vals[j] - min_vals[j] for j in range(m)]
        reference_point = [max_vals[j] + 0.1 * max(ranges[j], abs(max_vals[j]) * 0.01, 0.01) for j in range(m)]

    if ideal_point is None:
        min_vals = [min(h[j] for h in objective_histories) for j in range(m)]
        ideal_point = min_vals

    all_points = objective_histories
    pareto_front = extract_pareto_front(all_points)

    hv = hypervolume(pareto_front, reference_point)
    pg = pareto_gap(final_values, ideal_point)

    return {
        "hypervolume": hv,
        "pareto_gap": pg,
        "num_pareto_points": len(pareto_front),
        "final_objectives": final_values,
    }
