from __future__ import annotations

import random

import torch


def jain_fairness_index(values: list[float]) -> float:
    if not values:
        return 0.0
    n = len(values)
    sum_vals = sum(values)
    sum_sq = sum(v ** 2 for v in values)
    if sum_sq < 1e-15:
        return 1.0 if sum_vals < 1e-15 else 0.0
    return (sum_vals ** 2) / (n * sum_sq)


def min_max_gap(values: list[float]) -> float:
    if not values:
        return 0.0
    return max(values) - min(values)


def compute_f1_scores(predictions: torch.Tensor, targets: torch.Tensor, num_tasks: int) -> list[float]:
    f1_scores = []
    with torch.no_grad():
        for t in range(num_tasks):
            pred_labels = predictions[:, t].argmax(dim=-1)
            true_labels = targets[:, t].long()
            tp = ((pred_labels == 1) & (true_labels == 1)).sum().float()
            fp = ((pred_labels == 1) & (true_labels == 0)).sum().float()
            fn = ((pred_labels == 0) & (true_labels == 1)).sum().float()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1.item())
    return f1_scores


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor, num_tasks: int) -> list[float]:
    correct = [0] * num_tasks
    total = 0
    with torch.no_grad():
        for t in range(num_tasks):
            pred_labels = predictions[:, t].argmax(dim=-1)
            true_labels = targets[:, t].long()
            correct[t] += (pred_labels == true_labels).sum().item()
        total += predictions.shape[0]
    return [c / max(total, 1) for c in correct]


def compute_mse_per_task(predictions: torch.Tensor, targets: torch.Tensor, num_tasks: int) -> list[float]:
    mse_list = []
    with torch.no_grad():
        for t in range(num_tasks):
            mse = ((predictions[:, t] - targets[:, t]) ** 2).mean().item()
            mse_list.append(mse)
    return mse_list


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
