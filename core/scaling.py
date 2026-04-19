from __future__ import annotations

import random

import torch

from fedjd.aggregators import MinNormAggregator


class AdaptiveRescaling:
    def __init__(self, epsilon: float = 1e-8, max_scale: float = 10.0):
        self.epsilon = epsilon
        self.max_scale = max_scale
        self.last_scale = 1.0

    def __call__(self, direction: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        mean_grad = jacobian.mean(dim=0)
        N_raw = float(torch.norm(mean_grad, p=2).item())
        N_d = float(torch.norm(direction, p=2).item()) + self.epsilon
        scale = min(N_raw / N_d, self.max_scale)
        self.last_scale = scale
        return direction * scale


class StochasticGramianSolver:
    def __init__(self, subset_size: int = 4, max_iters: int = 250, lr: float = 0.1, seed: int | None = None):
        self.subset_size = subset_size
        self.max_iters = max_iters
        self.lr = lr
        self.rng = random.Random(seed)
        self.last_indices: list[int] = []

    def solve(self, jacobian: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        m = jacobian.shape[0]
        if m <= self.subset_size:
            aggregator = MinNormAggregator(max_iters=self.max_iters, lr=self.lr)
            direction = aggregator(jacobian)
            self.last_indices = list(range(m))
            return direction, self.last_indices

        indices = sorted(self.rng.sample(range(m), self.subset_size))
        sub_jacobian = jacobian[indices]

        # 内部MinNorm计算，同时得到lambda
        sub_gramian = sub_jacobian @ sub_jacobian.T
        lambdas = torch.full((self.subset_size,), 1.0 / self.subset_size, dtype=jacobian.dtype, device=jacobian.device)
        for _ in range(self.max_iters):
            gradient = sub_gramian @ lambdas
            from fedjd.aggregators import _project_simplex
            candidate = _project_simplex(lambdas - self.lr * gradient)
            if torch.norm(candidate - lambdas, p=2) <= 1e-8:
                lambdas = candidate
                break
            lambdas = candidate

        # 将子集lambda扩展到完整空间
        full_lambda = torch.zeros(m, dtype=jacobian.dtype, device=jacobian.device)
        for i, idx in enumerate(indices):
            full_lambda[idx] = lambdas[i]

        direction = jacobian.T @ full_lambda
        self.last_lambda = full_lambda
        self.last_indices = indices
        return direction, indices

    @property
    def last_lambda(self) -> torch.Tensor | None:
        return getattr(self, '_last_lambda', None)

    @last_lambda.setter
    def last_lambda(self, value: torch.Tensor):
        self._last_lambda = value


def compute_avg_cosine_sim(jacobian: torch.Tensor) -> float:
    m = jacobian.shape[0]
    if m <= 1:
        return 1.0
    norms = torch.norm(jacobian, p=2, dim=1, keepdim=True).clamp(min=1e-8)
    normalized = jacobian / norms
    sim_matrix = normalized @ normalized.T
    mask = ~torch.eye(m, dtype=torch.bool, device=jacobian.device)
    avg_sim = float(sim_matrix[mask].mean().item())
    return max(-1.0, min(1.0, avg_sim))


class ConflictAwareMomentum:
    def __init__(self, base_beta: float = 0.9, min_beta: float = 0.1):
        self.base_beta = base_beta
        self.min_beta = min_beta
        self.last_avg_cosine_sim: float = 0.0
        self.last_effective_beta: float = base_beta

    def compute_beta(self, avg_cosine_sim: float) -> float:
        effective_beta = self.base_beta * (1.0 - avg_cosine_sim)
        effective_beta = max(effective_beta, self.min_beta)
        self.last_avg_cosine_sim = avg_cosine_sim
        self.last_effective_beta = effective_beta
        return effective_beta


class LocalMomentum:
    def __init__(self, beta: float = 0.9, conflict_aware: bool = False, min_beta: float = 0.1):
        self.base_beta = beta
        self.beta = beta
        self.velocity: torch.Tensor | None = None
        self.conflict_aware = conflict_aware
        self.conflict_module = ConflictAwareMomentum(base_beta=beta, min_beta=min_beta) if conflict_aware else None

    def update(self, direction: torch.Tensor, jacobian: torch.Tensor | None = None) -> torch.Tensor:
        if self.conflict_aware and self.conflict_module is not None and jacobian is not None:
            avg_sim = compute_avg_cosine_sim(jacobian)
            self.beta = self.conflict_module.compute_beta(avg_sim)

        if self.velocity is None:
            self.velocity = direction.clone()
        else:
            self.velocity = self.beta * self.velocity + (1 - self.beta) * direction
        return self.velocity.clone()

    def reset(self):
        self.velocity = None


class GlobalMomentum:
    def __init__(self, beta: float = 0.9, conflict_aware: bool = False, min_beta: float = 0.1):
        self.base_beta = beta
        self.beta = beta
        self.velocity: torch.Tensor | None = None
        self.conflict_aware = conflict_aware
        self.conflict_module = ConflictAwareMomentum(base_beta=beta, min_beta=min_beta) if conflict_aware else None

    def update(self, aggregated_delta: torch.Tensor, avg_cosine_sim: float | None = None) -> torch.Tensor:
        if self.conflict_aware and self.conflict_module is not None and avg_cosine_sim is not None:
            self.beta = self.conflict_module.compute_beta(avg_cosine_sim)

        if self.velocity is None:
            self.velocity = aggregated_delta.clone()
        else:
            self.velocity = self.beta * self.velocity + (1 - self.beta) * aggregated_delta
        return self.velocity.clone()

    def reset(self):
        self.velocity = None


__all__ = [
    "AdaptiveRescaling", "StochasticGramianSolver", "LocalMomentum", "GlobalMomentum",
    "ConflictAwareMomentum", "compute_avg_cosine_sim",
]
