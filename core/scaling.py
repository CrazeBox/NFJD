﻿from __future__ import annotations

import random

import torch

from fedjd.aggregators import MinNormAggregator, _project_simplex


class AdaptiveRescaling:
    def __init__(self, epsilon: float = 1e-8, max_scale: float = 10.0):
        self.epsilon = epsilon
        self.max_scale = max_scale
        self.last_scale = 1.0
        self._cached_mean_grad_norm: torch.Tensor | None = None
        self._cached_direction_norm: torch.Tensor | None = None

    def __call__(self, direction: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        mean_grad = jacobian.mean(dim=0)
        raw_norm = torch.norm(mean_grad, p=2)
        direction_norm = torch.norm(direction, p=2) + self.epsilon
        scale = torch.minimum(raw_norm / direction_norm, torch.tensor(self.max_scale, device=direction.device, dtype=direction.dtype))
        self.last_scale = float(scale.item())
        self._cached_mean_grad_norm = raw_norm
        self._cached_direction_norm = direction_norm
        return direction * scale


class StochasticGramianSolver:
    def __init__(self, subset_size: int = 4, max_iters: int = 250, lr: float = 0.1, seed: int | None = None):
        self.subset_size = subset_size
        self.max_iters = max_iters
        self.lr = lr
        self.rng = random.Random(seed)
        self.last_indices: list[int] = []
        self._last_lambda: torch.Tensor | None = None

    def sample_indices(self, num_objectives: int) -> list[int]:
        if num_objectives <= self.subset_size:
            return list(range(num_objectives))
        return sorted(self.rng.sample(range(num_objectives), self.subset_size))

    def solve_sampled(
        self,
        sampled_jacobian: torch.Tensor,
        total_objectives: int,
        sampled_indices: list[int],
    ) -> tuple[torch.Tensor, list[int]]:
        if sampled_jacobian.ndim != 2:
            raise ValueError("Jacobian must have shape [num_objectives, num_params].")

        sampled_count = sampled_jacobian.shape[0]
        if sampled_count == 0:
            raise ValueError("At least one objective must be sampled.")

        if sampled_count == 1:
            lambdas = torch.ones(1, dtype=sampled_jacobian.dtype, device=sampled_jacobian.device)
        else:
            sub_gramian = sampled_jacobian @ sampled_jacobian.T
            lambdas = torch.full(
                (sampled_count,),
                1.0 / sampled_count,
                dtype=sampled_jacobian.dtype,
                device=sampled_jacobian.device,
            )
            for _ in range(self.max_iters):
                gradient = sub_gramian @ lambdas
                candidate = _project_simplex(lambdas - self.lr * gradient)
                if torch.norm(candidate - lambdas, p=2) <= 1e-8:
                    lambdas = candidate
                    break
                lambdas = candidate

        full_lambda = torch.zeros(
            total_objectives,
            dtype=sampled_jacobian.dtype,
            device=sampled_jacobian.device,
        )
        for i, idx in enumerate(sampled_indices):
            full_lambda[idx] = lambdas[i]

        direction = sampled_jacobian.T @ lambdas
        self.last_lambda = full_lambda
        self.last_indices = list(sampled_indices)
        return direction, self.last_indices

    def solve(self, jacobian: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        if jacobian.ndim != 2:
            raise ValueError("Jacobian must have shape [num_objectives, num_params].")

        m = jacobian.shape[0]
        if m <= self.subset_size:
            aggregator = MinNormAggregator(max_iters=self.max_iters, lr=self.lr)
            direction = aggregator(jacobian)
            self.last_indices = list(range(m))
            self.last_lambda = torch.ones(m, dtype=jacobian.dtype, device=jacobian.device) / max(m, 1)
            return direction, self.last_indices

        indices = self.sample_indices(m)
        return self.solve_sampled(jacobian[indices], total_objectives=m, sampled_indices=indices)

    @property
    def last_lambda(self) -> torch.Tensor | None:
        return self._last_lambda

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
