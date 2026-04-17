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

        aggregator = MinNormAggregator(max_iters=self.max_iters, lr=self.lr)
        sub_direction = aggregator(sub_jacobian)

        full_lambda = torch.zeros(m, dtype=jacobian.dtype, device=jacobian.device)
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
        for i, idx in enumerate(indices):
            full_lambda[idx] = lambdas[i]

        direction = jacobian.T @ full_lambda
        self.last_indices = indices
        return direction, indices


class LocalMomentum:
    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.velocity: torch.Tensor | None = None

    def update(self, direction: torch.Tensor) -> torch.Tensor:
        if self.velocity is None:
            self.velocity = direction.clone()
        else:
            self.velocity = self.beta * self.velocity + (1 - self.beta) * direction
        return self.velocity.clone()

    def reset(self):
        self.velocity = None


class GlobalMomentum:
    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.velocity: torch.Tensor | None = None

    def update(self, aggregated_delta: torch.Tensor) -> torch.Tensor:
        if self.velocity is None:
            self.velocity = aggregated_delta.clone()
        else:
            self.velocity = self.beta * self.velocity + (1 - self.beta) * aggregated_delta
        return self.velocity.clone()

    def reset(self):
        self.velocity = None


__all__ = ["AdaptiveRescaling", "StochasticGramianSolver", "LocalMomentum", "GlobalMomentum"]
