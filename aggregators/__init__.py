from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class JacobianAggregator(ABC):
    @abstractmethod
    def __call__(self, jacobian: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def _project_simplex(vector: torch.Tensor) -> torch.Tensor:
    sorted_vector, _ = torch.sort(vector, descending=True)
    cumsum = torch.cumsum(sorted_vector, dim=0)
    steps = torch.arange(1, vector.numel() + 1, device=vector.device, dtype=vector.dtype)
    support = sorted_vector - (cumsum - 1.0) / steps > 0
    rho = int(torch.nonzero(support, as_tuple=False)[-1].item())
    theta = (cumsum[rho] - 1.0) / float(rho + 1)
    return torch.clamp(vector - theta, min=0.0)


class MinNormAggregator(JacobianAggregator):
    """MGDA-style minimum-norm direction finder via projected gradient descent on the simplex."""

    def __init__(self, max_iters: int = 250, lr: float = 0.1, tol: float = 1e-8, max_direction_norm: float = 0.0) -> None:
        self.max_iters = max_iters
        self.lr = lr
        self.tol = tol
        self.max_direction_norm = max_direction_norm

    def __call__(self, jacobian: torch.Tensor) -> torch.Tensor:
        if jacobian.ndim != 2:
            raise ValueError("Jacobian must have shape [num_objectives, num_params].")

        num_objectives = jacobian.shape[0]
        if num_objectives == 1:
            direction = jacobian[0]
        else:
            gramian = jacobian @ jacobian.T
            lambdas = torch.full(
                (num_objectives,),
                1.0 / num_objectives,
                dtype=jacobian.dtype,
                device=jacobian.device,
            )

            for _ in range(self.max_iters):
                gradient = gramian @ lambdas
                candidate = _project_simplex(lambdas - self.lr * gradient)
                if torch.norm(candidate - lambdas, p=2) <= self.tol:
                    lambdas = candidate
                    break
                lambdas = candidate

            direction = jacobian.T @ lambdas

        if self.max_direction_norm > 0:
            dir_norm = torch.norm(direction, p=2)
            if dir_norm > self.max_direction_norm:
                direction = direction * (self.max_direction_norm / dir_norm)

        return direction


class MeanAggregator(JacobianAggregator):
    """Simple average of all objective gradients. Serves as a baseline."""

    def __call__(self, jacobian: torch.Tensor) -> torch.Tensor:
        if jacobian.ndim != 2:
            raise ValueError("Jacobian must have shape [num_objectives, num_params].")
        return jacobian.mean(dim=0)


class RandomAggregator(JacobianAggregator):
    """Random direction from uniform [0,1) weighting of objective gradients. Serves as a baseline."""

    def __init__(self, seed: int | None = None) -> None:
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(seed)

    def __call__(self, jacobian: torch.Tensor) -> torch.Tensor:
        if jacobian.ndim != 2:
            raise ValueError("Jacobian must have shape [num_objectives, num_params].")
        num_objectives = jacobian.shape[0]
        weights = torch.rand(num_objectives, generator=self._gen, device=jacobian.device, dtype=jacobian.dtype)
        weights = weights / weights.sum()
        return jacobian.T @ weights


__all__ = ["JacobianAggregator", "MinNormAggregator", "MeanAggregator", "RandomAggregator"]
