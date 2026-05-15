from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import combinations

import torch


class JacobianAggregator(ABC):
    @abstractmethod
    def __call__(self, jacobian: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def _project_simplex(vector: torch.Tensor) -> torch.Tensor:
    """
    Project onto the probability simplex using fully vectorized operations.

    This implementation avoids Python loops and minimizes .item() calls
    for maximum GPU performance, especially important for small dimensions
    (m<=10) where each .item() causes costly GPU-CPU synchronization.

    Optimization: Uses tensor indexing instead of .item() to avoid
    GPU-CPU synchronization overhead during projection.
    """
    n = vector.numel()

    if n <= 1:
        return torch.clamp(vector, min=0.0)

    sorted_v, _ = torch.sort(vector, descending=True)
    cumsum = torch.cumsum(sorted_v, dim=0)
    steps = torch.arange(1, n + 1, device=vector.device, dtype=vector.dtype)
    support = sorted_v - (cumsum - 1.0) / steps > 0

    if support.any():
        support_indices = torch.nonzero(support, as_tuple=False)
        rho = support_indices[-1, 0]
        theta = (cumsum[rho] - 1.0) / (rho + 1.0)
    else:
        theta = 0.0

    return torch.clamp(vector - theta, min=0.0)


def _project_lower_bound(vector: torch.Tensor, lower_bound: torch.Tensor) -> torch.Tensor:
    return torch.maximum(vector, lower_bound)


class MinNormAggregator(JacobianAggregator):
    """MGDA-style minimum-norm direction finder via projected gradient descent on the simplex."""

    def __init__(self, max_iters: int = 250, lr: float = 0.1, tol: float = 1e-6, max_direction_norm: float = 0.0) -> None:
        self.max_iters = max_iters
        self.lr = lr
        self.tol = tol
        self.max_direction_norm = max_direction_norm

    def __call__(self, jacobian: torch.Tensor) -> torch.Tensor:
        direction, _ = self.solve(jacobian)
        return direction

    def solve(self, jacobian: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if jacobian.ndim != 2:
            raise ValueError("Jacobian must have shape [num_objectives, num_params].")

        num_objectives = jacobian.shape[0]
        if num_objectives == 1:
            direction = jacobian[0]
            lambdas = torch.ones(1, dtype=jacobian.dtype, device=jacobian.device)
        else:
            gramian = jacobian @ jacobian.T
            lambdas = torch.full(
                (num_objectives,),
                1.0 / num_objectives,
                dtype=jacobian.dtype,
                device=jacobian.device,
            )

            with torch.no_grad():
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

        return direction, lambdas.detach().clone()


class UPGradAggregator(JacobianAggregator):
    """UPGrad via the paper's Gramian-space dual formulation."""

    _SOLVERS = {"auto", "active_set", "pgd", "batched_pgd"}

    def __init__(
        self,
        max_iters: int = 250,
        lr: float = 0.1,
        tol: float = 1e-6,
        max_direction_norm: float = 0.0,
        solver: str = "pgd",
    ) -> None:
        if solver not in self._SOLVERS:
            raise ValueError(f"Unknown UPGrad solver: {solver}")
        self.max_iters = max_iters
        self.lr = lr
        self.tol = tol
        self.max_direction_norm = max_direction_norm
        self.solver = solver

    def _solve_box_qp_active_set(self, gramian: torch.Tensor, lower_bound: torch.Tensor) -> torch.Tensor | None:
        num_objectives = gramian.shape[0]
        if num_objectives > 10:
            return None

        all_indices = list(range(num_objectives))
        best_solution = None
        best_value = None
        tol = max(self.tol, 1e-8)

        for free_size in range(num_objectives + 1):
            for free_indices_tuple in combinations(all_indices, free_size):
                free_indices = list(free_indices_tuple)
                bound_indices = [idx for idx in all_indices if idx not in free_indices]
                candidate = lower_bound.clone()

                if free_indices:
                    free_tensor = torch.tensor(free_indices, device=gramian.device, dtype=torch.long)
                    bound_tensor = torch.tensor(bound_indices, device=gramian.device, dtype=torch.long)
                    g_ff = gramian.index_select(0, free_tensor).index_select(1, free_tensor)
                    rhs = torch.zeros(len(free_indices), dtype=gramian.dtype, device=gramian.device)
                    if bound_indices:
                        rhs = -gramian.index_select(0, free_tensor).index_select(1, bound_tensor) @ lower_bound.index_select(0, bound_tensor)
                    solution = torch.linalg.pinv(g_ff, hermitian=True) @ rhs
                    candidate.index_copy_(0, free_tensor, solution)

                if torch.any(candidate < lower_bound - tol):
                    continue

                gradient = gramian @ candidate
                if free_indices:
                    free_tensor = torch.tensor(free_indices, device=gramian.device, dtype=torch.long)
                    if torch.max(torch.abs(gradient.index_select(0, free_tensor))).item() > 5 * tol:
                        continue
                if bound_indices:
                    bound_tensor = torch.tensor(bound_indices, device=gramian.device, dtype=torch.long)
                    if torch.min(gradient.index_select(0, bound_tensor)).item() < -5 * tol:
                        continue

                value = float(candidate @ gradient)
                if best_value is None or value < best_value:
                    best_value = value
                    best_solution = candidate

        return best_solution

    def _solve_box_qp_pgd(self, gramian: torch.Tensor, lower_bound: torch.Tensor) -> torch.Tensor:
        spectral_norm = torch.linalg.matrix_norm(gramian, ord=2)
        safe_step = 1.0 / (2.0 * spectral_norm + 1e-8)
        step_size = min(self.lr, float(safe_step.item())) if torch.isfinite(safe_step) else self.lr
        return self._solve_box_qp_pgd_with_step(gramian, lower_bound, step_size)

    def _solve_box_qp_pgd_with_step(self, gramian: torch.Tensor, lower_bound: torch.Tensor, step_size: float) -> torch.Tensor:
        current = lower_bound.clone()

        with torch.no_grad():
            for _ in range(self.max_iters):
                gradient = 2.0 * (gramian @ current)
                candidate = _project_lower_bound(current - step_size * gradient, lower_bound)
                if torch.norm(candidate - current, p=2) <= self.tol:
                    current = candidate
                    break
                current = candidate
        return current

    def _solve_box_qp_batched_pgd(self, gramian: torch.Tensor, lower_bounds: torch.Tensor) -> torch.Tensor:
        current = lower_bounds.clone()
        spectral_norm = torch.linalg.matrix_norm(gramian, ord=2)
        safe_step = 1.0 / (2.0 * spectral_norm + 1e-8)
        step_size = min(self.lr, float(safe_step.item())) if torch.isfinite(safe_step) else self.lr

        with torch.no_grad():
            for _ in range(self.max_iters):
                gradient = 2.0 * (current @ gramian)
                candidate = _project_lower_bound(current - step_size * gradient, lower_bounds)
                if torch.norm(candidate - current, p="fro") <= self.tol:
                    current = candidate
                    break
                current = candidate
        return current

    def _solve_box_qp(self, gramian: torch.Tensor, lower_bound: torch.Tensor) -> torch.Tensor:
        use_active_set = self.solver == "active_set" or (self.solver == "auto" and gramian.shape[0] <= 3)
        if use_active_set:
            active_set_solution = self._solve_box_qp_active_set(gramian, lower_bound)
            if active_set_solution is not None:
                return active_set_solution
            if self.solver == "active_set":
                raise RuntimeError("Active-set UPGrad solve failed.")
        return self._solve_box_qp_pgd(gramian, lower_bound)

    def __call__(self, jacobian: torch.Tensor) -> torch.Tensor:
        direction, _ = self.solve(jacobian)
        return direction

    def solve(self, jacobian: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if jacobian.ndim != 2:
            raise ValueError("Jacobian must have shape [num_objectives, num_params].")

        num_objectives = jacobian.shape[0]
        if num_objectives == 1:
            direction = jacobian[0]
            weights = torch.ones(1, dtype=jacobian.dtype, device=jacobian.device)
        else:
            gramian = jacobian @ jacobian.T
            gramian = 0.5 * (gramian + gramian.T)
            lower_bounds = torch.eye(num_objectives, dtype=jacobian.dtype, device=jacobian.device)

            if self.solver == "batched_pgd":
                solutions = self._solve_box_qp_batched_pgd(gramian, lower_bounds)
            else:
                solutions = []
                for objective_idx in range(num_objectives):
                    lower_bound = lower_bounds[objective_idx]
                    solutions.append(self._solve_box_qp(gramian, lower_bound))
                solutions = torch.stack(solutions, dim=0)

            weights = solutions.mean(dim=0)
            direction = jacobian.T @ weights

        if self.max_direction_norm > 0:
            dir_norm = torch.norm(direction, p=2)
            if dir_norm > self.max_direction_norm:
                direction = direction * (self.max_direction_norm / dir_norm)

        return direction, weights.detach().clone()


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


__all__ = [
    "JacobianAggregator",
    "MinNormAggregator",
    "UPGradAggregator",
    "MeanAggregator",
    "RandomAggregator",
    "_project_simplex",
    "_project_lower_bound",
]
