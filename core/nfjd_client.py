from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import chain
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedjd.aggregators import UPGradAggregator
from fedjd.core.scaling import AdaptiveRescaling, LocalMomentum, StochasticGramianSolver, compute_avg_cosine_sim

ObjectiveFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], list[torch.Tensor]]


def flatten_gradients(parameters) -> torch.Tensor:
    chunks = []
    for parameter in parameters:
        if parameter.grad is None:
            chunks.append(torch.zeros_like(parameter).reshape(-1))
        else:
            chunks.append(parameter.grad.detach().reshape(-1).clone())
    return torch.cat(chunks)


def flatten_parameters(parameters) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in parameters])


def assign_flat_parameters(parameters, flat_vector: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        parameter.data.copy_(flat_vector[offset:offset + size].view_as(parameter))
        offset += size


def add_flat_update_(parameters, flat_update: torch.Tensor, alpha: float = 1.0) -> None:
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        parameter.data.add_(flat_update[offset:offset + size].view_as(parameter), alpha=alpha)
        offset += size


def _unique_parameters(modules: list[nn.Module]) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for param in module.parameters():
            if id(param) not in seen:
                params.append(param)
                seen.add(id(param))
    return params


def get_model_parameter_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    shared_modules: list[nn.Module] = []
    for attr in ("shared", "fc_shared", "features", "shared_fc"):
        module = getattr(model, attr, None)
        if isinstance(module, nn.Module):
            shared_modules.append(module)

    head_modules = getattr(model, "heads", None)
    shared_params = _unique_parameters(shared_modules)
    head_params: list[nn.Parameter] = []
    if isinstance(head_modules, nn.ModuleList):
        head_params = _unique_parameters(list(head_modules))

    known_ids = {id(param) for param in chain(shared_params, head_params)}
    leftover = [param for param in model.parameters() if id(param) not in known_ids]
    if leftover:
        shared_params.extend(leftover)
    if not shared_params:
        shared_params = list(model.parameters())

    return shared_params, head_params


def _flatten_gradient_list(grads: tuple[torch.Tensor | None, ...], params: list[nn.Parameter]) -> torch.Tensor:
    if not params:
        return torch.zeros(0, dtype=torch.float32)
    chunks = []
    for grad, param in zip(grads, params):
        if grad is None:
            chunks.append(torch.zeros_like(param).reshape(-1))
        else:
            chunks.append(grad.detach().reshape(-1))
    return torch.cat(chunks)


def _apply_gradients(params: list[nn.Parameter], grads: tuple[torch.Tensor | None, ...], learning_rate: float) -> None:
    if not params:
        return
    with torch.no_grad():
        for param, grad in zip(params, grads):
            if grad is not None:
                param.add_(grad, alpha=-learning_rate)


def _apply_flat_direction(params: list[nn.Parameter], direction: torch.Tensor, learning_rate: float) -> None:
    if not params:
        return
    offset = 0
    with torch.no_grad():
        for param in params:
            size = param.numel()
            param.add_(direction[offset:offset + size].view_as(param), alpha=-learning_rate)
            offset += size


def _project_onto_dual_cone(
    reference: torch.Tensor,
    jacobian: torch.Tensor,
    max_iters: int = 100,
    lr: float = 0.1,
    tol: float = 1e-6,
) -> torch.Tensor:
    if reference.ndim != 1:
        raise ValueError("Reference direction must be a flat vector.")
    if jacobian.ndim != 2:
        raise ValueError("Jacobian must have shape [num_objectives, num_params].")
    if jacobian.shape[1] != reference.numel():
        raise ValueError("Reference dimension must match Jacobian parameter dimension.")
    if jacobian.numel() == 0 or reference.numel() == 0:
        return reference

    linear = jacobian @ reference
    if torch.min(linear).item() >= -tol:
        return reference

    gramian = jacobian @ jacobian.T
    gramian = 0.5 * (gramian + gramian.T)
    spectral_norm = torch.linalg.matrix_norm(gramian, ord=2)
    if not torch.isfinite(spectral_norm) or spectral_norm.item() <= 1e-12:
        return reference

    safe_step = 1.0 / (spectral_norm + 1e-8)
    step_size = min(lr, float(safe_step.item()))
    lambdas = torch.zeros(jacobian.shape[0], dtype=jacobian.dtype, device=jacobian.device)

    with torch.no_grad():
        for _ in range(max_iters):
            gradient = gramian @ lambdas + linear
            candidate = torch.clamp(lambdas - step_size * gradient, min=0.0)
            if torch.norm(candidate - lambdas, p=2) <= tol:
                lambdas = candidate
                break
            lambdas = candidate

    return reference + jacobian.T @ lambdas


def _solve_nonnegative_least_squares(
    basis: torch.Tensor,
    target: torch.Tensor,
    max_iters: int = 100,
    lr: float = 0.1,
    tol: float = 1e-6,
) -> torch.Tensor:
    if basis.ndim != 2:
        raise ValueError("Basis must have shape [num_basis, num_params].")
    if target.ndim != 1:
        raise ValueError("Target must be a flat vector.")
    if basis.shape[1] != target.numel():
        raise ValueError("Basis dimension must match target dimension.")
    num_basis = basis.shape[0]
    if num_basis == 0:
        return torch.zeros(0, dtype=target.dtype, device=target.device)

    gramian = basis @ basis.T
    linear = basis @ target
    spectral_norm = torch.linalg.matrix_norm(gramian, ord=2)
    if not torch.isfinite(spectral_norm) or spectral_norm.item() <= 1e-12:
        return torch.zeros(num_basis, dtype=target.dtype, device=target.device)

    safe_step = 1.0 / (spectral_norm + 1e-8)
    step_size = min(lr, float(safe_step.item()))
    coeffs = torch.zeros(num_basis, dtype=target.dtype, device=target.device)

    with torch.no_grad():
        for _ in range(max_iters):
            gradient = gramian @ coeffs - linear
            candidate = torch.clamp(coeffs - step_size * gradient, min=0.0)
            if torch.norm(candidate - coeffs, p=2) <= tol:
                coeffs = candidate
                break
            coeffs = candidate
    return coeffs


@dataclass
class ClientResult:
    client_id: int
    delta_theta: torch.Tensor
    shared_delta_theta: torch.Tensor | None
    control_delta_shared: torch.Tensor | None
    num_examples: int
    compute_time: float = 0.0
    num_local_epochs: int = 1
    final_lambda: torch.Tensor | None = None
    rescale_factor: float = 1.0
    sampled_indices: list[int] | None = None
    avg_cosine_sim: float = 0.0
    avg_prox_ratio: float = 0.0
    avg_scaffold_ratio: float = 0.0
    avg_cone_margin: float = 0.0
    avg_cone_cosine: float = 0.0
    jacobian: torch.Tensor | None = None
    align_scores: torch.Tensor | None = None


class NFJDClient:
    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        batch_size: int,
        device: torch.device,
        local_epochs: int = 3,
        learning_rate: float = 0.01,
        local_momentum_beta: float = 0.9,
        use_stochastic_gramian: bool = True,
        stochastic_subset_size: int = 4,
        stochastic_seed: int | None = None,
        use_adaptive_rescaling: bool = True,
        rescaling_max_scale: float = 5.0,
        minnorm_max_iters: int = 250,
        minnorm_lr: float = 0.1,
        conflict_aware_momentum: bool = True,
        momentum_min_beta: float = 0.1,
        recompute_interval: int = 4,
        use_mixed_precision: bool = True,
        exact_upgrad: bool = False,
        use_objective_normalization: bool = False,
        objective_norm_momentum: float = 0.9,
        objective_norm_epsilon: float = 1e-8,
        upload_align_scores: bool = True,
        shared_prox_mu: float = 0.0,
        cone_align_positive_only: bool = False,
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.minnorm_max_iters = minnorm_max_iters
        self.minnorm_lr = minnorm_lr
        self.recompute_interval = max(recompute_interval, 1)
        self.exact_upgrad = exact_upgrad

        self.use_mixed_precision = use_mixed_precision and device.type == "cuda"
        self.use_objective_normalization = use_objective_normalization
        self.objective_norm_momentum = objective_norm_momentum
        self.objective_norm_epsilon = objective_norm_epsilon
        self.upload_align_scores = upload_align_scores
        self.shared_prox_mu = max(float(shared_prox_mu), 0.0)
        self.cone_align_positive_only = cone_align_positive_only
        self.shared_control_local: torch.Tensor | None = None

        self.prev_lambda: torch.Tensor | None = None
        self._objective_norm_ema: torch.Tensor | None = None
        self.local_momentum = LocalMomentum(
            beta=local_momentum_beta,
            conflict_aware=conflict_aware_momentum,
            min_beta=momentum_min_beta,
        )
        self.adaptive_rescaling = AdaptiveRescaling(epsilon=1e-8, max_scale=rescaling_max_scale) if use_adaptive_rescaling else None
        self.stochastic_solver = StochasticGramianSolver(
            subset_size=stochastic_subset_size,
            max_iters=minnorm_max_iters,
            lr=minnorm_lr,
            seed=stochastic_seed,
            mode="upgrad",
        ) if use_stochastic_gramian else None
        self.conflict_aware_momentum = conflict_aware_momentum

    def _normalize_jacobian(self, jacobian: torch.Tensor) -> torch.Tensor:
        if not self.use_objective_normalization:
            return jacobian

        row_norms = torch.norm(jacobian, p=2, dim=1).detach()
        if self._objective_norm_ema is None or self._objective_norm_ema.shape != row_norms.shape:
            self._objective_norm_ema = row_norms
        else:
            momentum = self.objective_norm_momentum
            self._objective_norm_ema = momentum * self._objective_norm_ema + (1.0 - momentum) * row_norms

        denom = self._objective_norm_ema.clamp(min=self.objective_norm_epsilon).to(jacobian.device, jacobian.dtype).unsqueeze(1)
        return jacobian / denom

    @property
    def num_examples(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _compute_dynamic_iters(m: int) -> int:
        if m <= 2:
            return 50
        if m <= 5:
            return 100
        if m <= 8:
            return 200
        return 250

    def probe_shared_direction(
        self,
        model: nn.Module,
        objective_fn: ObjectiveFn,
        task_weights: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        try:
            batch_inputs, batch_targets = next(iter(loader))
        except StopIteration:
            return None

        batch_inputs = batch_inputs.to(self.device)
        batch_targets = batch_targets.to(self.device)
        self._objective_norm_ema = None

        shared_params, _ = get_model_parameter_groups(model)
        if not shared_params:
            return None

        with torch.amp.autocast("cuda", enabled=self.use_mixed_precision):
            predictions = model(batch_inputs)
            losses = objective_fn(predictions, batch_targets, batch_inputs)

        m = len(losses)
        dynamic_iters = self._compute_dynamic_iters(m)
        exact_solver = self.exact_upgrad and m <= 8
        aggregator = UPGradAggregator(
            max_iters=max(dynamic_iters, 500) if exact_solver else dynamic_iters,
            lr=self.minnorm_lr,
            tol=1e-9 if exact_solver else 1e-6,
            solver="auto",
        )

        if task_weights is None:
            weights_t = torch.ones(m, dtype=losses[0].dtype, device=self.device)
        else:
            weights_t = task_weights.to(self.device, dtype=losses[0].dtype)
            if weights_t.numel() != m:
                raise ValueError(f"Expected {m} task weights, got {weights_t.numel()}.")

        independent_grads = []
        for grad_pos, objective_idx in enumerate(range(m)):
            grads = torch.autograd.grad(
                losses[objective_idx],
                shared_params,
                retain_graph=grad_pos < m - 1,
                allow_unused=True,
            )
            independent_grads.append(_flatten_gradient_list(grads, shared_params))

        jacobian = torch.stack(independent_grads, dim=0)
        jacobian = self._normalize_jacobian(jacobian)
        jacobian = jacobian * weights_t.unsqueeze(1)
        direction, _ = aggregator.solve(jacobian)
        dir_norm = torch.norm(direction, p=2)
        if dir_norm.item() <= 1e-12:
            return None
        return (direction / dir_norm).detach().clone()

    def local_update(
        self,
        model: nn.Module,
        objective_fn: ObjectiveFn,
        task_weights: torch.Tensor | None = None,
        shared_control_global: torch.Tensor | None = None,
        cone_reference_shared_direction: torch.Tensor | None = None,
        cone_reference_shared_basis: torch.Tensor | None = None,
        cone_align_alpha: float = 0.0,
    ) -> ClientResult:
        start = time.time()
        self.prev_lambda = None
        self._objective_norm_ema = None
        self.local_momentum.reset()
        theta_init = flatten_parameters(model.parameters()).clone()
        m = None
        final_lambda = None
        rescale_factor = 1.0
        last_rescale_factor = 1.0
        sampled_indices = None
        avg_cosine_sim = 0.0
        aggregator = None
        step_idx = 0
        rescale_history: list[float] = []
        cosine_history: list[float] = []
        prox_ratio_history: list[float] = []
        scaffold_ratio_history: list[float] = []
        cone_margin_history: list[float] = []
        cone_cosine_history: list[float] = []
        last_avg_cosine_sim = 0.0
        last_raw_jacobian = None
        last_projected_reference_unit = None
        shared_params, head_params = get_model_parameter_groups(model)
        theta_init_shared = flatten_parameters(shared_params).clone()
        if self.shared_control_local is None or self.shared_control_local.numel() != theta_init_shared.numel():
            self.shared_control_local = torch.zeros_like(theta_init_shared)
        scaffold_correction = None
        if shared_control_global is not None:
            scaffold_correction = shared_control_global.to(self.device, dtype=theta_init_shared.dtype) - self.shared_control_local.to(self.device, dtype=theta_init_shared.dtype)
        cone_reference = None
        if cone_reference_shared_direction is not None:
            cone_reference = cone_reference_shared_direction.to(self.device)
        cone_basis = None
        if cone_reference_shared_basis is not None:
            cone_basis = cone_reference_shared_basis.to(self.device)

        def _apply_cone_alignment(direction: torch.Tensor, jacobian: torch.Tensor | None) -> torch.Tensor:
            nonlocal last_projected_reference_unit
            if cone_reference is None or cone_align_alpha <= 0 or jacobian is None or direction.numel() == 0:
                return direction

            ref_norm = torch.norm(cone_reference, p=2)
            if ref_norm.item() <= 1e-12:
                return direction

            ref_unit = cone_reference / ref_norm
            margin = float(torch.min(jacobian @ ref_unit).item())
            cone_margin_history.append(margin)
            projected = _project_onto_dual_cone(
                ref_unit.to(direction.device, dtype=direction.dtype),
                jacobian,
                max_iters=max(50, self.minnorm_max_iters),
                lr=self.minnorm_lr,
                tol=1e-6,
            )
            proj_norm = torch.norm(projected, p=2)
            dir_norm = torch.norm(direction, p=2)
            if proj_norm.item() <= 1e-12 or dir_norm.item() <= 1e-12:
                return direction

            last_projected_reference_unit = projected / proj_norm
            projected_scaled = last_projected_reference_unit * dir_norm
            cosine = torch.dot(direction, projected_scaled) / (torch.norm(direction, p=2) * torch.norm(projected_scaled, p=2) + 1e-12)
            cone_cosine_history.append(float(cosine.item()))
            if self.cone_align_positive_only and float(cosine.item()) <= 0.0:
                return direction
            return (1.0 - cone_align_alpha) * direction + cone_align_alpha * projected_scaled

        def _reuse_last_projected_reference(direction: torch.Tensor) -> torch.Tensor:
            if last_projected_reference_unit is None or cone_align_alpha <= 0 or direction.numel() == 0:
                return direction
            dir_norm = torch.norm(direction, p=2)
            if dir_norm.item() <= 1e-12:
                return direction
            projected_scaled = last_projected_reference_unit.to(direction.device, dtype=direction.dtype) * dir_norm
            cosine = torch.dot(direction, projected_scaled) / (torch.norm(direction, p=2) * torch.norm(projected_scaled, p=2) + 1e-12)
            cone_cosine_history.append(float(cosine.item()))
            if self.cone_align_positive_only and float(cosine.item()) <= 0.0:
                return direction
            return (1.0 - cone_align_alpha) * direction + cone_align_alpha * projected_scaled

        def _apply_cone_basis_alignment(direction: torch.Tensor, jacobian: torch.Tensor | None) -> torch.Tensor:
            if cone_basis is None or cone_align_alpha <= 0 or jacobian is None or direction.numel() == 0:
                return direction
            projected_list = []
            raw_margins = []
            for basis_vec in cone_basis:
                basis_norm = torch.norm(basis_vec, p=2)
                if basis_norm.item() <= 1e-12:
                    continue
                basis_unit = basis_vec / basis_norm
                raw_margins.append(float(torch.min(jacobian @ basis_unit).item()))
                projected = _project_onto_dual_cone(
                    basis_unit.to(direction.device, dtype=direction.dtype),
                    jacobian,
                    max_iters=max(50, self.minnorm_max_iters),
                    lr=self.minnorm_lr,
                    tol=1e-6,
                )
                proj_norm = torch.norm(projected, p=2)
                if proj_norm.item() <= 1e-12:
                    continue
                projected_list.append(projected / proj_norm)

            if not projected_list:
                return direction

            cone_margin_history.append(max(raw_margins) if raw_margins else 0.0)
            projected_basis = torch.stack(projected_list, dim=0)
            coeffs = _solve_nonnegative_least_squares(
                projected_basis,
                direction,
                max_iters=max(50, self.minnorm_max_iters),
                lr=self.minnorm_lr,
                tol=1e-6,
            )
            if coeffs.numel() == 0 or torch.max(coeffs).item() <= 1e-12:
                return direction
            cone_direction = projected_basis.T @ coeffs
            cone_norm = torch.norm(cone_direction, p=2)
            dir_norm = torch.norm(direction, p=2)
            if cone_norm.item() <= 1e-12 or dir_norm.item() <= 1e-12:
                return direction

            cone_direction = cone_direction * (dir_norm / cone_norm)
            cosine = torch.dot(direction, cone_direction) / (torch.norm(direction, p=2) * torch.norm(cone_direction, p=2) + 1e-12)
            cone_cosine_history.append(float(cosine.item()))
            if self.cone_align_positive_only and float(cosine.item()) <= 0.0:
                return direction
            return (1.0 - cone_align_alpha) * direction + cone_align_alpha * cone_direction

        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.local_epochs):
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                with torch.amp.autocast("cuda", enabled=self.use_mixed_precision):
                    predictions = model(batch_inputs)
                    losses = objective_fn(predictions, batch_targets, batch_inputs)
                if m is None:
                    m = len(losses)
                    dynamic_iters = self._compute_dynamic_iters(m)
                    exact_solver = self.exact_upgrad and m <= 8
                    aggregator = UPGradAggregator(
                        max_iters=max(dynamic_iters, 500) if exact_solver else dynamic_iters,
                        lr=self.minnorm_lr,
                        tol=1e-9 if exact_solver else 1e-6,
                        solver="auto",
                    )
                    if self.stochastic_solver is not None:
                        self.stochastic_solver.max_iters = dynamic_iters

                if task_weights is None:
                    weights_t = torch.ones(m, dtype=losses[0].dtype, device=self.device)
                else:
                    weights_t = task_weights.to(self.device, dtype=losses[0].dtype)
                    if weights_t.numel() != m:
                        raise ValueError(f"Expected {m} task weights, got {weights_t.numel()}.")
                total_loss = sum(weights_t[i] * losses[i] for i in range(m))

                need_recompute = self.exact_upgrad or (step_idx % self.recompute_interval == 0) or (self.prev_lambda is None)

                if need_recompute:
                    if self.stochastic_solver is not None and m > self.stochastic_solver.subset_size:
                        sampled_indices = self.stochastic_solver.sample_indices(m)
                        objective_indices = sampled_indices
                    else:
                        sampled_indices = list(range(m))
                        objective_indices = sampled_indices

                    head_grads: tuple[torch.Tensor | None, ...] = tuple()
                    if head_params:
                        head_grads = torch.autograd.grad(total_loss, head_params, retain_graph=True, allow_unused=True)

                    independent_grads = []
                    for grad_pos, objective_idx in enumerate(objective_indices):
                        grads = torch.autograd.grad(
                            losses[objective_idx],
                            shared_params,
                            retain_graph=grad_pos < len(objective_indices) - 1 or bool(head_params),
                            allow_unused=True,
                        )
                        flat_grad = _flatten_gradient_list(grads, shared_params)
                        if scaffold_correction is not None:
                            scaffold_ratio_history.append(float(torch.norm(scaffold_correction, p=2).item() / (torch.norm(flat_grad, p=2).item() + 1e-12)))
                            flat_grad = flat_grad + scaffold_correction
                        if self.shared_prox_mu > 0:
                            prox_offset = flatten_parameters(shared_params) - theta_init_shared
                            prox_term = self.shared_prox_mu * prox_offset
                            prox_ratio_history.append(float(torch.norm(prox_term, p=2).item() / (torch.norm(flat_grad, p=2).item() + 1e-12)))
                            flat_grad = flat_grad + prox_term
                        independent_grads.append(flat_grad)

                    jacobian = torch.stack(independent_grads, dim=0)
                    last_raw_jacobian = jacobian.detach().clone()
                    jacobian = self._normalize_jacobian(jacobian)
                    objective_weights = weights_t[objective_indices].unsqueeze(1)
                    jacobian = jacobian * objective_weights

                    if self.stochastic_solver is not None and m > self.stochastic_solver.subset_size:
                        direction, sampled_indices = self.stochastic_solver.solve_sampled(
                            jacobian,
                            total_objectives=m,
                            sampled_indices=sampled_indices,
                        )
                        final_lambda = self.stochastic_solver.last_lambda
                    else:
                        direction, final_lambda = aggregator.solve(jacobian)
                        sampled_indices = list(range(m))

                    if self.adaptive_rescaling is not None:
                        direction = self.adaptive_rescaling(direction, jacobian)
                        rescale_factor = self.adaptive_rescaling.last_scale
                        last_rescale_factor = rescale_factor

                    if cone_basis is not None:
                        direction = _apply_cone_basis_alignment(direction, jacobian)
                    else:
                        direction = _apply_cone_alignment(direction, jacobian)

                    avg_cosine_sim = compute_avg_cosine_sim(jacobian)
                    last_avg_cosine_sim = avg_cosine_sim
                    if self.conflict_aware_momentum:
                        momentum_direction = self.local_momentum.update(direction, jacobian=jacobian)
                    else:
                        momentum_direction = self.local_momentum.update(direction)
                else:
                    if self.prev_lambda is not None:
                        lam = self.prev_lambda.detach()
                        L_total = sum(lam[i] * weights_t[i] * losses[i] for i in range(m))
                        shared_grads = torch.autograd.grad(L_total, shared_params, retain_graph=bool(head_params), allow_unused=True)
                        direction = _flatten_gradient_list(shared_grads, shared_params)
                        if scaffold_correction is not None:
                            scaffold_ratio_history.append(float(torch.norm(scaffold_correction, p=2).item() / (torch.norm(direction, p=2).item() + 1e-12)))
                            direction = direction + scaffold_correction
                        if self.shared_prox_mu > 0:
                            prox_offset = flatten_parameters(shared_params) - theta_init_shared
                            prox_term = self.shared_prox_mu * prox_offset
                            prox_ratio_history.append(float(torch.norm(prox_term, p=2).item() / (torch.norm(direction, p=2).item() + 1e-12)))
                            direction = direction + prox_term
                        head_grads = torch.autograd.grad(total_loss, head_params, allow_unused=True) if head_params else tuple()
                        if self.adaptive_rescaling is not None:
                            direction = direction * last_rescale_factor
                        rescale_factor = last_rescale_factor
                        if cone_basis is None:
                            direction = _reuse_last_projected_reference(direction)
                    else:
                        if self.stochastic_solver is not None and m > self.stochastic_solver.subset_size:
                            sampled_indices = self.stochastic_solver.sample_indices(m)
                            objective_indices = sampled_indices
                        else:
                            sampled_indices = list(range(m))
                            objective_indices = sampled_indices

                        head_grads = torch.autograd.grad(total_loss, head_params, retain_graph=True, allow_unused=True) if head_params else tuple()
                        independent_grads = []
                        for grad_pos, objective_idx in enumerate(objective_indices):
                            grads = torch.autograd.grad(
                                losses[objective_idx],
                                shared_params,
                                retain_graph=grad_pos < len(objective_indices) - 1 or bool(head_params),
                                allow_unused=True,
                            )
                            flat_grad = _flatten_gradient_list(grads, shared_params)
                            if scaffold_correction is not None:
                                scaffold_ratio_history.append(float(torch.norm(scaffold_correction, p=2).item() / (torch.norm(flat_grad, p=2).item() + 1e-12)))
                                flat_grad = flat_grad + scaffold_correction
                            if self.shared_prox_mu > 0:
                                prox_offset = flatten_parameters(shared_params) - theta_init_shared
                                prox_term = self.shared_prox_mu * prox_offset
                                prox_ratio_history.append(float(torch.norm(prox_term, p=2).item() / (torch.norm(flat_grad, p=2).item() + 1e-12)))
                                flat_grad = flat_grad + prox_term
                            independent_grads.append(flat_grad)

                        jacobian = torch.stack(independent_grads, dim=0)
                        last_raw_jacobian = jacobian.detach().clone()
                        jacobian = self._normalize_jacobian(jacobian)
                        objective_weights = weights_t[objective_indices].unsqueeze(1)
                        jacobian = jacobian * objective_weights

                        if self.stochastic_solver is not None and m > self.stochastic_solver.subset_size:
                            direction, sampled_indices = self.stochastic_solver.solve_sampled(
                                jacobian,
                                total_objectives=m,
                                sampled_indices=sampled_indices,
                            )
                            final_lambda = self.stochastic_solver.last_lambda
                        else:
                            direction, final_lambda = aggregator.solve(jacobian)
                            sampled_indices = list(range(m))

                        if self.adaptive_rescaling is not None:
                            direction = self.adaptive_rescaling(direction, jacobian)
                            rescale_factor = self.adaptive_rescaling.last_scale
                            last_rescale_factor = rescale_factor

                        if cone_basis is not None:
                            direction = _apply_cone_basis_alignment(direction, jacobian)
                        else:
                            direction = _apply_cone_alignment(direction, jacobian)

                        avg_cosine_sim = compute_avg_cosine_sim(jacobian)
                        last_avg_cosine_sim = avg_cosine_sim
                    momentum_direction = self.local_momentum.update(direction)

                rescale_history.append(last_rescale_factor)
                cosine_history.append(last_avg_cosine_sim)
                _apply_flat_direction(shared_params, momentum_direction, self.learning_rate)
                _apply_gradients(head_params, head_grads, self.learning_rate)
                if final_lambda is not None:
                    self.prev_lambda = final_lambda.detach().clone()
                step_idx += 1

        theta_final = flatten_parameters(model.parameters())
        delta_theta = theta_final - theta_init
        shared_delta_theta = flatten_parameters(shared_params) - theta_init_shared
        compute_time = time.time() - start
        control_delta_shared = None
        if shared_control_global is not None and step_idx > 0:
            local_old = self.shared_control_local.to(self.device, dtype=theta_init_shared.dtype)
            global_control = shared_control_global.to(self.device, dtype=theta_init_shared.dtype)
            local_new = local_old - global_control - shared_delta_theta / max(step_idx * self.learning_rate, 1e-12)
            control_delta_shared = (local_new - local_old).detach().clone()
            self.shared_control_local = local_new.detach().clone().cpu()

        align_scores = None
        if self.upload_align_scores and last_raw_jacobian is not None:
            # Positive alignment means the local update is predicted to reduce the objective.
            align_scores = -(last_raw_jacobian @ shared_delta_theta)

        avg_rescale_factor = sum(rescale_history) / len(rescale_history) if rescale_history else last_rescale_factor
        avg_cosine_sim = sum(cosine_history) / len(cosine_history) if cosine_history else last_avg_cosine_sim
        avg_prox_ratio = sum(prox_ratio_history) / len(prox_ratio_history) if prox_ratio_history else 0.0
        avg_scaffold_ratio = sum(scaffold_ratio_history) / len(scaffold_ratio_history) if scaffold_ratio_history else 0.0
        avg_cone_margin = sum(cone_margin_history) / len(cone_margin_history) if cone_margin_history else 0.0
        avg_cone_cosine = sum(cone_cosine_history) / len(cone_cosine_history) if cone_cosine_history else 0.0

        return ClientResult(
            client_id=self.client_id,
            delta_theta=delta_theta.detach().clone(),
            shared_delta_theta=shared_delta_theta.detach().clone(),
            control_delta_shared=control_delta_shared,
            num_examples=self.num_examples,
            compute_time=compute_time,
            num_local_epochs=self.local_epochs,
            final_lambda=final_lambda,
            rescale_factor=avg_rescale_factor,
            sampled_indices=sampled_indices,
            avg_cosine_sim=avg_cosine_sim,
            avg_prox_ratio=avg_prox_ratio,
            avg_scaffold_ratio=avg_scaffold_ratio,
            avg_cone_margin=avg_cone_margin,
            avg_cone_cosine=avg_cone_cosine,
            jacobian=last_raw_jacobian.detach().clone() if last_raw_jacobian is not None else None,
            align_scores=align_scores.detach().clone() if align_scores is not None else None,
        )

    def evaluate_objectives(self, model: nn.Module, objective_fn: ObjectiveFn) -> list[float]:
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        model.eval()
        all_values = None
        total_weight = 0
        with torch.no_grad():
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                predictions = model(batch_inputs)
                values = objective_fn(predictions, batch_targets, batch_inputs)
                stacked = torch.stack([v.detach() for v in values])
                batch_values = stacked.reshape(stacked.shape[0], -1).mean(dim=1)
                batch_weight = int(batch_inputs.shape[0])
                if all_values is None:
                    all_values = batch_values * batch_weight
                else:
                    all_values.add_(batch_values, alpha=batch_weight)
                total_weight += batch_weight
        model.train()
        if all_values is None or total_weight == 0:
            return [float("nan")]
        means = all_values / total_weight
        return [float(v.item()) for v in means]
