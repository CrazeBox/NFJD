from __future__ import annotations

import time
from dataclasses import dataclass
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


@dataclass
class ClientResult:
    client_id: int
    delta_theta: torch.Tensor
    num_examples: int
    compute_time: float = 0.0
    num_local_epochs: int = 1
    final_lambda: torch.Tensor | None = None
    rescale_factor: float = 1.0
    sampled_indices: list[int] | None = None
    avg_cosine_sim: float = 0.0
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

        self.use_mixed_precision = use_mixed_precision and device.type == "cuda"

        self.prev_lambda: torch.Tensor | None = None
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

    def local_update(self, model: nn.Module, objective_fn: ObjectiveFn) -> ClientResult:
        start = time.time()
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
        last_avg_cosine_sim = 0.0

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
                    aggregator = UPGradAggregator(max_iters=dynamic_iters, lr=self.minnorm_lr)
                    if self.stochastic_solver is not None:
                        self.stochastic_solver.max_iters = dynamic_iters

                need_recompute = (step_idx % self.recompute_interval == 0) or (self.prev_lambda is None)

                if need_recompute:
                    if self.stochastic_solver is not None and m > self.stochastic_solver.subset_size:
                        sampled_indices = self.stochastic_solver.sample_indices(m)
                        objective_indices = sampled_indices
                    else:
                        sampled_indices = list(range(m))
                        objective_indices = sampled_indices

                    independent_grads = []
                    for grad_pos, objective_idx in enumerate(objective_indices):
                        model.zero_grad(set_to_none=True)
                        retain = grad_pos < len(objective_indices) - 1
                        losses[objective_idx].backward(retain_graph=retain)
                        independent_grads.append(flatten_gradients(model.parameters()))

                    jacobian = torch.stack(independent_grads, dim=0)
                    model.zero_grad(set_to_none=True)

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

                    avg_cosine_sim = compute_avg_cosine_sim(jacobian)
                    last_avg_cosine_sim = avg_cosine_sim
                    if self.conflict_aware_momentum:
                        momentum_direction = self.local_momentum.update(direction, jacobian=jacobian)
                    else:
                        momentum_direction = self.local_momentum.update(direction)
                else:
                    if self.prev_lambda is not None:
                        lam = self.prev_lambda.detach()
                        L_total = sum(lam[i] * losses[i] for i in range(m))
                        model.zero_grad(set_to_none=True)
                        L_total.backward()
                        direction = flatten_gradients(model.parameters())
                        model.zero_grad(set_to_none=True)
                        if self.adaptive_rescaling is not None:
                            direction = direction * last_rescale_factor
                        rescale_factor = last_rescale_factor
                    else:
                        if self.stochastic_solver is not None and m > self.stochastic_solver.subset_size:
                            sampled_indices = self.stochastic_solver.sample_indices(m)
                            objective_indices = sampled_indices
                        else:
                            sampled_indices = list(range(m))
                            objective_indices = sampled_indices

                        independent_grads = []
                        for grad_pos, objective_idx in enumerate(objective_indices):
                            model.zero_grad(set_to_none=True)
                            retain = grad_pos < len(objective_indices) - 1
                            losses[objective_idx].backward(retain_graph=retain)
                            independent_grads.append(flatten_gradients(model.parameters()))

                        jacobian = torch.stack(independent_grads, dim=0)
                        model.zero_grad(set_to_none=True)

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

                        avg_cosine_sim = compute_avg_cosine_sim(jacobian)
                        last_avg_cosine_sim = avg_cosine_sim
                    momentum_direction = self.local_momentum.update(direction)

                rescale_history.append(last_rescale_factor)
                cosine_history.append(last_avg_cosine_sim)
                add_flat_update_(model.parameters(), momentum_direction, alpha=-self.learning_rate)
                if final_lambda is not None:
                    self.prev_lambda = final_lambda.detach().clone()
                step_idx += 1

        theta_final = flatten_parameters(model.parameters())
        delta_theta = theta_final - theta_init
        compute_time = time.time() - start

        align_scores = None
        if 'jacobian' in locals() and jacobian is not None:
            # Positive alignment means the local update is predicted to reduce the objective.
            align_scores = -(jacobian @ delta_theta)

        avg_rescale_factor = sum(rescale_history) / len(rescale_history) if rescale_history else last_rescale_factor
        avg_cosine_sim = sum(cosine_history) / len(cosine_history) if cosine_history else last_avg_cosine_sim

        return ClientResult(
            client_id=self.client_id,
            delta_theta=delta_theta.detach().clone(),
            num_examples=self.num_examples,
            compute_time=compute_time,
            num_local_epochs=self.local_epochs,
            final_lambda=final_lambda,
            rescale_factor=avg_rescale_factor,
            sampled_indices=sampled_indices,
            avg_cosine_sim=avg_cosine_sim,
            jacobian=jacobian.detach().clone() if 'jacobian' in locals() else None,
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
