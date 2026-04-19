from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedjd.aggregators import MinNormAggregator
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
        rescaling_max_scale: float = 10.0,
        minnorm_max_iters: int = 250,
        minnorm_lr: float = 0.1,
        conflict_aware_momentum: bool = False,
        momentum_min_beta: float = 0.1,
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.minnorm_max_iters = minnorm_max_iters
        self.minnorm_lr = minnorm_lr

        self.prev_lambda: torch.Tensor | None = None
        self.local_momentum = LocalMomentum(
            beta=local_momentum_beta,
            conflict_aware=conflict_aware_momentum,
            min_beta=momentum_min_beta,
        )
        self.adaptive_rescaling = AdaptiveRescaling(epsilon=1e-8, max_scale=rescaling_max_scale) if use_adaptive_rescaling else None
        self.stochastic_solver = StochasticGramianSolver(
            subset_size=stochastic_subset_size,
            max_iters=minnorm_max_iters,  # 使用配置的max_iters（默认250），早停机制会自动提前停止
            lr=minnorm_lr,
            seed=stochastic_seed,
        ) if use_stochastic_gramian else None
        self.conflict_aware_momentum = conflict_aware_momentum

    @property
    def num_examples(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _compute_dynamic_iters(m: int) -> int:
        """根据目标数动态调整QP迭代次数，平衡速度与收敛质量"""
        if m <= 2:
            return 50
        elif m <= 5:
            return 100
        elif m <= 8:
            return 200
        else:
            return 250

    def local_update(self, model: nn.Module, objective_fn: ObjectiveFn) -> ClientResult:
        start = time.time()
        theta_init = flatten_parameters(model.parameters()).clone()
        m = None
        final_lambda = None
        rescale_factor = 1.0
        sampled_indices = None
        avg_cosine_sim = 0.0

        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.local_epochs):
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                model.zero_grad(set_to_none=True)
                predictions = model(batch_inputs)
                losses = objective_fn(predictions, batch_targets, batch_inputs)
                if m is None:
                    m = len(losses)
                    self._dynamic_minnorm_iters = self._compute_dynamic_iters(m)
                    
                    # 更新stochastic_solver的max_iters（如果存在）
                    if self.stochastic_solver is not None:
                        self.stochastic_solver.max_iters = self._dynamic_minnorm_iters

                if self.prev_lambda is not None:
                    lambda_weights = self.prev_lambda.detach().to(self.device)
                else:
                    lambda_weights = torch.ones(m, device=self.device) / m

                L_total = sum(lambda_weights[i] * losses[i] for i in range(m))
                model.zero_grad(set_to_none=True)
                L_total.backward(retain_graph=True)

                independent_grads = []
                for i in range(m):
                    model.zero_grad(set_to_none=True)
                    retain = i < m - 1
                    losses[i].backward(retain_graph=retain)
                    independent_grads.append(flatten_gradients(model.parameters()))

                jacobian = torch.stack(independent_grads, dim=0)
                model.zero_grad(set_to_none=True)

                if self.stochastic_solver is not None and m > self.stochastic_solver.subset_size:
                    direction, sampled_indices = self.stochastic_solver.solve(jacobian)
                    final_lambda = self.stochastic_solver.last_lambda
                else:
                    aggregator = MinNormAggregator(max_iters=self._dynamic_minnorm_iters, lr=self.minnorm_lr)
                    direction = aggregator(jacobian)
                    sampled_indices = list(range(m))
                    # 直接获取lambda：从方向重建（因为MinNormAggregator不返回lambda）
                    # 这里简化处理，直接用均匀权重
                    final_lambda = torch.ones(m, device=jacobian.device) / m

                if self.adaptive_rescaling is not None:
                    direction = self.adaptive_rescaling(direction, jacobian)
                    rescale_factor = self.adaptive_rescaling.last_scale

                if self.conflict_aware_momentum:
                    avg_cosine_sim = compute_avg_cosine_sim(jacobian)
                    momentum_direction = self.local_momentum.update(direction, jacobian=jacobian)
                else:
                    momentum_direction = self.local_momentum.update(direction)

                current_flat = flatten_parameters(model.parameters())
                current_flat = current_flat - self.learning_rate * momentum_direction
                assign_flat_parameters(model.parameters(), current_flat)

                self.prev_lambda = final_lambda

        theta_final = flatten_parameters(model.parameters())
        delta_theta = theta_final - theta_init

        compute_time = time.time() - start

        return ClientResult(
            client_id=self.client_id,
            delta_theta=delta_theta.detach().clone(),
            num_examples=self.num_examples,
            compute_time=compute_time,
            num_local_epochs=self.local_epochs,
            final_lambda=final_lambda,
            rescale_factor=rescale_factor,
            sampled_indices=sampled_indices,
            avg_cosine_sim=avg_cosine_sim,
        )

    def evaluate_objectives(self, model: nn.Module, objective_fn: ObjectiveFn) -> list[float]:
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        model.eval()
        all_values = None
        with torch.no_grad():
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                predictions = model(batch_inputs)
                values = objective_fn(predictions, batch_targets, batch_inputs)
                stacked = torch.stack([v.detach() for v in values])
                if all_values is None:
                    all_values = stacked
                else:
                    all_values = torch.cat([all_values, stacked], dim=1)
        model.train()
        if all_values is None:
            return [float("nan")]
        means = all_values.mean(dim=1)
        return [float(v.item()) for v in means]
