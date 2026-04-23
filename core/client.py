from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ObjectiveFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], list[torch.Tensor]]


@dataclass
class ClientResult:
    client_id: int
    jacobian: torch.Tensor
    num_examples: int
    compute_time: float = 0.0
    serialize_time: float = 0.0
    upload_bytes: int = 0
    peak_memory_mb: float = 0.0


@dataclass
class VectorClientResult:
    client_id: int
    vector: torch.Tensor
    num_examples: int
    compute_time: float = 0.0
    serialize_time: float = 0.0
    upload_bytes: int = 0
    peak_memory_mb: float = 0.0


def flatten_gradients(parameters) -> torch.Tensor:
    chunks = []
    for parameter in parameters:
        if parameter.grad is None:
            chunks.append(torch.zeros_like(parameter).reshape(-1))
        else:
            chunks.append(parameter.grad.detach().reshape(-1).clone())
    return torch.cat(chunks)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _measure_peak_memory() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


class FedJDClient:
    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        batch_size: int,
        device: torch.device,
        use_full_loader: bool = False,
        local_epochs: int = 1,
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.use_full_loader = use_full_loader
        self.local_epochs = local_epochs

    @property
    def num_examples(self) -> int:
        return len(self.dataset)

    def compute_jacobian(self, model: nn.Module, objective_fn: ObjectiveFn) -> ClientResult:
        import time

        start = time.time()

        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        jacobian_sum = None
        total_weight = 0
        epoch_count = self.local_epochs if self.use_full_loader else 1

        for _ in range(epoch_count):
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                model.zero_grad(set_to_none=True)
                predictions = model(batch_inputs)
                objective_values = objective_fn(predictions, batch_targets, batch_inputs)

                rows = []
                for index, objective in enumerate(objective_values):
                    retain_graph = index < len(objective_values) - 1
                    model.zero_grad(set_to_none=True)
                    objective.backward(retain_graph=retain_graph)
                    rows.append(flatten_gradients(model.parameters()))

                batch_jacobian = torch.stack(rows, dim=0)
                model.zero_grad(set_to_none=True)

                batch_weight = int(batch_inputs.shape[0])
                if jacobian_sum is None:
                    jacobian_sum = batch_jacobian * batch_weight
                else:
                    jacobian_sum.add_(batch_jacobian, alpha=batch_weight)
                total_weight += batch_weight

                if not self.use_full_loader:
                    break

        if jacobian_sum is None or total_weight == 0:
            raise RuntimeError("No batch available to compute Jacobian.")

        jacobian = jacobian_sum / total_weight
        compute_time = time.time() - start

        serialize_start = time.time()
        upload_bytes = jacobian.numel() * jacobian.element_size()
        serialize_time = time.time() - serialize_start

        peak_memory_mb = _measure_peak_memory()

        return ClientResult(
            client_id=self.client_id,
            jacobian=jacobian,
            num_examples=self.num_examples,
            compute_time=compute_time,
            serialize_time=serialize_time,
            upload_bytes=upload_bytes,
            peak_memory_mb=peak_memory_mb,
        )

    def compute_weighted_gradient(
        self,
        model: nn.Module,
        objective_fn: ObjectiveFn,
        weights: torch.Tensor | None = None,
    ) -> VectorClientResult:
        import time

        start = time.time()

        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        gradient_sum = None
        total_weight = 0
        epoch_count = self.local_epochs if self.use_full_loader else 1

        for _ in range(epoch_count):
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                model.zero_grad(set_to_none=True)
                predictions = model(batch_inputs)
                objective_values = objective_fn(predictions, batch_targets, batch_inputs)
                if weights is None:
                    batch_weights = torch.ones(len(objective_values), device=self.device, dtype=objective_values[0].dtype) / max(len(objective_values), 1)
                else:
                    batch_weights = weights.to(self.device, dtype=objective_values[0].dtype)

                weighted_loss = sum(batch_weights[idx] * objective_values[idx] for idx in range(len(objective_values)))
                weighted_loss.backward()
                batch_gradient = flatten_gradients(model.parameters())
                model.zero_grad(set_to_none=True)

                batch_weight = int(batch_inputs.shape[0])
                if gradient_sum is None:
                    gradient_sum = batch_gradient * batch_weight
                else:
                    gradient_sum.add_(batch_gradient, alpha=batch_weight)
                total_weight += batch_weight

                if not self.use_full_loader:
                    break

        if gradient_sum is None or total_weight == 0:
            raise RuntimeError("No batch available to compute weighted gradient.")

        gradient = gradient_sum / total_weight
        compute_time = time.time() - start

        serialize_start = time.time()
        upload_bytes = gradient.numel() * gradient.element_size()
        serialize_time = time.time() - serialize_start

        peak_memory_mb = _measure_peak_memory()

        return VectorClientResult(
            client_id=self.client_id,
            vector=gradient,
            num_examples=self.num_examples,
            compute_time=compute_time,
            serialize_time=serialize_time,
            upload_bytes=upload_bytes,
            peak_memory_mb=peak_memory_mb,
        )

    def full_dataset_objectives(
        self, model: nn.Module, objective_fn: ObjectiveFn
    ) -> list[float]:
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
