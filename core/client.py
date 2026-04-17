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
    def __init__(self, client_id: int, dataset: Dataset, batch_size: int, device: torch.device) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

    @property
    def num_examples(self) -> int:
        return len(self.dataset)

    def compute_jacobian(self, model: nn.Module, objective_fn: ObjectiveFn) -> ClientResult:
        import time

        start = time.time()

        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        batch_inputs, batch_targets = next(iter(loader))
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

        jacobian = torch.stack(rows, dim=0)
        model.zero_grad(set_to_none=True)

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

    def full_dataset_objectives(
        self, model: nn.Module, objective_fn: ObjectiveFn
    ) -> list[float]:
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
