from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from fedjd.aggregators import JacobianAggregator
from fedjd.compressors import JacobianCompressor, NoCompressor

from .client import FedJDClient, ObjectiveFn, _measure_peak_memory
from .evaluation import evaluate_objectives_on_dataset


@dataclass
class RoundStats:
    round_idx: int
    sampled_client_ids: list[int]
    num_sampled_clients: int
    objective_values: list[float]
    direction_norm: float
    jacobian_norm: float
    round_time: float = 0.0
    upload_bytes: int = 0
    download_bytes: int = 0
    nan_inf_count: int = 0
    client_compute_time: float = 0.0
    client_serialize_time: float = 0.0
    aggregation_time: float = 0.0
    direction_time: float = 0.0
    update_time: float = 0.0
    client_peak_memory_mb: float = 0.0
    server_peak_memory_mb: float = 0.0
    jacobian_upload_per_client: int = 0
    gradient_upload_per_client: int = 0
    jacobian_vs_gradient_ratio: float = 0.0
    compressor_name: str = "none"
    compressed_upload_per_client: int = 0
    compression_ratio: float = 1.0
    is_full_sync_round: bool = True
    local_steps: int = 1
    method_name: str = "fedjd"


def flatten_parameters(parameters) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def assign_flat_parameters(parameters, flat_vector: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        parameter.data.copy_(flat_vector[offset : offset + size].view_as(parameter))
        offset += size


def _count_nan_inf(tensor: torch.Tensor) -> int:
    return int(torch.isnan(tensor).sum().item() + torch.isinf(tensor).sum().item())


class FedJDServer:
    def __init__(
        self,
        model: nn.Module,
        clients: list[FedJDClient],
        aggregator: JacobianAggregator,
        objective_fn: ObjectiveFn,
        participation_rate: float,
        learning_rate: float,
        device: torch.device,
        compressor: JacobianCompressor | None = None,
        full_sync_interval: int = 1,
        local_steps: int = 1,
        eval_dataset=None,
    ) -> None:
        self.model = model.to(device)
        self.clients = clients
        self.aggregator = aggregator
        self.objective_fn = objective_fn
        self.participation_rate = participation_rate
        self.learning_rate = learning_rate
        self.device = device
        self.compressor = compressor or NoCompressor()
        self.full_sync_interval = full_sync_interval
        self.local_steps = local_steps
        self.eval_dataset = eval_dataset
        self._last_direction: torch.Tensor | None = None

    def sample_clients(self) -> list[FedJDClient]:
        sample_size = max(int(len(self.clients) * self.participation_rate), 1)
        return random.sample(self.clients, sample_size)

    def _clone_model(self) -> nn.Module:
        cloned = copy.deepcopy(self.model)
        cloned.to(self.device)
        return cloned

    def _is_full_sync_round(self, round_idx: int) -> bool:
        if self.full_sync_interval <= 1:
            return True
        return round_idx % self.full_sync_interval == 0

    def run_round(self, round_idx: int) -> RoundStats:
        round_start = time.time()
        is_full_sync = self._is_full_sync_round(round_idx)
        current_flat = flatten_parameters(self.model.parameters())
        model_bytes = current_flat.numel() * current_flat.element_size()

        if not is_full_sync and self._last_direction is not None:
            direction = self._last_direction
            update_start = time.time()
            for _ in range(self.local_steps):
                current_flat = current_flat - self.learning_rate * direction
            assign_flat_parameters(self.model.parameters(), current_flat)
            update_time = time.time() - update_start

            round_time = time.time() - round_start
            return RoundStats(
                round_idx=round_idx,
                sampled_client_ids=[],
                num_sampled_clients=0,
                objective_values=self.evaluate_global_objectives(),
                direction_norm=float(torch.norm(direction, p=2).item()),
                jacobian_norm=0.0,
                round_time=round_time,
                upload_bytes=0,
                download_bytes=0,
                nan_inf_count=0,
                client_compute_time=0.0,
                client_serialize_time=0.0,
                aggregation_time=0.0,
                direction_time=0.0,
                update_time=update_time,
                client_peak_memory_mb=0.0,
                server_peak_memory_mb=_measure_peak_memory(),
                jacobian_upload_per_client=0,
                gradient_upload_per_client=model_bytes,
                jacobian_vs_gradient_ratio=0.0,
                compressor_name=self.compressor.name,
                compressed_upload_per_client=0,
                compression_ratio=0.0,
                is_full_sync_round=False,
                local_steps=self.local_steps,
                method_name="fedjd",
            )

        sampled_clients = self.sample_clients()
        total_examples = sum(client.num_examples for client in sampled_clients)

        client_start = time.time()
        aggregated_jacobian = None
        sampled_client_ids = []
        total_upload_bytes = 0
        total_nan_inf = 0
        total_serialize_time = 0.0
        max_client_peak_mem = 0.0
        per_client_upload = 0
        per_client_compressed_upload = 0

        for client in sampled_clients:
            result = client.compute_jacobian(self._clone_model(), self.objective_fn)
            weight = result.num_examples / total_examples

            compressed_jac, comp_meta = self.compressor.compress(result.jacobian)
            decompressed_jac = self.compressor.decompress(compressed_jac, comp_meta)

            compressed_bytes = compressed_jac.numel() * compressed_jac.element_size()
            if "indices" in comp_meta and isinstance(comp_meta["indices"], torch.Tensor):
                compressed_bytes += comp_meta["indices"].numel() * comp_meta["indices"].element_size()

            weighted_jac = decompressed_jac.to(self.device) * weight
            if aggregated_jacobian is None:
                aggregated_jacobian = weighted_jac
                per_client_upload = result.upload_bytes
                per_client_compressed_upload = compressed_bytes
            else:
                aggregated_jacobian.add_(weighted_jac)
            sampled_client_ids.append(result.client_id)
            total_upload_bytes += compressed_bytes
            total_nan_inf += _count_nan_inf(decompressed_jac)
            total_serialize_time += result.serialize_time
            if result.peak_memory_mb > max_client_peak_mem:
                max_client_peak_mem = result.peak_memory_mb

        client_compute_time = time.time() - client_start

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if aggregated_jacobian is None:
            raise RuntimeError("No client contributed a Jacobian.")

        agg_start = time.time()
        aggregation_time = time.time() - agg_start

        dir_start = time.time()
        direction = self.aggregator(aggregated_jacobian)
        direction_time = time.time() - dir_start
        self._last_direction = direction.detach().clone()

        total_nan_inf += _count_nan_inf(direction)

        update_start = time.time()
        for _ in range(self.local_steps):
            current_flat = current_flat - self.learning_rate * direction
        assign_flat_parameters(self.model.parameters(), current_flat)
        update_time = time.time() - update_start

        download_bytes = model_bytes * len(sampled_clients)

        num_params = current_flat.numel()
        num_objectives = aggregated_jacobian.shape[0]
        gradient_upload_per_client = num_params * current_flat.element_size()
        jacobian_vs_gradient_ratio = per_client_upload / max(gradient_upload_per_client, 1)
        compression_ratio = per_client_compressed_upload / max(per_client_upload, 1)

        server_peak_mem = _measure_peak_memory()
        round_time = time.time() - round_start

        return RoundStats(
            round_idx=round_idx,
            sampled_client_ids=sampled_client_ids,
            num_sampled_clients=len(sampled_clients),
            objective_values=self.evaluate_global_objectives(),
            direction_norm=float(torch.norm(direction, p=2).item()),
            jacobian_norm=float(torch.norm(aggregated_jacobian, p="fro").item()),
            round_time=round_time,
            upload_bytes=total_upload_bytes,
            download_bytes=download_bytes,
            nan_inf_count=total_nan_inf,
            client_compute_time=client_compute_time,
            client_serialize_time=total_serialize_time,
            aggregation_time=aggregation_time,
            direction_time=direction_time,
            update_time=update_time,
            client_peak_memory_mb=max_client_peak_mem,
            server_peak_memory_mb=server_peak_mem,
            jacobian_upload_per_client=per_client_upload,
            gradient_upload_per_client=gradient_upload_per_client,
            jacobian_vs_gradient_ratio=jacobian_vs_gradient_ratio,
            compressor_name=self.compressor.name,
            compressed_upload_per_client=per_client_compressed_upload,
            compression_ratio=compression_ratio,
            is_full_sync_round=True,
            local_steps=self.local_steps,
            method_name="fedjd",
        )

    def evaluate_global_objectives(self) -> list[float]:
        if self.eval_dataset is not None:
            return evaluate_objectives_on_dataset(self.model, self.eval_dataset, self.objective_fn, self.device)

        total_examples = sum(client.num_examples for client in self.clients)
        running = None
        self.model.eval()
        with torch.no_grad():
            for client in self.clients:
                loader = DataLoader(client.dataset, batch_size=256, shuffle=False)
                client_sum = None
                client_weight = 0
                for batch_inputs, batch_targets in loader:
                    batch_inputs = batch_inputs.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    predictions = self.model(batch_inputs)
                    values = self.objective_fn(predictions, batch_targets, batch_inputs)
                    stacked = torch.stack([value.detach() for value in values])
                    batch_values = stacked.reshape(stacked.shape[0], -1).mean(dim=1)
                    batch_size = int(batch_inputs.shape[0])
                    if client_sum is None:
                        client_sum = batch_values * batch_size
                    else:
                        client_sum.add_(batch_values, alpha=batch_size)
                    client_weight += batch_size
                if client_sum is not None and client_weight > 0:
                    avg_values = client_sum / client_weight
                    weight = client.num_examples / total_examples
                    if running is None:
                        running = weight * avg_values
                    else:
                        running.add_(weight * avg_values)
        self.model.train()
        if running is None:
            raise RuntimeError("No clients available for evaluation.")
        return [float(value.item()) for value in running]
