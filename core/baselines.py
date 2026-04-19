from __future__ import annotations

import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .client import FedJDClient, ObjectiveFn, _measure_peak_memory
from .server import RoundStats, _count_nan_inf, assign_flat_parameters, flatten_parameters


def _evaluate_global_objectives(model, clients, objective_fn, device):
    total_examples = sum(c.num_examples for c in clients)
    running = None
    model.eval()
    with torch.no_grad():
        for client in clients:
            loader = DataLoader(client.dataset, batch_size=256, shuffle=False)
            client_values = []
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                predictions = model(batch_inputs)
                values = objective_fn(predictions, batch_targets, batch_inputs)
                stacked = torch.stack([value.detach() for value in values])
                client_values.append(stacked.mean(dim=1))
            if client_values:
                avg_values = torch.stack(client_values).mean(dim=0)
                weight = client.num_examples / total_examples
                if running is None:
                    running = weight * avg_values
                else:
                    running.add_(weight * avg_values)
    model.train()
    if running is None:
        raise RuntimeError("No clients available.")
    return [float(value.item()) for value in running]


class FMGDAServer:
    """FMGDA: Federated MGDA. Uploads full Jacobian (same as FedJD), uses fixed equal weights for direction."""

    def __init__(self, model, clients, objective_fn, participation_rate,
                 learning_rate, device, weights=None):
        self.model = model.to(device)
        self.clients = clients
        self.objective_fn = objective_fn
        self.participation_rate = participation_rate
        self.learning_rate = learning_rate
        self.device = device
        self.weights = weights

    def evaluate_global_objectives(self):
        return _evaluate_global_objectives(self.model, self.clients, self.objective_fn, self.device)

    def sample_clients(self):
        sample_size = max(int(len(self.clients) * self.participation_rate), 1)
        return random.sample(self.clients, sample_size)

    def _clone_model(self):
        import copy
        return copy.deepcopy(self.model).to(self.device)

    def run_round(self, round_idx):
        round_start = time.time()
        sampled = self.sample_clients()
        total_examples = sum(c.num_examples for c in sampled)

        client_start = time.time()
        sampled_ids = []
        total_upload = 0
        total_nan_inf = 0
        max_client_mem = 0.0
        per_client_upload = 0
        all_jacobians = []

        for client in sampled:
            result = client.compute_jacobian(self._clone_model(), self.objective_fn)
            all_jacobians.append(result.jacobian)
            sampled_ids.append(result.client_id)
            total_upload += result.upload_bytes
            total_nan_inf += _count_nan_inf(result.jacobian)
            if result.peak_memory_mb > max_client_mem:
                max_client_mem = result.peak_memory_mb
            if per_client_upload == 0:
                per_client_upload = result.upload_bytes
        client_compute_time = time.time() - client_start

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dir_start = time.time()
        aggregated = torch.zeros_like(all_jacobians[0])
        for jac, client in zip(all_jacobians, sampled):
            w = client.num_examples / total_examples
            aggregated.add_(jac * w)

        m = aggregated.shape[0]
        if self.weights is None:
            weights = torch.ones(m, device=self.device) / m
        else:
            weights = torch.tensor(self.weights, device=self.device, dtype=aggregated.dtype)
        direction = aggregated.T @ weights
        direction_time = time.time() - dir_start
        total_nan_inf += _count_nan_inf(direction)

        update_start = time.time()
        current_flat = flatten_parameters(self.model.parameters())
        next_flat = current_flat - self.learning_rate * direction
        assign_flat_parameters(self.model.parameters(), next_flat)
        update_time = time.time() - update_start

        download_bytes = current_flat.numel() * current_flat.element_size()
        num_params = current_flat.numel()
        grad_upload = num_params * current_flat.element_size()
        round_time = time.time() - round_start

        return RoundStats(
            round_idx=round_idx, sampled_client_ids=sampled_ids,
            num_sampled_clients=len(sampled),
            objective_values=self.evaluate_global_objectives(),
            direction_norm=float(torch.norm(direction, p=2).item()),
            jacobian_norm=float(torch.norm(aggregated, p="fro").item()),
            round_time=round_time, upload_bytes=total_upload,
            download_bytes=download_bytes, nan_inf_count=total_nan_inf,
            client_compute_time=client_compute_time,
            direction_time=direction_time, update_time=update_time,
            client_peak_memory_mb=max_client_mem,
            server_peak_memory_mb=_measure_peak_memory(),
            jacobian_upload_per_client=per_client_upload,
            gradient_upload_per_client=grad_upload,
            jacobian_vs_gradient_ratio=per_client_upload / max(grad_upload, 1),
            method_name="fmgda",
        )


class WeightedSumServer:
    """Weighted Sum: Each client computes weighted gradient locally, uploads only d-dim gradient."""

    def __init__(self, model, clients, objective_fn, participation_rate,
                 learning_rate, device, weights=None):
        self.model = model.to(device)
        self.clients = clients
        self.objective_fn = objective_fn
        self.participation_rate = participation_rate
        self.learning_rate = learning_rate
        self.device = device
        self.weights = weights

    def evaluate_global_objectives(self):
        return _evaluate_global_objectives(self.model, self.clients, self.objective_fn, self.device)

    def sample_clients(self):
        sample_size = max(int(len(self.clients) * self.participation_rate), 1)
        return random.sample(self.clients, sample_size)

    def _clone_model(self):
        import copy
        return copy.deepcopy(self.model).to(self.device)

    def run_round(self, round_idx):
        round_start = time.time()
        sampled = self.sample_clients()
        total_examples = sum(c.num_examples for c in sampled)
        sampled_ids = []

        client_start = time.time()
        total_nan_inf = 0
        max_client_mem = 0.0
        all_weighted_grads = []
        per_client_grad_upload = 0

        m = None
        for client in sampled:
            result = client.compute_jacobian(self._clone_model(), self.objective_fn)
            sampled_ids.append(result.client_id)
            total_nan_inf += _count_nan_inf(result.jacobian)
            if result.peak_memory_mb > max_client_mem:
                max_client_mem = result.peak_memory_mb

            if m is None:
                m = result.jacobian.shape[0]
                if self.weights is None:
                    w = torch.ones(m, device=self.device) / m
                else:
                    w = torch.tensor(self.weights, device=self.device, dtype=torch.float32)

            weighted_grad = result.jacobian.T @ w
            all_weighted_grads.append(weighted_grad)
            if per_client_grad_upload == 0:
                per_client_grad_upload = weighted_grad.numel() * weighted_grad.element_size()
        client_compute_time = time.time() - client_start

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dir_start = time.time()
        direction = torch.stack(all_weighted_grads).mean(dim=0)
        direction_time = time.time() - dir_start
        total_nan_inf += _count_nan_inf(direction)

        update_start = time.time()
        current_flat = flatten_parameters(self.model.parameters())
        next_flat = current_flat - self.learning_rate * direction
        assign_flat_parameters(self.model.parameters(), next_flat)
        update_time = time.time() - update_start

        download_bytes = current_flat.numel() * current_flat.element_size()
        num_params = current_flat.numel()
        grad_upload = num_params * current_flat.element_size()
        total_upload = per_client_grad_upload * len(sampled)
        round_time = time.time() - round_start

        return RoundStats(
            round_idx=round_idx, sampled_client_ids=sampled_ids,
            num_sampled_clients=len(sampled),
            objective_values=self.evaluate_global_objectives(),
            direction_norm=float(torch.norm(direction, p=2).item()),
            jacobian_norm=0.0,
            round_time=round_time, upload_bytes=total_upload,
            download_bytes=download_bytes, nan_inf_count=total_nan_inf,
            client_compute_time=client_compute_time,
            direction_time=direction_time, update_time=update_time,
            client_peak_memory_mb=max_client_mem,
            server_peak_memory_mb=_measure_peak_memory(),
            jacobian_upload_per_client=0,
            gradient_upload_per_client=per_client_grad_upload,
            jacobian_vs_gradient_ratio=0.0,
            method_name="weighted_sum",
        )


class DirectionAvgServer:
    """Direction Averaging: Each client averages Jacobian rows locally, uploads only d-dim direction."""

    def __init__(self, model, clients, objective_fn, participation_rate,
                 learning_rate, device):
        self.model = model.to(device)
        self.clients = clients
        self.objective_fn = objective_fn
        self.participation_rate = participation_rate
        self.learning_rate = learning_rate
        self.device = device

    def evaluate_global_objectives(self):
        return _evaluate_global_objectives(self.model, self.clients, self.objective_fn, self.device)

    def sample_clients(self):
        sample_size = max(int(len(self.clients) * self.participation_rate), 1)
        return random.sample(self.clients, sample_size)

    def _clone_model(self):
        import copy
        return copy.deepcopy(self.model).to(self.device)

    def run_round(self, round_idx):
        round_start = time.time()
        sampled = self.sample_clients()
        sampled_ids = []

        client_start = time.time()
        total_nan_inf = 0
        max_client_mem = 0.0
        per_client_dir_upload = 0
        all_directions = []

        m = None
        for client in sampled:
            result = client.compute_jacobian(self._clone_model(), self.objective_fn)
            if m is None:
                m = result.jacobian.shape[0]
            local_dir = result.jacobian.mean(dim=0)
            all_directions.append(local_dir)
            sampled_ids.append(result.client_id)
            total_nan_inf += _count_nan_inf(result.jacobian)
            if result.peak_memory_mb > max_client_mem:
                max_client_mem = result.peak_memory_mb
            if per_client_dir_upload == 0:
                per_client_dir_upload = local_dir.numel() * local_dir.element_size()
        client_compute_time = time.time() - client_start

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dir_start = time.time()
        direction = torch.stack(all_directions).mean(dim=0)
        direction_time = time.time() - dir_start
        total_nan_inf += _count_nan_inf(direction)

        update_start = time.time()
        current_flat = flatten_parameters(self.model.parameters())
        next_flat = current_flat - self.learning_rate * direction
        assign_flat_parameters(self.model.parameters(), next_flat)
        update_time = time.time() - update_start

        download_bytes = current_flat.numel() * current_flat.element_size()
        num_params = current_flat.numel()
        grad_upload = num_params * current_flat.element_size()
        total_upload = per_client_dir_upload * len(sampled)
        round_time = time.time() - round_start

        return RoundStats(
            round_idx=round_idx, sampled_client_ids=sampled_ids,
            num_sampled_clients=len(sampled),
            objective_values=self.evaluate_global_objectives(),
            direction_norm=float(torch.norm(direction, p=2).item()),
            jacobian_norm=0.0,
            round_time=round_time, upload_bytes=total_upload,
            download_bytes=download_bytes, nan_inf_count=total_nan_inf,
            client_compute_time=client_compute_time,
            direction_time=direction_time, update_time=update_time,
            client_peak_memory_mb=max_client_mem,
            server_peak_memory_mb=_measure_peak_memory(),
            jacobian_upload_per_client=0,
            gradient_upload_per_client=per_client_dir_upload,
            jacobian_vs_gradient_ratio=0.0,
            method_name="direction_avg",
        )
