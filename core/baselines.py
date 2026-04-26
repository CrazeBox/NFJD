from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedjd.aggregators import MinNormAggregator, UPGradAggregator

from .client import FedJDClient, ObjectiveFn, _measure_peak_memory, flatten_gradients
from .evaluation import evaluate_objectives_on_dataset
from .server import FedJDServer, RoundStats, _count_nan_inf, assign_flat_parameters, flatten_parameters


def _evaluate_global_objectives(model, clients, objective_fn, device):
    total_examples = sum(c.num_examples for c in clients)
    running = None
    model.eval()
    with torch.no_grad():
        for client in clients:
            loader = DataLoader(client.dataset, batch_size=256, shuffle=False)
            client_sum = None
            client_weight = 0
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                predictions = model(batch_inputs)
                values = objective_fn(predictions, batch_targets, batch_inputs)
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
    model.train()
    if running is None:
        raise RuntimeError("No clients available.")
    return [float(value.item()) for value in running]


@dataclass
class FMGDAClientResult:
    client_id: int
    objective_updates: torch.Tensor
    objective_mask: torch.Tensor
    num_examples: int
    compute_time: float = 0.0
    upload_bytes: int = 0
    peak_memory_mb: float = 0.0


class FMGDAClient:
    """Paper-aligned FMGDA client with one local trajectory per objective."""

    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        batch_size: int,
        device: torch.device,
        learning_rate: float,
        local_epochs: int = 1,
        objective_indices: list[int] | None = None,
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.objective_indices = objective_indices

    @property
    def num_examples(self) -> int:
        return len(self.dataset)

    def compute_objective_updates(
        self,
        model: nn.Module,
        objective_fn: ObjectiveFn,
        num_objectives: int,
    ) -> FMGDAClientResult:
        start = time.time()
        active_objectives = self.objective_indices or list(range(num_objectives))
        if not active_objectives:
            raise ValueError("FMGDAClient requires at least one active objective.")

        local_models = {
            objective_idx: copy.deepcopy(model).to(self.device)
            for objective_idx in active_objectives
        }
        objective_updates = None
        weights = torch.zeros(num_objectives, dtype=torch.float32, device=self.device)

        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.local_epochs):
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                batch_weight = float(batch_inputs.shape[0])

                for objective_idx in active_objectives:
                    local_model = local_models[objective_idx]
                    local_model.zero_grad(set_to_none=True)
                    predictions = local_model(batch_inputs)
                    losses = objective_fn(predictions, batch_targets, batch_inputs)
                    losses[objective_idx].backward()
                    gradient = flatten_gradients(local_model.parameters())

                    if objective_updates is None:
                        objective_updates = torch.zeros(
                            num_objectives,
                            gradient.numel(),
                            dtype=gradient.dtype,
                            device=self.device,
                        )

                    objective_updates[objective_idx].add_(gradient, alpha=batch_weight)
                    weights[objective_idx] += batch_weight

                    next_flat = flatten_parameters(local_model.parameters()) - self.learning_rate * gradient
                    assign_flat_parameters(local_model.parameters(), next_flat)

        if objective_updates is None:
            raise RuntimeError("No batch available to compute FMGDA objective updates.")

        objective_mask = weights > 0
        for objective_idx in active_objectives:
            if weights[objective_idx] > 0:
                objective_updates[objective_idx] /= weights[objective_idx]

        uploaded = objective_updates[objective_mask]
        upload_bytes = uploaded.numel() * uploaded.element_size()

        return FMGDAClientResult(
            client_id=self.client_id,
            objective_updates=objective_updates.detach().clone(),
            objective_mask=objective_mask.detach().clone(),
            num_examples=self.num_examples,
            compute_time=time.time() - start,
            upload_bytes=upload_bytes,
            peak_memory_mb=_measure_peak_memory(),
        )


class FMGDAServer:
    """Paper-aligned FMGDA with client-side local trajectories per objective."""

    def __init__(self, model, clients, objective_fn, participation_rate,
                 learning_rate, device, weights=None, aggregator=None,
                 eval_dataset=None, num_objectives=None):
        self.model = model.to(device)
        self.clients = clients
        self.objective_fn = objective_fn
        self.participation_rate = participation_rate
        self.learning_rate = learning_rate
        self.device = device
        self.weights = weights
        self.aggregator = aggregator or MinNormAggregator(max_iters=250, lr=0.1, max_direction_norm=0.0)
        self.eval_dataset = eval_dataset
        self.num_objectives = num_objectives

    def evaluate_global_objectives(self):
        if self.eval_dataset is not None:
            return evaluate_objectives_on_dataset(self.model, self.eval_dataset, self.objective_fn, self.device)
        return _evaluate_global_objectives(self.model, self.clients, self.objective_fn, self.device)

    def sample_clients(self):
        sample_size = max(int(len(self.clients) * self.participation_rate), 1)
        return random.sample(self.clients, sample_size)

    def _clone_model(self):
        return copy.deepcopy(self.model).to(self.device)

    def run_round(self, round_idx):
        round_start = time.time()
        sampled = self.sample_clients()
        num_objectives = self.num_objectives
        if num_objectives is None:
            num_objectives = len(self.evaluate_global_objectives())
            self.num_objectives = num_objectives

        client_start = time.time()
        sampled_ids = []
        total_upload = 0
        total_nan_inf = 0
        max_client_mem = 0.0
        per_client_upload = 0
        aggregated_updates = None
        aggregated_weights = None
        local_steps = 1
        total_examples = sum(client.num_examples for client in sampled)

        for client in sampled:
            result = client.compute_objective_updates(
                self._clone_model(),
                self.objective_fn,
                num_objectives=num_objectives,
            )
            client_weight = result.num_examples / max(total_examples, 1)
            if aggregated_updates is None:
                aggregated_updates = torch.zeros_like(result.objective_updates)
                aggregated_weights = torch.zeros(
                    result.objective_updates.shape[0],
                    dtype=result.objective_updates.dtype,
                    device=self.device,
                )
                local_steps = getattr(client, "local_epochs", 1)
            mask_indices = torch.nonzero(result.objective_mask.to(self.device), as_tuple=False).flatten()
            if mask_indices.numel() > 0:
                aggregated_updates[mask_indices] += client_weight * result.objective_updates[mask_indices].to(self.device)
                aggregated_weights[mask_indices] += client_weight
            sampled_ids.append(result.client_id)
            total_upload += result.upload_bytes
            total_nan_inf += _count_nan_inf(result.objective_updates)
            if result.peak_memory_mb > max_client_mem:
                max_client_mem = result.peak_memory_mb
            if per_client_upload == 0:
                per_client_upload = result.upload_bytes
        client_compute_time = time.time() - client_start

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if aggregated_updates is None or aggregated_weights is None:
            raise RuntimeError("No client contributed FMGDA objective updates.")

        active_mask = aggregated_weights > 0
        if not active_mask.any():
            raise RuntimeError("No objectives available after FMGDA client aggregation.")

        dir_start = time.time()
        aggregated = aggregated_updates[active_mask] / aggregated_weights[active_mask].unsqueeze(1)
        if self.aggregator is not None:
            direction = self.aggregator(aggregated)
        else:
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

        model_bytes = current_flat.numel() * current_flat.element_size()
        download_bytes = model_bytes * len(sampled)
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
            local_steps=local_steps,
            method_name="fmgda",
        )


class FedMGDAPlusServer(FMGDAServer):
    """FedMGDA+ style local MGDA directions followed by server averaging.

    This matches the practical communication-saving variant: each sampled client
    computes its local per-objective gradient matrix, solves a local MGDA
    problem, uploads one d-dimensional common direction, and the server averages
    these local common directions before updating the global model.
    """

    def __init__(self, *args, aggregator=None, **kwargs):
        super().__init__(*args, aggregator=aggregator or MinNormAggregator(max_iters=250, lr=0.1, max_direction_norm=0.0), **kwargs)

    def run_round(self, round_idx):
        round_start = time.time()
        sampled = self.sample_clients()
        num_objectives = self.num_objectives
        if num_objectives is None:
            num_objectives = len(self.evaluate_global_objectives())
            self.num_objectives = num_objectives

        client_start = time.time()
        sampled_ids = []
        total_upload = 0
        total_nan_inf = 0
        max_client_mem = 0.0
        per_client_upload = 0
        local_steps = 1
        total_examples = sum(client.num_examples for client in sampled)
        local_directions = []

        for client in sampled:
            result = client.compute_objective_updates(
                self._clone_model(),
                self.objective_fn,
                num_objectives=num_objectives,
            )
            active = result.objective_mask.to(self.device)
            if active.any():
                local_jacobian = result.objective_updates.to(self.device)[active]
                local_direction = self.aggregator(local_jacobian)
                local_directions.append((local_direction, result.num_examples / max(total_examples, 1)))
                total_nan_inf += _count_nan_inf(local_direction)
            sampled_ids.append(result.client_id)
            total_upload += result.upload_bytes
            total_nan_inf += _count_nan_inf(result.objective_updates)
            if result.peak_memory_mb > max_client_mem:
                max_client_mem = result.peak_memory_mb
            if per_client_upload == 0:
                per_client_upload = result.upload_bytes
            local_steps = getattr(client, "local_epochs", 1)
        client_compute_time = time.time() - client_start

        if not local_directions:
            raise RuntimeError("No client contributed FedMGDA+ local directions.")

        dir_start = time.time()
        direction = sum(local_dir * weight for local_dir, weight in local_directions)
        direction_time = time.time() - dir_start
        total_nan_inf += _count_nan_inf(direction)

        update_start = time.time()
        current_flat = flatten_parameters(self.model.parameters())
        next_flat = current_flat - self.learning_rate * direction
        assign_flat_parameters(self.model.parameters(), next_flat)
        update_time = time.time() - update_start

        model_bytes = current_flat.numel() * current_flat.element_size()
        download_bytes = model_bytes * len(sampled)
        grad_upload = model_bytes
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
            jacobian_upload_per_client=per_client_upload,
            gradient_upload_per_client=grad_upload,
            jacobian_vs_gradient_ratio=per_client_upload / max(grad_upload, 1),
            local_steps=local_steps,
            method_name="fedmgda_plus",
        )


class FedAvgUPGradServer(FedJDServer):
    """FedAvg-style client Jacobian upload with server-side UPGrad."""

    def __init__(self, model, clients, objective_fn, participation_rate,
                 learning_rate, device, eval_dataset=None):
        super().__init__(
            model=model,
            clients=clients,
            aggregator=UPGradAggregator(max_iters=250, lr=0.1, max_direction_norm=0.0, solver="auto"),
            objective_fn=objective_fn,
            participation_rate=participation_rate,
            learning_rate=learning_rate,
            device=device,
            eval_dataset=eval_dataset,
        )

    def run_round(self, round_idx):
        stats = super().run_round(round_idx)
        stats.method_name = "fedavg_upgrad"
        return stats


@dataclass
class LocalTrainClientResult:
    client_id: int
    delta_theta: torch.Tensor
    num_examples: int
    initial_loss: float = 0.0
    compute_time: float = 0.0
    upload_bytes: int = 0
    peak_memory_mb: float = 0.0


class FedLocalTrainClient:
    """Client that performs ordinary local weighted-sum training and uploads delta theta."""

    def __init__(self, client_id: int, dataset: Dataset, batch_size: int, device: torch.device,
                 learning_rate: float, local_epochs: int = 1) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs

    @property
    def num_examples(self) -> int:
        return len(self.dataset)

    def evaluate_weighted_loss(self, model: nn.Module, objective_fn: ObjectiveFn) -> float:
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        total_loss = 0.0
        total_examples = 0
        model.eval()
        with torch.no_grad():
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                predictions = model(batch_inputs)
                losses = objective_fn(predictions, batch_targets, batch_inputs)
                loss = sum(losses) / max(len(losses), 1)
                batch_size = int(batch_inputs.shape[0])
                total_loss += float(loss.item()) * batch_size
                total_examples += batch_size
        model.train()
        return total_loss / max(total_examples, 1)

    def local_update(self, model: nn.Module, objective_fn: ObjectiveFn) -> LocalTrainClientResult:
        start = time.time()
        initial_loss = self.evaluate_weighted_loss(model, objective_fn)
        theta_init = flatten_parameters(model.parameters()).clone()
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.local_epochs):
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                model.zero_grad(set_to_none=True)
                predictions = model(batch_inputs)
                losses = objective_fn(predictions, batch_targets, batch_inputs)
                total_loss = sum(losses) / max(len(losses), 1)
                total_loss.backward()
                grads = flatten_gradients(model.parameters())
                next_flat = flatten_parameters(model.parameters()) - self.learning_rate * grads
                assign_flat_parameters(model.parameters(), next_flat)

        delta_theta = flatten_parameters(model.parameters()) - theta_init
        upload_bytes = delta_theta.numel() * delta_theta.element_size()
        return LocalTrainClientResult(
            client_id=self.client_id,
            delta_theta=delta_theta.detach().clone(),
            num_examples=self.num_examples,
            initial_loss=initial_loss,
            compute_time=time.time() - start,
            upload_bytes=upload_bytes,
            peak_memory_mb=_measure_peak_memory(),
        )


class FedClientUPGradServer:
    """Server-side UPGrad over client local updates.

    Each sampled client performs ordinary local training and uploads a model
    delta. The server treats the negative client deltas as client-loss descent
    gradients, uses UPGrad to find a common direction across sampled clients,
    and applies that direction to the global model.
    """

    def __init__(self, model, clients, objective_fn, participation_rate,
                 learning_rate, device, eval_dataset=None, aggregator=None,
                 update_scale: float = 1.0, normalize_client_updates: bool = False):
        self.model = model.to(device)
        self.clients = clients
        self.objective_fn = objective_fn
        self.participation_rate = participation_rate
        self.learning_rate = learning_rate
        self.device = device
        self.eval_dataset = eval_dataset
        self.aggregator = aggregator or UPGradAggregator(max_iters=250, lr=0.1, max_direction_norm=0.0, solver="auto")
        self.update_scale = float(update_scale)
        self.normalize_client_updates = normalize_client_updates

    def evaluate_global_objectives(self):
        if self.eval_dataset is not None:
            return evaluate_objectives_on_dataset(self.model, self.eval_dataset, self.objective_fn, self.device)
        return _evaluate_global_objectives(self.model, self.clients, self.objective_fn, self.device)

    def sample_clients(self):
        sample_size = max(int(len(self.clients) * self.participation_rate), 1)
        return random.sample(self.clients, sample_size)

    def _clone_model(self):
        return copy.deepcopy(self.model).to(self.device)

    def run_round(self, round_idx):
        round_start = time.time()
        sampled = self.sample_clients()
        sampled_ids = []
        total_upload = 0
        total_nan_inf = 0
        max_client_mem = 0.0
        local_steps = 1
        client_rows = []

        client_start = time.time()
        for client in sampled:
            result = client.local_update(self._clone_model(), self.objective_fn)
            # A local delta approximates -eta * grad(client_loss); use -delta as the gradient row.
            row = (-result.delta_theta).to(self.device)
            if self.normalize_client_updates:
                row_norm = torch.norm(row, p=2)
                if row_norm.item() > 1e-12:
                    row = row / row_norm
            client_rows.append(row)
            sampled_ids.append(result.client_id)
            total_upload += result.upload_bytes
            total_nan_inf += _count_nan_inf(result.delta_theta)
            if result.peak_memory_mb > max_client_mem:
                max_client_mem = result.peak_memory_mb
            local_steps = getattr(client, "local_epochs", 1)
        client_compute_time = time.time() - client_start

        if not client_rows:
            raise RuntimeError("No client contributed local updates.")

        dir_start = time.time()
        client_jacobian = torch.stack(client_rows, dim=0)
        direction = self.aggregator(client_jacobian)
        direction_time = time.time() - dir_start
        total_nan_inf += _count_nan_inf(direction)

        update_start = time.time()
        current_flat = flatten_parameters(self.model.parameters())
        next_flat = current_flat - self.update_scale * direction
        assign_flat_parameters(self.model.parameters(), next_flat)
        update_time = time.time() - update_start

        model_bytes = current_flat.numel() * current_flat.element_size()
        download_bytes = model_bytes * len(sampled)
        per_client_upload = total_upload // max(len(sampled), 1)
        round_time = time.time() - round_start

        return RoundStats(
            round_idx=round_idx, sampled_client_ids=sampled_ids,
            num_sampled_clients=len(sampled),
            objective_values=self.evaluate_global_objectives(),
            direction_norm=float(torch.norm(direction, p=2).item()),
            jacobian_norm=float(torch.norm(client_jacobian, p="fro").item()),
            round_time=round_time, upload_bytes=total_upload,
            download_bytes=download_bytes, nan_inf_count=total_nan_inf,
            client_compute_time=client_compute_time,
            direction_time=direction_time, update_time=update_time,
            client_peak_memory_mb=max_client_mem,
            server_peak_memory_mb=_measure_peak_memory(),
            jacobian_upload_per_client=0,
            gradient_upload_per_client=per_client_upload,
            jacobian_vs_gradient_ratio=0.0,
            local_steps=local_steps,
            method_name="fedclient_upgrad",
        )


class QFedAvgServer:
    """q-FedAvg/q-FFL style loss-aware aggregation with FedAvg uploads."""

    def __init__(self, model, clients, objective_fn, participation_rate,
                 learning_rate, device, eval_dataset=None, q: float = 0.5,
                 eps: float = 1e-12):
        self.model = model.to(device)
        self.clients = clients
        self.objective_fn = objective_fn
        self.participation_rate = participation_rate
        self.learning_rate = learning_rate
        self.device = device
        self.eval_dataset = eval_dataset
        self.q = float(q)
        self.eps = eps

    def evaluate_global_objectives(self):
        if self.eval_dataset is not None:
            return evaluate_objectives_on_dataset(self.model, self.eval_dataset, self.objective_fn, self.device)
        return _evaluate_global_objectives(self.model, self.clients, self.objective_fn, self.device)

    def sample_clients(self):
        sample_size = max(int(len(self.clients) * self.participation_rate), 1)
        return random.sample(self.clients, sample_size)

    def _clone_model(self):
        return copy.deepcopy(self.model).to(self.device)

    def run_round(self, round_idx):
        round_start = time.time()
        sampled = self.sample_clients()
        sampled_ids = []
        total_upload = 0
        total_nan_inf = 0
        max_client_mem = 0.0
        local_steps = 1
        weighted_deltas = []
        h_values = []

        client_start = time.time()
        for client in sampled:
            result = client.local_update(self._clone_model(), self.objective_fn)
            delta = result.delta_theta.to(self.device)
            loss = max(float(result.initial_loss), self.eps)
            grad_proxy = -delta / max(self.learning_rate, self.eps)
            grad_norm_sq = float(torch.dot(grad_proxy, grad_proxy).item())
            loss_q = loss ** self.q
            h_i = self.q * (loss ** (self.q - 1.0)) * grad_norm_sq + (1.0 / max(self.learning_rate, self.eps)) * loss_q
            weighted_deltas.append(loss_q * delta)
            h_values.append(max(h_i, self.eps))
            sampled_ids.append(result.client_id)
            total_upload += result.upload_bytes
            total_nan_inf += _count_nan_inf(delta)
            if result.peak_memory_mb > max_client_mem:
                max_client_mem = result.peak_memory_mb
            local_steps = getattr(client, "local_epochs", 1)
        client_compute_time = time.time() - client_start

        if not weighted_deltas:
            raise RuntimeError("No client contributed q-FedAvg updates.")

        dir_start = time.time()
        aggregate_delta = sum(weighted_deltas) / max(sum(h_values), self.eps)
        direction = -aggregate_delta
        direction_time = time.time() - dir_start
        total_nan_inf += _count_nan_inf(direction)

        update_start = time.time()
        current_flat = flatten_parameters(self.model.parameters())
        next_flat = current_flat + aggregate_delta
        assign_flat_parameters(self.model.parameters(), next_flat)
        update_time = time.time() - update_start

        model_bytes = current_flat.numel() * current_flat.element_size()
        download_bytes = model_bytes * len(sampled)
        per_client_upload = total_upload // max(len(sampled), 1)
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
            gradient_upload_per_client=per_client_upload,
            jacobian_vs_gradient_ratio=0.0,
            local_steps=local_steps,
            method_name="qfedavg",
        )


class WeightedSumServer:
    """Weighted Sum: Each client computes weighted gradient locally, uploads only d-dim gradient."""

    def __init__(self, model, clients, objective_fn, participation_rate,
                 learning_rate, device, weights=None, eval_dataset=None):
        self.model = model.to(device)
        self.clients = clients
        self.objective_fn = objective_fn
        self.participation_rate = participation_rate
        self.learning_rate = learning_rate
        self.device = device
        self.weights = weights
        self.eval_dataset = eval_dataset

    def evaluate_global_objectives(self):
        if self.eval_dataset is not None:
            return evaluate_objectives_on_dataset(self.model, self.eval_dataset, self.objective_fn, self.device)
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
        all_weighted_grads = []
        per_client_grad_upload = 0
        total_examples = sum(c.num_examples for c in sampled)

        weights_tensor = None
        for client in sampled:
            if weights_tensor is None and self.weights is not None:
                weights_tensor = torch.tensor(self.weights, device=self.device, dtype=torch.float32)
            result = client.compute_weighted_gradient(self._clone_model(), self.objective_fn, weights=weights_tensor)
            sampled_ids.append(result.client_id)
            total_nan_inf += _count_nan_inf(result.vector)
            if result.peak_memory_mb > max_client_mem:
                max_client_mem = result.peak_memory_mb

            all_weighted_grads.append((result.vector.to(self.device), client.num_examples / total_examples))
            if per_client_grad_upload == 0:
                per_client_grad_upload = result.upload_bytes
        client_compute_time = time.time() - client_start

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dir_start = time.time()
        direction = sum(grad * weight for grad, weight in all_weighted_grads)
        direction_time = time.time() - dir_start
        total_nan_inf += _count_nan_inf(direction)

        update_start = time.time()
        current_flat = flatten_parameters(self.model.parameters())
        next_flat = current_flat - self.learning_rate * direction
        assign_flat_parameters(self.model.parameters(), next_flat)
        update_time = time.time() - update_start

        model_bytes = current_flat.numel() * current_flat.element_size()
        download_bytes = model_bytes * len(sampled)
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
                 learning_rate, device, eval_dataset=None):
        self.model = model.to(device)
        self.clients = clients
        self.objective_fn = objective_fn
        self.participation_rate = participation_rate
        self.learning_rate = learning_rate
        self.device = device
        self.eval_dataset = eval_dataset

    def evaluate_global_objectives(self):
        if self.eval_dataset is not None:
            return evaluate_objectives_on_dataset(self.model, self.eval_dataset, self.objective_fn, self.device)
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
        total_examples = sum(c.num_examples for c in sampled)

        client_start = time.time()
        total_nan_inf = 0
        max_client_mem = 0.0
        per_client_dir_upload = 0
        all_directions = []

        m = None
        for client in sampled:
            result = client.compute_weighted_gradient(self._clone_model(), self.objective_fn, weights=None)
            local_dir = result.vector.to(self.device)
            all_directions.append((local_dir, client.num_examples / total_examples))
            sampled_ids.append(result.client_id)
            total_nan_inf += _count_nan_inf(result.vector)
            if result.peak_memory_mb > max_client_mem:
                max_client_mem = result.peak_memory_mb
            if per_client_dir_upload == 0:
                per_client_dir_upload = result.upload_bytes
        client_compute_time = time.time() - client_start

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dir_start = time.time()
        direction = sum(local_dir * weight for local_dir, weight in all_directions)
        direction_time = time.time() - dir_start
        total_nan_inf += _count_nan_inf(direction)

        update_start = time.time()
        current_flat = flatten_parameters(self.model.parameters())
        next_flat = current_flat - self.learning_rate * direction
        assign_flat_parameters(self.model.parameters(), next_flat)
        update_time = time.time() - update_start

        model_bytes = current_flat.numel() * current_flat.element_size()
        download_bytes = model_bytes * len(sampled)
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
