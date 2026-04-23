from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedjd.aggregators import MinNormAggregator

from .client import ObjectiveFn, _measure_peak_memory
from .evaluation import evaluate_objectives_on_dataset
from .nfjd_client import add_flat_update_, flatten_gradients
from .server import RoundStats, _count_nan_inf, assign_flat_parameters, flatten_parameters


@dataclass(frozen=True)
class Phase5MethodSpec:
    method: str
    display_name: str
    family: str
    paper_title: str
    paper_url: str
    official_repo: str
    implementation_note: str


PHASE5_METHOD_SPECS: dict[str, Phase5MethodSpec] = {
    "nfjd": Phase5MethodSpec(
        method="nfjd",
        display_name="NFJD",
        family="proposed",
        paper_title="NFJD (this repository)",
        paper_url="",
        official_repo="",
        implementation_note="Repository-native implementation of NFJD.",
    ),
    "fedavg_ls": Phase5MethodSpec(
        method="fedavg_ls",
        display_name="FedAvg+LS",
        family="official_baseline",
        paper_title="Linear Scalarization baseline (official Nash-MTL codebase API)",
        paper_url="https://github.com/AvivNavon/nash-mtl",
        official_repo="https://github.com/AvivNavon/nash-mtl",
        implementation_note="Federated wrapper with local linear-scalarization updates; local combiner aligned with the official Nash-MTL baseline API.",
    ),
    "fedavg_mgda": Phase5MethodSpec(
        method="fedavg_mgda",
        display_name="FedAvg+MGDA-UB",
        family="official_baseline",
        paper_title="Multi-Task Learning as Multi-Objective Optimization (Sener and Koltun, NeurIPS 2018)",
        paper_url="https://arxiv.org/abs/1810.04650",
        official_repo="https://github.com/isl-org/MultiObjectiveOptimization",
        implementation_note="Federated wrapper with local MGDA-UB updates; local min-norm solver aligned with the official MGDA repository.",
    ),
    "fedavg_pcgrad": Phase5MethodSpec(
        method="fedavg_pcgrad",
        display_name="FedAvg+PCGrad",
        family="official_baseline",
        paper_title="Gradient Surgery for Multi-Task Learning (Yu et al., ICLR 2020)",
        paper_url="https://arxiv.org/abs/2001.06782",
        official_repo="https://github.com/tianheyu927/PCGrad",
        implementation_note="Federated wrapper with local PCGrad updates; local projection rule aligned with the official PCGrad repository.",
    ),
    "fedavg_cagrad": Phase5MethodSpec(
        method="fedavg_cagrad",
        display_name="FedAvg+CAGrad",
        family="official_baseline",
        paper_title="Conflict-Averse Gradient Descent for Multi-task Learning (Liu et al., NeurIPS 2021)",
        paper_url="https://arxiv.org/abs/2110.14048",
        official_repo="https://github.com/Cranial-XIX/CAGrad",
        implementation_note="Federated wrapper with local CAGrad updates; local dual objective follows the official CAGrad formulation.",
    ),
}

PHASE5_FORMAL_BASELINES = ["fedavg_ls", "fedavg_mgda", "fedavg_pcgrad", "fedavg_cagrad"]


@dataclass
class Phase5ClientResult:
    client_id: int
    delta_theta: torch.Tensor
    num_examples: int
    compute_time: float = 0.0
    upload_bytes: int = 0
    peak_memory_mb: float = 0.0


def get_phase5_method_spec(method: str) -> Phase5MethodSpec:
    if method not in PHASE5_METHOD_SPECS:
        raise KeyError(f"Unknown Phase 5 method: {method}")
    return PHASE5_METHOD_SPECS[method]


def _compute_task_jacobian(model: nn.Module, losses: list[torch.Tensor]) -> torch.Tensor:
    rows = []
    for idx, objective in enumerate(losses):
        model.zero_grad(set_to_none=True)
        objective.backward(retain_graph=idx < len(losses) - 1)
        rows.append(flatten_gradients(model.parameters()))
    model.zero_grad(set_to_none=True)
    return torch.stack(rows, dim=0)


def _linear_scalarization_direction(losses: list[torch.Tensor], model: nn.Module) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    total_loss = sum(losses) / max(len(losses), 1)
    total_loss.backward()
    direction = flatten_gradients(model.parameters())
    model.zero_grad(set_to_none=True)
    return direction


def _pcgrad_direction(jacobian: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    task_grads = [jacobian[i].clone() for i in range(jacobian.shape[0])]
    projected = []
    for i, grad in enumerate(task_grads):
        mixed = grad.clone()
        order = torch.randperm(jacobian.shape[0], generator=generator).tolist()
        for j in order:
            if j == i:
                continue
            reference = task_grads[j]
            denom = torch.dot(reference, reference).item()
            if denom <= 1e-12:
                continue
            dot = torch.dot(mixed, reference).item()
            if dot < 0.0:
                mixed = mixed - (dot / denom) * reference
        projected.append(mixed)
    return torch.stack(projected, dim=0).mean(dim=0)


def _cagrad_direction(jacobian: torch.Tensor, c: float = 0.5) -> torch.Tensor:
    if jacobian.shape[0] == 1:
        return jacobian[0]

    jac_cpu = jacobian.detach().cpu().double().numpy()
    g0 = jacobian.mean(dim=0)
    g0_cpu = g0.detach().cpu().double().numpy()
    g0_norm = float(np.linalg.norm(g0_cpu))
    m = jacobian.shape[0]

    def objective(weights: np.ndarray) -> float:
        gw = np.sum(jac_cpu * weights[:, None], axis=0)
        return float(np.dot(g0_cpu, gw) + c * g0_norm * np.linalg.norm(gw))

    x0 = np.full(m, 1.0 / m, dtype=np.float64)
    bounds = [(0.0, 1.0)] * m
    constraints = ({"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},)
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 100, "ftol": 1e-12})
    weights = result.x if result.success else x0

    weights_t = torch.tensor(weights, dtype=jacobian.dtype, device=jacobian.device)
    gw = jacobian.T @ weights_t
    gw_norm = torch.norm(gw, p=2).item()
    lam = c * g0_norm / max(gw_norm, 1e-12)
    return (g0 + lam * gw) / (1.0 + c)


def combine_multi_task_gradients(method: str, losses: list[torch.Tensor], model: nn.Module, generator: torch.Generator) -> torch.Tensor:
    if method == "fedavg_ls":
        return _linear_scalarization_direction(losses, model)

    jacobian = _compute_task_jacobian(model, losses)
    if method == "fedavg_mgda":
        aggregator = MinNormAggregator(max_iters=250, lr=0.1, max_direction_norm=0.0)
        return aggregator(jacobian)
    if method == "fedavg_pcgrad":
        return _pcgrad_direction(jacobian, generator)
    if method == "fedavg_cagrad":
        return _cagrad_direction(jacobian)
    raise ValueError(f"Unsupported official baseline method: {method}")


class Phase5OfficialBaselineClient:
    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        batch_size: int,
        device: torch.device,
        learning_rate: float,
        local_epochs: int,
        method: str,
        seed: int,
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.method = method
        self.generator = torch.Generator().manual_seed(seed)

    @property
    def num_examples(self) -> int:
        return len(self.dataset)

    def local_update(self, model: nn.Module, objective_fn: ObjectiveFn) -> Phase5ClientResult:
        start = time.time()
        theta_init = flatten_parameters(model.parameters()).clone()
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.local_epochs):
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                predictions = model(batch_inputs)
                losses = objective_fn(predictions, batch_targets, batch_inputs)
                direction = combine_multi_task_gradients(self.method, losses, model, self.generator)
                add_flat_update_(model.parameters(), direction, alpha=-self.learning_rate)

        delta_theta = flatten_parameters(model.parameters()) - theta_init
        compute_time = time.time() - start
        upload_bytes = delta_theta.numel() * delta_theta.element_size()

        return Phase5ClientResult(
            client_id=self.client_id,
            delta_theta=delta_theta.detach().clone(),
            num_examples=self.num_examples,
            compute_time=compute_time,
            upload_bytes=upload_bytes,
            peak_memory_mb=_measure_peak_memory(),
        )


class Phase5OfficialBaselineServer:
    def __init__(
        self,
        model: nn.Module,
        clients: list[Phase5OfficialBaselineClient],
        objective_fn: ObjectiveFn,
        participation_rate: float,
        device: torch.device,
        method_name: str,
        eval_dataset=None,
    ) -> None:
        self.model = model.to(device)
        self.clients = clients
        self.objective_fn = objective_fn
        self.participation_rate = participation_rate
        self.device = device
        self.method_name = method_name
        self.eval_dataset = eval_dataset

    def sample_clients(self) -> list[Phase5OfficialBaselineClient]:
        sample_size = max(int(len(self.clients) * self.participation_rate), 1)
        return random.sample(self.clients, sample_size)

    def _clone_model(self) -> nn.Module:
        cloned = copy.deepcopy(self.model)
        cloned.to(self.device)
        return cloned

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

    def run_round(self, round_idx: int) -> RoundStats:
        round_start = time.time()
        sampled_clients = self.sample_clients()
        total_examples = sum(client.num_examples for client in sampled_clients)
        current_flat = flatten_parameters(self.model.parameters())
        model_bytes = current_flat.numel() * current_flat.element_size()

        client_start = time.time()
        total_upload = 0
        total_nan_inf = 0
        max_client_peak_mem = 0.0
        sampled_ids = []
        deltas = []

        for client in sampled_clients:
            result = client.local_update(self._clone_model(), self.objective_fn)
            sampled_ids.append(result.client_id)
            total_upload += result.upload_bytes
            total_nan_inf += _count_nan_inf(result.delta_theta)
            if result.peak_memory_mb > max_client_peak_mem:
                max_client_peak_mem = result.peak_memory_mb
            deltas.append((result.delta_theta.to(self.device), result.num_examples / total_examples))

        client_compute_time = time.time() - client_start
        aggregation_time = 0.0
        direction_time = 0.0

        aggregated_delta = sum(delta * weight for delta, weight in deltas)
        update_start = time.time()
        current_flat = current_flat + aggregated_delta
        assign_flat_parameters(self.model.parameters(), current_flat)
        update_time = time.time() - update_start

        download_bytes = model_bytes * len(sampled_clients)
        round_time = time.time() - round_start
        per_client_upload = total_upload // max(len(sampled_clients), 1)

        return RoundStats(
            round_idx=round_idx,
            sampled_client_ids=sampled_ids,
            num_sampled_clients=len(sampled_clients),
            objective_values=self.evaluate_global_objectives(),
            direction_norm=float(torch.norm(aggregated_delta, p=2).item()),
            jacobian_norm=0.0,
            round_time=round_time,
            upload_bytes=total_upload,
            download_bytes=download_bytes,
            nan_inf_count=total_nan_inf,
            client_compute_time=client_compute_time,
            aggregation_time=aggregation_time,
            direction_time=direction_time,
            update_time=update_time,
            client_peak_memory_mb=max_client_peak_mem,
            server_peak_memory_mb=_measure_peak_memory(),
            jacobian_upload_per_client=0,
            gradient_upload_per_client=per_client_upload,
            jacobian_vs_gradient_ratio=0.0,
            method_name=self.method_name,
        )


__all__ = [
    "PHASE5_FORMAL_BASELINES",
    "PHASE5_METHOD_SPECS",
    "Phase5MethodSpec",
    "Phase5OfficialBaselineClient",
    "Phase5OfficialBaselineServer",
    "combine_multi_task_gradients",
    "get_phase5_method_spec",
]
