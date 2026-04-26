from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from itertools import chain

import numpy as np
import torch
try:
    from scipy.optimize import minimize, minimize_scalar
except ModuleNotFoundError:  # Optional: only FedAvg+CAGrad needs SciPy.
    minimize = None
    minimize_scalar = None
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .client import ObjectiveFn, _measure_peak_memory
from .evaluation import evaluate_objectives_on_dataset
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
        implementation_note="Current mainline NFJD uses global-progress-aware exact local UPGrad on shared parameters, objective normalization, delta_theta-only upload, and sample-weighted FedAvg aggregation.",
    ),
    "nfjd_fast": Phase5MethodSpec(
        method="nfjd_fast",
        display_name="NFJD-fast",
        family="proposed_ablation",
        paper_title="NFJD fast approximate variant (this repository)",
        paper_url="",
        official_repo="",
        implementation_note="Approximate NFJD for real/many-task datasets: stochastic Gramian, cached UPGrad weights, adaptive rescaling, moderate momentum, and smoothed global progress weights.",
    ),
    "nfjd_noweight": Phase5MethodSpec(
        method="nfjd_noweight",
        display_name="NFJD-noWeight",
        family="proposed_ablation",
        paper_title="NFJD without global progress weighting (this repository)",
        paper_url="",
        official_repo="",
        implementation_note="Exact NFJD configuration with global progress weighting disabled, used to test whether progress compensation hurts non-IID generalization.",
    ),
    "nfjd_cached": Phase5MethodSpec(
        method="nfjd_cached",
        display_name="NFJD-cached",
        family="proposed_ablation",
        paper_title="NFJD cached-lambda ablation (this repository)",
        paper_url="",
        official_repo="",
        implementation_note="Isolates cached local UPGrad weights by disabling exact per-step solves and recomputing every 4 local steps while keeping the reference NFJD weighting and no momentum/rescaling.",
    ),
    "nfjd_momentum": Phase5MethodSpec(
        method="nfjd_momentum",
        display_name="NFJD-momentum",
        family="proposed_ablation",
        paper_title="NFJD momentum ablation (this repository)",
        paper_url="",
        official_repo="",
        implementation_note="Isolates local and global momentum on top of exact NFJD to test whether smoothing helps non-IID or harms IID accuracy.",
    ),
    "nfjd_rescale": Phase5MethodSpec(
        method="nfjd_rescale",
        display_name="NFJD-rescale",
        family="proposed_ablation",
        paper_title="NFJD adaptive-rescaling ablation (this repository)",
        paper_url="",
        official_repo="",
        implementation_note="Isolates adaptive direction rescaling on top of exact NFJD.",
    ),
    "nfjd_softweight": Phase5MethodSpec(
        method="nfjd_softweight",
        display_name="NFJD-softWeight",
        family="proposed_ablation",
        paper_title="NFJD smoothed progress-weight ablation (this repository)",
        paper_url="",
        official_repo="",
        implementation_note="Keeps exact NFJD but replaces aggressive progress weights with bounded, EMA-smoothed task weights.",
    ),
    "nfjd_hybrid": Phase5MethodSpec(
        method="nfjd_hybrid",
        display_name="NFJD-hybrid",
        family="proposed_ablation",
        paper_title="NFJD hybrid practical variant (this repository)",
        paper_url="",
        official_repo="",
        implementation_note="A lighter practical candidate between exact and fast NFJD: cached solves every 2 steps, stochastic task subsets for many-task cases, mild momentum, and smoothed progress weights without adaptive rescaling.",
    ),
    "nfjd_fedprox_shared": Phase5MethodSpec(
        method="nfjd_fedprox_shared",
        display_name="NFJD+FedProx(shared)",
        family="proposed_ablation",
        paper_title="NFJD with shared-trunk FedProx regularization (this repository)",
        paper_url="",
        official_repo="",
        implementation_note="Exact NFJD on shared parameters with an added shared-trunk proximal term toward the current global model, used to test whether client drift rather than task weighting is the main non-IID bottleneck.",
    ),
    "nfjd_scaffold_shared": Phase5MethodSpec(
        method="nfjd_scaffold_shared",
        display_name="NFJD+SCAFFOLD(shared)",
        family="proposed_mainline",
        paper_title="NFJD with shared-trunk SCAFFOLD correction (this repository)",
        paper_url="",
        official_repo="",
        implementation_note="Exact NFJD on shared parameters with shared-only control-variate correction; each task gradient row is corrected by the same shared SCAFFOLD term before local UPGrad, targeting client drift without changing head-specific updates.",
    ),
    "nfjd_common_safe": Phase5MethodSpec(
        method="nfjd_common_safe",
        display_name="NFJD-common-safe",
        family="proposed_ablation",
        paper_title="NFJD low-cost common-safe preprocessing variant (this repository)",
        paper_url="",
        official_repo="",
        implementation_note="A low-cost NFJD variant that refreshes a public common-safe shared direction only every few rounds using one small probe batch per sampled client, caches that direction on the server, and mixes it only into the first few local shared-parameter steps before reverting to standard local NFJD.",
    ),
    "fedavg_ls": Phase5MethodSpec(
        method="fedavg_ls",
        display_name="FedAvg+LS",
        family="official_baseline",
        paper_title="Linear Scalarization baseline (official Nash-MTL codebase API)",
        paper_url="https://github.com/AvivNavon/nash-mtl",
        official_repo="https://github.com/AvivNavon/nash-mtl",
        implementation_note="Federated wrapper with local sum-loss updates; local scalarization follows the official Nash-MTL LS baseline without 1/m normalization.",
    ),
    "fmgda": Phase5MethodSpec(
        method="fmgda",
        display_name="FMGDA",
        family="federated_baseline",
        paper_title="Federated Multi-Objective Learning (Yang et al., NeurIPS 2023)",
        paper_url="https://arxiv.org/abs/2310.09866",
        official_repo="",
        implementation_note="Paper-aligned FMGDA with one local trajectory per objective on each client, objective-wise gradient aggregation on the server, and a server-side MGDA min-norm solve.",
    ),
    "fedmgda_plus": Phase5MethodSpec(
        method="fedmgda_plus",
        display_name="FedMGDA+",
        family="federated_baseline",
        paper_title="Federated Multi-Objective Learning (Yang et al., NeurIPS 2023)",
        paper_url="https://arxiv.org/abs/2310.09866",
        official_repo="",
        implementation_note="Communication-saving FedMGDA+ style baseline: each client computes local per-objective updates, solves a local MGDA min-norm problem, uploads one local common direction, and the server averages those directions before updating the global model.",
    ),
    "fedavg_upgrad": Phase5MethodSpec(
        method="fedavg_upgrad",
        display_name="FedAvg+UPGrad(server)",
        family="proposed_ablation",
        paper_title="Server-side UPGrad over aggregated task gradients (this repository)",
        paper_url="",
        official_repo="",
        implementation_note="Clients upload local per-task gradient matrices as in FedJD/FedAvg-style full sync; the server averages them by sample count and applies UPGrad once on the aggregated task matrix before updating the global model.",
    ),
    "fedclient_upgrad": Phase5MethodSpec(
        method="fedclient_upgrad",
        display_name="FedClient-UPGrad",
        family="proposed_ablation",
        paper_title="Server-side UPGrad over client local losses (this repository)",
        paper_url="",
        official_repo="",
        implementation_note="Clients perform ordinary local weighted-sum training and upload model deltas; the server treats sampled clients as descent objectives and uses UPGrad to choose a common update that mitigates client-level conflicts.",
    ),
    "qfedavg": Phase5MethodSpec(
        method="qfedavg",
        display_name="q-FedAvg",
        family="federated_fairness_baseline",
        paper_title="Fair Resource Allocation in Federated Learning (q-FFL / q-FedAvg)",
        paper_url="https://arxiv.org/abs/1905.10497",
        official_repo="",
        implementation_note="FedAvg-style local training with loss-aware q-FFL aggregation. Clients upload one model delta; the server weights updates by the current local objective value to emphasize high-loss clients.",
    ),
    "fedavg_pcgrad": Phase5MethodSpec(
        method="fedavg_pcgrad",
        display_name="FedAvg+PCGrad",
        family="official_baseline",
        paper_title="Gradient Surgery for Multi-Task Learning (Yu et al., ICLR 2020)",
        paper_url="https://arxiv.org/abs/2001.06782",
        official_repo="https://github.com/tianheyu927/PCGrad",
        implementation_note="Federated wrapper with local PCGrad updates on shared parameters only; task-specific heads follow the summed task loss, matching the official multitask usage pattern.",
    ),
    "fedavg_cagrad": Phase5MethodSpec(
        method="fedavg_cagrad",
        display_name="FedAvg+CAGrad",
        family="official_baseline",
        paper_title="Conflict-Averse Gradient Descent for Multi-task Learning (Liu et al., NeurIPS 2021)",
        paper_url="https://arxiv.org/abs/2110.14048",
        official_repo="https://github.com/Cranial-XIX/CAGrad",
        implementation_note="Federated wrapper with local CAGrad updates on shared parameters only; the dual is solved in task space with the official rescaling convention and summed-loss head updates.",
    ),
}

PHASE5_FORMAL_BASELINES = ["fedavg_ls", "fedavg_pcgrad", "fedavg_cagrad"]


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


def _unique_parameters(modules: list[nn.Module]) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for param in module.parameters():
            if id(param) not in seen:
                params.append(param)
                seen.add(id(param))
    return params


def _get_model_parameter_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
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
    return torch.stack(projected, dim=0).sum(dim=0)


def _cagrad_direction(jacobian: torch.Tensor, c: float = 0.4) -> torch.Tensor:
    if minimize is None or minimize_scalar is None:
        raise ImportError("FedAvg+CAGrad requires scipy. Install scipy or omit fedavg_cagrad.")
    if jacobian.shape[0] == 1:
        return jacobian[0]

    m = jacobian.shape[0]
    gram = (jacobian @ jacobian.T).detach().cpu().double().numpy()
    alpha = np.full(m, 1.0 / m, dtype=np.float64)
    g0_norm = float(np.sqrt(alpha @ gram @ alpha + 1e-8))
    coef = c * g0_norm

    def objective(weights: np.ndarray) -> float:
        quad = float(weights @ gram @ weights)
        coupling = float(alpha @ gram @ weights)
        return coupling + coef * np.sqrt(max(quad, 1e-8))

    if m == 2:
        result = minimize_scalar(lambda x: objective(np.array([x, 1.0 - x], dtype=np.float64)), bounds=(0.0, 1.0), method="bounded")
        weights = np.array([result.x, 1.0 - result.x], dtype=np.float64) if result.success else alpha
    else:
        bounds = [(0.0, 1.0)] * m
        constraints = ({"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},)
        result = minimize(objective, alpha, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 100, "ftol": 1e-12})
        weights = result.x if result.success else alpha

    weights_t = torch.tensor(weights, dtype=jacobian.dtype, device=jacobian.device)
    g0 = jacobian.T @ torch.full((m,), 1.0 / m, dtype=jacobian.dtype, device=jacobian.device)
    gw = jacobian.T @ weights_t
    gw_norm = torch.norm(gw, p=2).item()
    lam = c * g0_norm / max(gw_norm, 1e-12)
    return (g0 + lam * gw) / (1.0 + c * c)


def _compute_shared_task_jacobian(
    losses: list[torch.Tensor],
    shared_params: list[nn.Parameter],
) -> torch.Tensor:
    rows = []
    for task_idx, objective in enumerate(losses):
        grads = torch.autograd.grad(
            objective,
            shared_params,
            retain_graph=task_idx < len(losses) - 1,
            allow_unused=True,
        )
        rows.append(_flatten_gradient_list(grads, shared_params))
    return torch.stack(rows, dim=0)


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
        shared_params, head_params = _get_model_parameter_groups(model)

        for _ in range(self.local_epochs):
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                predictions = model(batch_inputs)
                losses = objective_fn(predictions, batch_targets, batch_inputs)
                total_loss = sum(losses)

                if self.method == "fedavg_ls":
                    all_params = list(model.parameters())
                    grads = torch.autograd.grad(total_loss, all_params, allow_unused=True)
                    _apply_gradients(all_params, grads, self.learning_rate)
                    continue

                head_grads: tuple[torch.Tensor | None, ...] = tuple()
                if head_params:
                    head_grads = torch.autograd.grad(total_loss, head_params, retain_graph=True, allow_unused=True)

                jacobian = _compute_shared_task_jacobian(losses, shared_params)
                if self.method == "fedavg_pcgrad":
                    shared_direction = _pcgrad_direction(jacobian, self.generator)
                elif self.method == "fedavg_cagrad":
                    shared_direction = jacobian.shape[0] * _cagrad_direction(jacobian)
                else:
                    raise ValueError(f"Unsupported official baseline method: {self.method}")

                _apply_flat_direction(shared_params, shared_direction, self.learning_rate)
                _apply_gradients(head_params, head_grads, self.learning_rate)

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
    "get_phase5_method_spec",
]
