from __future__ import annotations

import copy
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from fedjd.aggregators import MinNormAggregator
from fedjd.core.scaling import GlobalMomentum

from .evaluation import evaluate_objectives_on_dataset
from .nfjd_client import NFJDClient, ObjectiveFn, flatten_parameters, assign_flat_parameters, flatten_gradients, get_model_parameter_groups, _flatten_gradient_list


@dataclass
class RoundStats:
    round_idx: int
    sampled_client_ids: list[int]
    num_sampled_clients: int
    objective_values: list[float]
    delta_norm: float
    global_momentum_norm: float
    round_time: float = 0.0
    upload_bytes: int = 0
    download_bytes: int = 0
    client_compute_time: float = 0.0
    aggregation_time: float = 0.0
    update_time: float = 0.0
    avg_rescale_factor: float = 1.0
    avg_local_epochs: int = 0
    avg_cosine_sim: float = 0.0
    avg_prox_ratio: float = 0.0
    avg_scaffold_ratio: float = 0.0
    avg_cone_margin: float = 0.0
    avg_cone_cosine: float = 0.0
    effective_global_beta: float = 0.9
    task_weights: list[float] = field(default_factory=list)
    task_weight_gap: float = 0.0
    method_name: str = "nfjd"


class NFJDServer:
    def __init__(
        self,
        model: nn.Module,
        clients: list[NFJDClient],
        objective_fn: ObjectiveFn,
        participation_rate: float,
        learning_rate: float,
        device: torch.device,
        global_momentum_beta: float = 0.9,
        conflict_aware_momentum: bool = False,
        momentum_min_beta: float = 0.1,
        parallel_clients: bool | None = None,
        eval_dataset=None,
        use_global_progress_weights: bool = False,
        progress_beta: float = 2.0,
        progress_min_weight: float = 0.5,
        progress_max_weight: float = 2.0,
        progress_ema_beta: float = 0.0,
        progress_max_change: float = 0.0,
        method_name: str = "nfjd",
        cone_align_alpha: float = 0.0,
        cone_reference_mode: str = "delta",
        cone_basis_size: int = 0,
        use_shared_scaffold: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.clients = clients
        self.objective_fn = objective_fn
        self.participation_rate = participation_rate
        self.learning_rate = learning_rate
        self.device = device
        self.global_momentum = GlobalMomentum(
            beta=global_momentum_beta,
            conflict_aware=conflict_aware_momentum,
            min_beta=momentum_min_beta,
        )
        self.conflict_aware_momentum = conflict_aware_momentum
        self.parallel_clients = device.type != "cuda" if parallel_clients is None else parallel_clients
        self.eval_dataset = eval_dataset
        self.use_global_progress_weights = use_global_progress_weights
        self.progress_beta = progress_beta
        self.progress_min_weight = progress_min_weight
        self.progress_max_weight = progress_max_weight
        self.progress_ema_beta = progress_ema_beta
        self.progress_max_change = progress_max_change
        self.method_name = method_name
        self.cone_align_alpha = max(float(cone_align_alpha), 0.0)
        self.cone_reference_mode = cone_reference_mode
        self.cone_basis_size = max(int(cone_basis_size), 0)
        self.use_shared_scaffold = use_shared_scaffold
        self.initial_objectives: list[float] | None = None
        self.previous_objectives: list[float] | None = None
        self.task_weights: torch.Tensor | None = None
        self.shared_direction_reference: torch.Tensor | None = None
        self.shared_direction_basis: torch.Tensor | None = None
        shared_params, _ = get_model_parameter_groups(self.model)
        self.shared_control_global = torch.zeros_like(flatten_parameters(shared_params), device=self.device) if self.use_shared_scaffold and shared_params else None

    def _compute_validation_shared_reference(self) -> torch.Tensor | None:
        if self.eval_dataset is None:
            return None
        shared_params, _ = get_model_parameter_groups(self.model)
        if not shared_params:
            return None

        loader = DataLoader(self.eval_dataset, batch_size=256, shuffle=False)
        gradient_sum = torch.zeros_like(flatten_parameters(shared_params), device=self.device)
        total_batches = 0

        self.model.train()
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(self.device)
            batch_targets = batch_targets.to(self.device)
            predictions = self.model(batch_inputs)
            losses = self.objective_fn(predictions, batch_targets, batch_inputs)
            total_loss = sum(losses)
            grads = torch.autograd.grad(total_loss, shared_params, allow_unused=True)
            gradient_sum.add_(_flatten_gradient_list(grads, shared_params))
            total_batches += 1

        if total_batches == 0:
            return None
        gradient_avg = gradient_sum / total_batches
        ref = -gradient_avg
        ref_norm = torch.norm(ref, p=2)
        if ref_norm.item() <= 1e-12:
            return None
        return (ref / ref_norm).detach()

    def _select_direction_basis(self, probe_directions: list[tuple[torch.Tensor, float]]) -> torch.Tensor | None:
        if not probe_directions:
            return None
        normalized = []
        weights = []
        for direction, weight in probe_directions:
            norm = torch.norm(direction, p=2)
            if norm.item() <= 1e-12:
                continue
            normalized.append(direction / norm)
            weights.append(float(weight))
        if not normalized:
            return None
        if self.cone_basis_size <= 1 or len(normalized) == 1:
            center = sum(direction * weight for direction, weight in zip(normalized, weights))
            center_norm = torch.norm(center, p=2)
            if center_norm.item() <= 1e-12:
                return normalized[0].unsqueeze(0).detach()
            return (center / center_norm).unsqueeze(0).detach()

        center = sum(direction * weight for direction, weight in zip(normalized, weights))
        center_norm = torch.norm(center, p=2)
        center = center / center_norm if center_norm.item() > 1e-12 else normalized[0]
        cosines_to_center = [float(torch.dot(direction, center).item()) for direction in normalized]
        selected = [max(range(len(normalized)), key=lambda idx: cosines_to_center[idx])]

        while len(selected) < min(self.cone_basis_size, len(normalized)):
            best_idx = None
            best_score = None
            for idx, direction in enumerate(normalized):
                if idx in selected:
                    continue
                max_similarity = max(float(torch.dot(direction, normalized[s]).item()) for s in selected)
                score = 1.0 - max_similarity
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
        return torch.stack([normalized[idx] for idx in selected], dim=0).detach()

    def set_initial_objectives(self, objectives: list[float]) -> None:
        self.initial_objectives = list(objectives)
        self.previous_objectives = list(objectives)
        self.task_weights = torch.ones(len(objectives), dtype=torch.float32, device=self.device)

    def _compute_task_weights(self) -> torch.Tensor:
        if not self.use_global_progress_weights:
            if self.task_weights is not None:
                return self.task_weights
            objectives = self.previous_objectives or self.evaluate_global_objectives()
            self.task_weights = torch.ones(len(objectives), dtype=torch.float32, device=self.device)
            return self.task_weights

        if self.initial_objectives is None:
            current = self.evaluate_global_objectives()
            self.set_initial_objectives(current)

        if self.previous_objectives is None:
            self.previous_objectives = list(self.initial_objectives)

        init = torch.tensor(self.initial_objectives, dtype=torch.float32, device=self.device)
        current = torch.tensor(self.previous_objectives, dtype=torch.float32, device=self.device)
        ri = (init - current) / torch.clamp(torch.abs(init), min=1e-8)
        deficit = torch.mean(ri) - ri
        weights = torch.exp(self.progress_beta * deficit)
        weights = torch.clamp(weights, min=self.progress_min_weight, max=self.progress_max_weight)
        weights = weights / torch.clamp(torch.mean(weights), min=1e-8)

        if self.task_weights is not None and self.progress_max_change > 0:
            max_ratio = 1.0 + self.progress_max_change
            min_ratio = max(1.0 - self.progress_max_change, 1e-6)
            lower = self.task_weights * min_ratio
            upper = self.task_weights * max_ratio
            weights = torch.minimum(torch.maximum(weights, lower), upper)

        if self.task_weights is not None and self.progress_ema_beta > 0:
            beta = min(max(self.progress_ema_beta, 0.0), 0.999)
            weights = beta * self.task_weights + (1.0 - beta) * weights

        weights = weights / torch.clamp(torch.mean(weights), min=1e-8)
        self.task_weights = weights.detach()
        return self.task_weights

    def sample_clients(self) -> list[NFJDClient]:
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
        task_weights = self._compute_task_weights()
        if self.cone_align_alpha > 0 and self.cone_reference_mode == "validation_gradient":
            self.shared_direction_reference = self._compute_validation_shared_reference()
            self.shared_direction_basis = None
        elif self.cone_align_alpha > 0 and self.cone_reference_mode == "probe_basis":
            probe_start = time.time()
            probe_directions = []
            for client in sampled_clients:
                model_clone = self._clone_model()
                probe = client.probe_shared_direction(model_clone, self.objective_fn, task_weights=task_weights)
                if probe is not None:
                    probe_directions.append((probe.to(self.device), client.num_examples / total_examples))
            self.shared_direction_basis = self._select_direction_basis(probe_directions)
            self.shared_direction_reference = None
        elif self.cone_align_alpha > 0 and self.cone_reference_mode != "delta":
            self.shared_direction_reference = None
            self.shared_direction_basis = None

        client_start = time.time()
        delta_thetas = []
        sampled_ids = []
        total_upload = 0
        rescale_factors = []
        local_epochs_list = []
        cosine_sims = []
        prox_ratios = []
        scaffold_ratios = []
        cone_margins = []
        cone_cosines = []

        def _run_single_client(client):
            model_clone = self._clone_model()
            return client.local_update(
                model_clone,
                self.objective_fn,
                task_weights=task_weights,
                shared_control_global=self.shared_control_global,
                cone_reference_shared_direction=self.shared_direction_reference,
                cone_reference_shared_basis=self.shared_direction_basis,
                cone_align_alpha=self.cone_align_alpha,
            )

        if self.parallel_clients and len(sampled_clients) > 1:
            with ThreadPoolExecutor(max_workers=len(sampled_clients)) as executor:
                results = list(executor.map(_run_single_client, sampled_clients))
        else:
            results = [_run_single_client(client) for client in sampled_clients]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        delta_thetas = []
        align_scores_list = []
        sampled_ids = []
        total_upload = 0
        rescale_factors = []
        local_epochs_list = []
        cosine_sims = []
        prox_ratios = []
        scaffold_ratios = []

        for result in results:
            weight = result.num_examples / total_examples
            delta_thetas.append((result.delta_theta.to(self.device), weight))
            if result.align_scores is not None:
                align_scores_list.append((result.align_scores.to(self.device), weight))
            sampled_ids.append(result.client_id)
            total_upload += result.delta_theta.numel() * result.delta_theta.element_size()
            if result.align_scores is not None:
                total_upload += result.align_scores.numel() * result.align_scores.element_size()
            rescale_factors.append(result.rescale_factor)
            local_epochs_list.append(result.num_local_epochs)
            cosine_sims.append(result.avg_cosine_sim)
            prox_ratios.append(result.avg_prox_ratio)
            scaffold_ratios.append(result.avg_scaffold_ratio)
            cone_margins.append(result.avg_cone_margin)
            cone_cosines.append(result.avg_cone_cosine)

        client_compute_time = time.time() - client_start

        agg_start = time.time()
        
        # 计算全局对齐分数
        global_align = None
        if align_scores_list:
            global_align = sum(align * weight for align, weight in align_scores_list)
        
        # 调整客户端权重
        adjusted_weights = []
        for i, (delta, weight) in enumerate(delta_thetas):
            if align_scores_list and global_align is not None:
                # 获取当前客户端的对齐分数
                client_align = align_scores_list[i][0]
                # 计算权重调整因子：对被普遍伤害的目标，增加对其友好的客户端权重
                adjustment = 1.0
                for j in range(len(client_align)):
                    if global_align[j] < 0:  # 目标被普遍伤害
                        if client_align[j] > 0:  # 当前客户端对该目标友好
                            adjustment *= 1.5  # 增加权重
                adjusted_weights.append((delta, weight * adjustment))
            else:
                adjusted_weights.append((delta, weight))
        
        # 归一化调整后的权重
        total_adjusted_weight = sum(w for _, w in adjusted_weights)
        if total_adjusted_weight > 0:
            adjusted_weights = [(delta, w / total_adjusted_weight) for delta, w in adjusted_weights]
        
        # 基于调整后的权重聚合 delta_theta
        aggregated_delta = sum(delta * weight for delta, weight in adjusted_weights)
        shared_deltas = [
            (result.shared_delta_theta.to(self.device), result.num_examples / total_examples)
            for result in results
            if result.shared_delta_theta is not None
        ]
        if self.cone_align_alpha > 0 and self.cone_reference_mode == "delta" and shared_deltas:
            aggregated_shared_delta = sum(delta * weight for delta, weight in shared_deltas)
            shared_norm = torch.norm(aggregated_shared_delta, p=2)
            if shared_norm.item() > 1e-12:
                self.shared_direction_reference = (aggregated_shared_delta / shared_norm).detach()
                self.shared_direction_basis = None

        if self.use_shared_scaffold and self.shared_control_global is not None:
            control_deltas = [result.control_delta_shared.to(self.device) for result in results if result.control_delta_shared is not None]
            if control_deltas:
                mean_delta = sum(control_deltas) / len(self.clients)
                self.shared_control_global = (self.shared_control_global + mean_delta).detach()
        
        aggregation_time = time.time() - agg_start

        avg_cosine_sim = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0
        avg_prox_ratio = sum(prox_ratios) / len(prox_ratios) if prox_ratios else 0.0
        avg_scaffold_ratio = sum(scaffold_ratios) / len(scaffold_ratios) if scaffold_ratios else 0.0
        avg_cone_margin = sum(cone_margins) / len(cone_margins) if cone_margins else 0.0
        avg_cone_cosine = sum(cone_cosines) / len(cone_cosines) if cone_cosines else 0.0
        if self.conflict_aware_momentum:
            momentum_delta = self.global_momentum.update(aggregated_delta, avg_cosine_sim=avg_cosine_sim)
        else:
            momentum_delta = self.global_momentum.update(aggregated_delta)

        effective_beta = self.global_momentum.beta

        update_start = time.time()
        current_flat = flatten_parameters(self.model.parameters())
        current_flat = current_flat + momentum_delta
        assign_flat_parameters(self.model.parameters(), current_flat)
        update_time = time.time() - update_start

        model_bytes = current_flat.numel() * current_flat.element_size()
        download_bytes = model_bytes * len(sampled_clients)
        round_time = time.time() - round_start

        avg_rescale = sum(rescale_factors) / len(rescale_factors) if rescale_factors else 1.0
        avg_local_epochs = sum(local_epochs_list) // len(local_epochs_list) if local_epochs_list else 1

        objective_values = self.evaluate_global_objectives()
        self.previous_objectives = list(objective_values)

        task_weight_list = [float(v.item()) for v in task_weights.detach().cpu()]
        task_weight_gap = max(task_weight_list) - min(task_weight_list) if task_weight_list else 0.0

        return RoundStats(
            round_idx=round_idx,
            sampled_client_ids=sampled_ids,
            num_sampled_clients=len(sampled_clients),
            objective_values=objective_values,
            delta_norm=float(torch.norm(aggregated_delta, p=2).item()),
            global_momentum_norm=float(torch.norm(momentum_delta, p=2).item()),
            round_time=round_time,
            upload_bytes=total_upload,
            download_bytes=download_bytes,
            client_compute_time=client_compute_time,
            aggregation_time=aggregation_time,
            update_time=update_time,
            avg_rescale_factor=avg_rescale,
            avg_local_epochs=avg_local_epochs,
            avg_cosine_sim=avg_cosine_sim,
            avg_prox_ratio=avg_prox_ratio,
            avg_scaffold_ratio=avg_scaffold_ratio,
            avg_cone_margin=avg_cone_margin,
            avg_cone_cosine=avg_cone_cosine,
            effective_global_beta=effective_beta,
            task_weights=task_weight_list,
            task_weight_gap=task_weight_gap,
            method_name=self.method_name,
        )

