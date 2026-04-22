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

from .nfjd_client import NFJDClient, ObjectiveFn, flatten_parameters, assign_flat_parameters, flatten_gradients


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
    effective_global_beta: float = 0.9
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

    def sample_clients(self) -> list[NFJDClient]:
        sample_size = max(int(len(self.clients) * self.participation_rate), 1)
        return random.sample(self.clients, sample_size)

    def _clone_model(self) -> nn.Module:
        cloned = copy.deepcopy(self.model)
        cloned.to(self.device)
        return cloned

    def evaluate_global_objectives(self) -> list[float]:
        total_examples = sum(client.num_examples for client in self.clients)
        running = None
        self.model.eval()
        with torch.no_grad():
            for client in self.clients:
                loader = DataLoader(client.dataset, batch_size=256, shuffle=False)
                client_values = []
                client_weights = []
                for batch_inputs, batch_targets in loader:
                    batch_inputs = batch_inputs.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    predictions = self.model(batch_inputs)
                    values = self.objective_fn(predictions, batch_targets, batch_inputs)
                    stacked = torch.stack([value.detach() for value in values])
                    if stacked.dim() == 2:
                        client_values.append(stacked.mean(dim=1))
                    else:
                        client_values.append(stacked)
                if client_values:
                    avg_values = torch.stack(client_values).mean(dim=0)
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

        client_start = time.time()
        delta_thetas = []
        sampled_ids = []
        total_upload = 0
        rescale_factors = []
        local_epochs_list = []
        cosine_sims = []

        def _run_single_client(client):
            model_clone = self._clone_model()
            return client.local_update(model_clone, self.objective_fn)

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
        
        aggregation_time = time.time() - agg_start

        avg_cosine_sim = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0
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

        download_bytes = current_flat.numel() * current_flat.element_size()
        round_time = time.time() - round_start

        avg_rescale = sum(rescale_factors) / len(rescale_factors) if rescale_factors else 1.0
        avg_local_epochs = sum(local_epochs_list) // len(local_epochs_list) if local_epochs_list else 1

        return RoundStats(
            round_idx=round_idx,
            sampled_client_ids=sampled_ids,
            num_sampled_clients=len(sampled_clients),
            objective_values=self.evaluate_global_objectives(),
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
            effective_global_beta=effective_beta,
            method_name="nfjd",
        )

