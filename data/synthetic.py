from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import ConcatDataset, TensorDataset


@dataclass
class FederatedRegressionData:
    client_datasets: list[TensorDataset]
    input_dim: int
    num_objectives: int = 2
    true_weights: list[torch.Tensor] | None = None
    val_dataset: ConcatDataset | None = None
    test_dataset: ConcatDataset | None = None


def make_synthetic_federated_regression(
    num_clients: int = 10,
    samples_per_client: int = 64,
    input_dim: int = 8,
    num_objectives: int = 2,
    noise_std: float = 0.1,
    seed: int = 7,
    val_samples_per_client: int | None = None,
    test_samples_per_client: int | None = None,
) -> FederatedRegressionData:
    generator = torch.Generator().manual_seed(seed)
    val_samples = val_samples_per_client if val_samples_per_client is not None else max(samples_per_client // 4, 16)
    test_samples = test_samples_per_client if test_samples_per_client is not None else max(samples_per_client // 2, 32)

    true_weights = []
    for obj_idx in range(num_objectives):
        if obj_idx == 0:
            w = torch.randn(input_dim, 1, generator=generator)
        else:
            w = -true_weights[0] / num_objectives + torch.randn(input_dim, 1, generator=generator) * (0.5 + 0.1 * obj_idx)
        true_weights.append(w)

    client_datasets = []
    val_client_datasets = []
    test_client_datasets = []
    for client_idx in range(num_clients):
        feature_shift = 0.2 * client_idx
        inputs = torch.randn(samples_per_client, input_dim, generator=generator) + feature_shift

        target_cols = []
        for w in true_weights:
            noise = noise_std * torch.randn(samples_per_client, 1, generator=generator)
            target_cols.append(inputs @ w + noise)
        targets = torch.cat(target_cols, dim=1)

        client_datasets.append(TensorDataset(inputs, targets))

        val_inputs = torch.randn(val_samples, input_dim, generator=generator) + feature_shift
        val_target_cols = []
        for w in true_weights:
            val_noise = noise_std * torch.randn(val_samples, 1, generator=generator)
            val_target_cols.append(val_inputs @ w + val_noise)
        val_targets = torch.cat(val_target_cols, dim=1)
        val_client_datasets.append(TensorDataset(val_inputs, val_targets))

        test_inputs = torch.randn(test_samples, input_dim, generator=generator) + feature_shift
        test_target_cols = []
        for w in true_weights:
            test_noise = noise_std * torch.randn(test_samples, 1, generator=generator)
            test_target_cols.append(test_inputs @ w + test_noise)
        test_targets = torch.cat(test_target_cols, dim=1)
        test_client_datasets.append(TensorDataset(test_inputs, test_targets))

    return FederatedRegressionData(
        client_datasets=client_datasets,
        input_dim=input_dim,
        num_objectives=num_objectives,
        true_weights=true_weights,
        val_dataset=ConcatDataset(val_client_datasets),
        test_dataset=ConcatDataset(test_client_datasets),
    )


__all__ = ["FederatedRegressionData", "make_synthetic_federated_regression", "make_high_conflict_federated_regression"]


def make_high_conflict_federated_regression(
    num_clients: int = 10,
    samples_per_client: int = 64,
    input_dim: int = 8,
    num_objectives: int = 2,
    noise_std: float = 0.1,
    conflict_strength: float = 1.0,
    seed: int = 7,
    val_samples_per_client: int | None = None,
    test_samples_per_client: int | None = None,
) -> FederatedRegressionData:
    generator = torch.Generator().manual_seed(seed)
    val_samples = val_samples_per_client if val_samples_per_client is not None else max(samples_per_client // 4, 16)
    test_samples = test_samples_per_client if test_samples_per_client is not None else max(samples_per_client // 2, 32)

    base_weight = torch.randn(input_dim, 1, generator=generator)
    base_weight = base_weight / torch.norm(base_weight)

    true_weights = [base_weight]
    for obj_idx in range(1, num_objectives):
        conflict_component = -base_weight * conflict_strength
        orthogonal = torch.randn(input_dim, 1, generator=generator)
        proj = (orthogonal.T @ base_weight) * base_weight
        orthogonal = orthogonal - proj
        orth_norm = torch.norm(orthogonal)
        if orth_norm > 1e-8:
            orthogonal = orthogonal / orth_norm
        else:
            orthogonal = torch.zeros_like(orthogonal)
        diversity_scale = 0.3 / (1.0 + 0.1 * obj_idx)
        w = conflict_component + orthogonal * diversity_scale
        true_weights.append(w)

    client_datasets = []
    val_client_datasets = []
    test_client_datasets = []
    for client_idx in range(num_clients):
        feature_shift = 0.3 * client_idx
        inputs = torch.randn(samples_per_client, input_dim, generator=generator) + feature_shift

        target_cols = []
        for w in true_weights:
            noise = noise_std * torch.randn(samples_per_client, 1, generator=generator)
            target_cols.append(inputs @ w + noise)
        targets = torch.cat(target_cols, dim=1)

        client_datasets.append(TensorDataset(inputs, targets))

        val_inputs = torch.randn(val_samples, input_dim, generator=generator) + feature_shift
        val_target_cols = []
        for w in true_weights:
            val_noise = noise_std * torch.randn(val_samples, 1, generator=generator)
            val_target_cols.append(val_inputs @ w + val_noise)
        val_targets = torch.cat(val_target_cols, dim=1)
        val_client_datasets.append(TensorDataset(val_inputs, val_targets))

        test_inputs = torch.randn(test_samples, input_dim, generator=generator) + feature_shift
        test_target_cols = []
        for w in true_weights:
            test_noise = noise_std * torch.randn(test_samples, 1, generator=generator)
            test_target_cols.append(test_inputs @ w + test_noise)
        test_targets = torch.cat(test_target_cols, dim=1)
        test_client_datasets.append(TensorDataset(test_inputs, test_targets))

    return FederatedRegressionData(
        client_datasets=client_datasets,
        input_dim=input_dim,
        num_objectives=num_objectives,
        true_weights=true_weights,
        val_dataset=ConcatDataset(val_client_datasets),
        test_dataset=ConcatDataset(test_client_datasets),
    )
