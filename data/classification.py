from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import ConcatDataset, TensorDataset


@dataclass
class FederatedData:
    client_datasets: list[TensorDataset]
    input_dim: int
    num_classes: int = 10
    num_tasks: int = 2
    is_noniid: bool = False
    val_dataset: ConcatDataset | None = None
    test_dataset: ConcatDataset | None = None


def make_federated_classification(
    num_clients: int = 10,
    samples_per_client: int = 128,
    input_dim: int = 64,
    num_classes: int = 10,
    num_tasks: int = 2,
    noniid_strength: float = 0.0,
    seed: int = 7,
    val_samples_per_client: int | None = None,
    test_samples_per_client: int | None = None,
) -> FederatedData:
    generator = torch.Generator().manual_seed(seed)
    val_samples = val_samples_per_client if val_samples_per_client is not None else max(samples_per_client // 4, 32)
    test_samples = test_samples_per_client if test_samples_per_client is not None else max(samples_per_client // 2, 64)

    task_heads = []
    for t in range(num_tasks):
        w = torch.randn(num_classes, input_dim, generator=generator) * 0.5
        task_heads.append(w)

    def _sample_client_dataset(class_probs, task_shifts, n_samples):
        labels = torch.multinomial(class_probs, n_samples, replacement=True, generator=generator)
        inputs = torch.randn(n_samples, input_dim, generator=generator) * 0.3

        for c_idx in range(num_classes):
            mask = labels == c_idx
            if mask.any():
                inputs[mask] += task_heads[0][c_idx] * 0.2

        task_labels_list = [labels]
        for shift in task_shifts:
            task_labels_list.append((labels + shift) % num_classes)
        multi_labels = torch.stack(task_labels_list, dim=1)
        return TensorDataset(inputs, multi_labels)

    client_datasets = []
    val_client_datasets = []
    test_client_datasets = []
    for client_idx in range(num_clients):
        if noniid_strength > 0:
            preferred_classes = torch.randint(0, num_classes, (max(1, int(num_classes * noniid_strength)),), generator=generator).tolist()
            class_probs = torch.ones(num_classes)
            for c in preferred_classes:
                class_probs[c] += 5.0 * noniid_strength
            class_probs = class_probs / class_probs.sum()
        else:
            class_probs = torch.ones(num_classes) / num_classes

        task_shifts = []
        for t in range(1, num_tasks):
            shift = torch.randint(0, num_classes, (1,), generator=generator).item()
            task_shifts.append(shift)

        client_datasets.append(_sample_client_dataset(class_probs, task_shifts, samples_per_client))
        val_client_datasets.append(_sample_client_dataset(class_probs, task_shifts, val_samples))
        test_client_datasets.append(_sample_client_dataset(class_probs, task_shifts, test_samples))

    return FederatedData(
        client_datasets=client_datasets,
        input_dim=input_dim,
        num_classes=num_classes,
        num_tasks=num_tasks,
        is_noniid=noniid_strength > 0,
        val_dataset=ConcatDataset(val_client_datasets),
        test_dataset=ConcatDataset(test_client_datasets),
    )


__all__ = ["FederatedData", "make_federated_classification"]
