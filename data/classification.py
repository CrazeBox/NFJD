from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import TensorDataset


@dataclass
class FederatedData:
    client_datasets: list[TensorDataset]
    input_dim: int
    num_classes: int = 10
    num_tasks: int = 2
    is_noniid: bool = False


def make_federated_classification(
    num_clients: int = 10,
    samples_per_client: int = 128,
    input_dim: int = 64,
    num_classes: int = 10,
    num_tasks: int = 2,
    noniid_strength: float = 0.0,
    seed: int = 7,
) -> FederatedData:
    generator = torch.Generator().manual_seed(seed)

    task_heads = []
    for t in range(num_tasks):
        w = torch.randn(num_classes, input_dim, generator=generator) * 0.5
        task_heads.append(w)

    client_datasets = []
    for client_idx in range(num_clients):
        if noniid_strength > 0:
            preferred_classes = torch.randint(0, num_classes, (max(1, int(num_classes * noniid_strength)),), generator=generator).tolist()
            class_probs = torch.ones(num_classes)
            for c in preferred_classes:
                class_probs[c] += 5.0 * noniid_strength
            class_probs = class_probs / class_probs.sum()
        else:
            class_probs = torch.ones(num_classes) / num_classes

        labels = torch.multinomial(class_probs, samples_per_client, replacement=True, generator=generator)
        inputs = torch.randn(samples_per_client, input_dim, generator=generator) * 0.3

        for c_idx in range(num_classes):
            mask = labels == c_idx
            if mask.any():
                inputs[mask] += task_heads[0][c_idx] * 0.2

        task_labels_list = [labels]
        for t in range(1, num_tasks):
            shift = torch.randint(0, num_classes, (1,), generator=generator).item()
            t_labels = (labels + shift) % num_classes
            task_labels_list.append(t_labels)

        multi_labels = torch.stack(task_labels_list, dim=1)

        client_datasets.append(TensorDataset(inputs, multi_labels))

    return FederatedData(
        client_datasets=client_datasets,
        input_dim=input_dim,
        num_classes=num_classes,
        num_tasks=num_tasks,
        is_noniid=noniid_strength > 0,
    )


__all__ = ["FederatedData", "make_federated_classification"]
