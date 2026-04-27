from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Subset, TensorDataset


def _load_torchvision_dataset(name: str, root: str, train: bool):
    try:
        from torchvision import datasets, transforms
    except ModuleNotFoundError as exc:
        raise ImportError("CIFAR10/FEMNIST experiments require torchvision.") from exc

    if name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        return datasets.CIFAR10(root=root, train=train, download=True, transform=transform), 10, 3
    if name == "femnist":
        # Torchvision does not expose LEAF writer partitions. EMNIST/byclass is
        # the closest built-in source for FEMNIST-style 62-way character data;
        # federation is induced by the partitioner below.
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.rot90(torch.flip(x, dims=[1]), k=-1, dims=[1, 2])),
            transforms.Normalize((0.1736,), (0.3248,)),
        ])
        return datasets.EMNIST(root=root, split="byclass", train=train, download=True, transform=transform), 62, 1
    raise ValueError(f"Unsupported image dataset: {name}")


def _materialize_subset(dataset, indices: list[int]) -> TensorDataset:
    xs = []
    ys = []
    for idx in indices:
        x, y = dataset[idx]
        xs.append(x)
        ys.append(int(y))
    inputs = torch.stack(xs, dim=0)
    labels = torch.tensor(ys, dtype=torch.long).unsqueeze(1)
    return TensorDataset(inputs, labels)


def _balanced_iid_indices(labels: list[int], num_clients: int, rng: random.Random) -> list[list[int]]:
    indices = list(range(len(labels)))
    rng.shuffle(indices)
    return [indices[i::num_clients] for i in range(num_clients)]


def _class_shard_indices(labels: list[int], num_clients: int, num_classes: int, rng: random.Random) -> list[list[int]]:
    by_class = [[] for _ in range(num_classes)]
    for idx, label in enumerate(labels):
        by_class[int(label)].append(idx)
    for class_indices in by_class:
        rng.shuffle(class_indices)

    client_indices = [[] for _ in range(num_clients)]
    class_order = list(range(num_classes))
    rng.shuffle(class_order)
    for rank, class_id in enumerate(class_order):
        chunks = [by_class[class_id][i::num_clients] for i in range(num_clients)]
        offset = rank % num_clients
        for client_id, chunk in enumerate(chunks):
            client_indices[(client_id + offset) % num_clients].extend(chunk)
    for indices in client_indices:
        rng.shuffle(indices)
    return client_indices


def _cap_indices(indices: list[int], max_count: int | None, rng: random.Random) -> list[int]:
    if max_count is None or len(indices) <= max_count:
        return indices
    copied = list(indices)
    rng.shuffle(copied)
    return copied[:max_count]


def make_federated_image_classification(
    dataset: str,
    num_clients: int = 10,
    iid: bool = True,
    seed: int = 7,
    root: str = "data/torchvision",
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> dict:
    dataset = dataset.lower()
    rng = random.Random(seed)
    train_set, num_classes, input_channels = _load_torchvision_dataset(dataset, root, train=True)
    test_set, _, _ = _load_torchvision_dataset(dataset, root, train=False)

    train_labels = [int(train_set[i][1]) for i in range(len(train_set))]
    test_labels = [int(test_set[i][1]) for i in range(len(test_set))]
    train_indices = list(range(len(train_set)))
    test_indices = list(range(len(test_set)))
    train_indices = _cap_indices(train_indices, max_train_samples, rng)
    test_indices = _cap_indices(test_indices, max_eval_samples, rng)

    train_labels_subset = [train_labels[i] for i in train_indices]
    if iid:
        relative_client_indices = _balanced_iid_indices(train_labels_subset, num_clients, rng)
    else:
        relative_client_indices = _class_shard_indices(train_labels_subset, num_clients, num_classes, rng)
    client_indices = [[train_indices[i] for i in rel] for rel in relative_client_indices]

    rng.shuffle(test_indices)
    split = len(test_indices) // 2
    val_indices = test_indices[:split]
    heldout_indices = test_indices[split:]

    client_datasets = [_materialize_subset(train_set, indices) for indices in client_indices]
    val_dataset = _materialize_subset(test_set, val_indices)
    test_dataset = _materialize_subset(test_set, heldout_indices)

    return {
        "client_datasets": client_datasets,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "num_tasks": 1,
        "num_classes": num_classes,
        "input_channels": input_channels,
        "dataset_note": "torchvision_emnist_byclass_proxy" if dataset == "femnist" else "torchvision_cifar10",
    }
