from __future__ import annotations

import logging
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets import CelebA

logger = logging.getLogger(__name__)


def make_celeba(
    num_clients: int,
    root: str = "data/celeba",
    download: bool = True,
    iid: bool = True,
    num_tasks: int = 4,
    val_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[List[Dataset], List[Dataset], List[Dataset]]:
    if seed is not None:
        torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = CelebA(
        root=root, split="train", transform=transform, download=download,
    )
    val_dataset = CelebA(
        root=root, split="valid", transform=transform, download=download,
    )
    test_dataset = CelebA(
        root=root, split="test", transform=transform, download=download,
    )

    attributes = train_dataset.attr_names
    selected_attributes = attributes[:num_tasks]
    attr_indices = [train_dataset.attr_names.index(attr) for attr in selected_attributes]
    logger.info("Selected attributes: %s", selected_attributes)

    if iid:
        train_datasets = _iid_split(train_dataset, num_clients, attr_indices)
        val_datasets = _iid_split(val_dataset, num_clients, attr_indices)
        test_datasets = _iid_split(test_dataset, num_clients, attr_indices)
    else:
        rng = np.random.RandomState(seed)
        train_datasets = _noniid_split(train_dataset, num_clients, attr_indices, rng)
        val_datasets = _noniid_split(val_dataset, num_clients, attr_indices, rng)
        test_datasets = _noniid_split(test_dataset, num_clients, attr_indices, rng)

    return train_datasets, val_datasets, test_datasets


def _iid_split(base_dataset, num_clients, attr_indices):
    total_len = len(base_dataset)
    per_client = total_len // num_clients
    datasets = []
    for i in range(num_clients):
        start = i * per_client
        end = (i + 1) * per_client if i < num_clients - 1 else total_len
        indices = list(range(start, end))
        datasets.append(CelebAAttrDataset(base_dataset, indices, attr_indices))
    return datasets


def _noniid_split(base_dataset, num_clients, attr_indices, rng):
    all_attrs = base_dataset.attr[:, attr_indices[0]].numpy()
    dirichlet_alpha = 0.5
    client_indices = [[] for _ in range(num_clients)]

    for attr_val in [0, 1]:
        mask = (all_attrs == attr_val)
        class_idx = np.where(mask)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.repeat(dirichlet_alpha, num_clients))
        proportions = proportions / proportions.sum()
        splits = (proportions * len(class_idx)).astype(int)
        splits[-1] = len(class_idx) - splits[:-1].sum()
        offset = 0
        for i in range(num_clients):
            client_indices[i].extend(class_idx[offset:offset + splits[i]].tolist())
            offset += splits[i]

    datasets = []
    for indices in client_indices:
        datasets.append(CelebAAttrDataset(base_dataset, indices, attr_indices))
    return datasets


class CelebAAttrDataset(Dataset):
    def __init__(self, base_dataset, indices, attr_indices):
        self.base_dataset = base_dataset
        self.indices = indices
        self.attr_indices = attr_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        img, attrs = self.base_dataset[data_idx]
        selected_attrs = attrs[self.attr_indices].float()
        return img, selected_attrs
