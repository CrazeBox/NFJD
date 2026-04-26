from __future__ import annotations

import logging
from pathlib import Path

import torch
import torchvision
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

MULTIMNIST_DIR = Path("data/multimnist")
MULTIMNIST_DIR.mkdir(parents=True, exist_ok=True)


def make_multimnist(
    num_clients: int = 10,
    iid: bool = True,
    noniid_classes_per_client: int = 2,
    seed: int = 7,
    image_size: int = 36,
    max_shift: int = 4,
) -> dict:
    torch.manual_seed(seed)

    train_dataset = torchvision.datasets.MNIST(
        root=str(MULTIMNIST_DIR), train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(
        root=str(MULTIMNIST_DIR), train=False, download=True)

    train_images = train_dataset.data.float() / 255.0
    train_labels = train_dataset.targets
    test_images = test_dataset.data.float() / 255.0
    test_labels = test_dataset.targets

    def _sample_indices(num_source, n_samples, rng):
        chunks = []
        remaining = n_samples
        while remaining > 0:
            perm = rng.permutation(num_source)
            take = min(remaining, num_source)
            chunks.append(perm[:take])
            remaining -= take
        return np.concatenate(chunks)

    def _generate_pairs(images, labels, n_samples, rng):
        idx_L = _sample_indices(len(images), n_samples, rng)
        idx_R = _sample_indices(len(images), n_samples, rng)

        canvas = torch.zeros(n_samples, 1, image_size, image_size)
        labels_L = labels[idx_L]
        labels_R = labels[idx_R]

        for i in range(n_samples):
            img_L = images[idx_L[i]].unsqueeze(0).unsqueeze(0)
            img_R = images[idx_R[i]].unsqueeze(0).unsqueeze(0)

            max_offset = image_size - 28
            if max_shift < 0 or max_shift > max_offset:
                raise ValueError(f"max_shift must be in [0, {max_offset}] for image_size={image_size}.")
            off_L_r = rng.randint(0, max_shift + 1)
            off_L_c = rng.randint(0, max_shift + 1)
            off_R_r = rng.randint(max_offset - max_shift, max_offset + 1)
            off_R_c = rng.randint(max_offset - max_shift, max_offset + 1)

            canvas[i, 0, off_L_r:off_L_r+28, off_L_c:off_L_c+28] += img_L[0, 0]
            canvas[i, 0, off_R_r:off_R_r+28, off_R_c:off_R_c+28] += img_R[0, 0]

        canvas.clamp_(0.0, 1.0)

        targets = torch.stack([labels_L, labels_R], dim=1)
        return canvas, targets

    import numpy as np
    rng = np.random.RandomState(seed)
    train_x, train_y = _generate_pairs(train_images, train_labels, 60000, rng)
    test_x, test_y = _generate_pairs(test_images, test_labels, 10000, rng)

    val_size = max(5000, len(train_x) // 10)
    val_x, val_y = train_x[:val_size], train_y[:val_size]
    train_x, train_y = train_x[val_size:], train_y[val_size:]

    if iid:
        indices = torch.randperm(len(train_x))
        per_client = len(train_x) // num_clients
        client_indices = [indices[i*per_client:(i+1)*per_client] for i in range(num_clients)]
    else:
        labels_for_split = train_y[:, 0].numpy()
        min_samples_per_client = max(1, len(train_x) // (num_clients * 10))
        dirichlet_alpha = 0.5
        client_indices = [[] for _ in range(num_clients)]
        for c in range(10):
            class_idx = np.where(labels_for_split == c)[0]
            rng.shuffle(class_idx)
            proportions = rng.dirichlet(np.repeat(dirichlet_alpha, num_clients))
            proportions = proportions * (1 - min_samples_per_client * num_clients / max(len(class_idx), 1))
            proportions = np.maximum(proportions, 0)
            proportions = proportions / proportions.sum()
            splits = (proportions * len(class_idx)).astype(int)
            splits[-1] = len(class_idx) - splits[:-1].sum()
            offset = 0
            for i in range(num_clients):
                client_indices[i].extend(class_idx[offset:offset + splits[i]].tolist())
                offset += splits[i]
        client_indices = [torch.tensor(idx, dtype=torch.long) for idx in client_indices]

    client_datasets = []
    for idx in client_indices:
        cx = train_x[idx]
        cy = train_y[idx]
        client_datasets.append(TensorDataset(cx, cy))

    test_dataset = TensorDataset(test_x, test_y)
    val_dataset = TensorDataset(val_x, val_y)

    return {
        "client_datasets": client_datasets,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "input_dim": (1, image_size, image_size),
        "num_tasks": 2,
        "num_classes": 10,
        "num_clients": num_clients,
        "is_iid": iid,
    }
