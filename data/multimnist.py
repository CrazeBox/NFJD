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
    max_overlap: int = 6,
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

    def _generate_pairs(images, labels, n_samples, rng):
        idx_L = rng.randint(0, len(images), size=n_samples)
        idx_R = rng.randint(0, len(images), size=n_samples)

        canvas = torch.zeros(n_samples, 1, image_size, image_size)
        labels_L = labels[idx_L]
        labels_R = labels[idx_R]

        for i in range(n_samples):
            img_L = images[idx_L[i]].unsqueeze(0).unsqueeze(0)
            img_R = images[idx_R[i]].unsqueeze(0).unsqueeze(0)

            off_L_r = rng.randint(0, max_overlap + 1)
            off_L_c = rng.randint(0, max_overlap + 1)
            off_R_r = rng.randint(image_size - 28 - max_overlap, image_size - 28 + 1)
            off_R_c = rng.randint(image_size - 28 - max_overlap, image_size - 28 + 1)

            canvas[i, 0, off_L_r:off_L_r+28, off_L_c:off_L_c+28] = torch.max(
                canvas[i, 0, off_L_r:off_L_r+28, off_L_c:off_L_c+28], img_L[0, 0])
            canvas[i, 0, off_R_r:off_R_r+28, off_R_c:off_R_c+28] = torch.max(
                canvas[i, 0, off_R_r:off_R_r+28, off_R_c:off_R_c+28], img_R[0, 0])

        targets = torch.stack([labels_L, labels_R], dim=1)
        return canvas, targets

    import numpy as np
    rng = np.random.RandomState(seed)
    train_x, train_y = _generate_pairs(train_images, train_labels, 60000, rng)
    test_x, test_y = _generate_pairs(test_images, test_labels, 10000, rng)

    if iid:
        indices = torch.randperm(len(train_x))
        per_client = len(train_x) // num_clients
        client_indices = [indices[i*per_client:(i+1)*per_client] for i in range(num_clients)]
    else:
        labels_for_split = train_y[:, 0].numpy()
        class_indices = {c: np.where(labels_for_split == c)[0] for c in range(10)}
        shuffled_classes = rng.permutation(10)
        client_indices = [[] for _ in range(num_clients)]
        for i in range(num_clients):
            assigned = shuffled_classes[i * noniid_classes_per_client:(i+1) * noniid_classes_per_client]
            for c in assigned:
                client_indices[i].extend(class_indices[c].tolist())
        client_indices = [torch.tensor(idx) for idx in client_indices]

    client_datasets = []
    for idx in client_indices:
        cx = train_x[idx]
        cy = train_y[idx]
        client_datasets.append(TensorDataset(cx, cy))

    test_dataset = TensorDataset(test_x, test_y)

    return {
        "client_datasets": client_datasets,
        "test_dataset": test_dataset,
        "input_dim": (1, image_size, image_size),
        "num_tasks": 2,
        "num_classes": 10,
        "num_clients": num_clients,
        "is_iid": iid,
    }
