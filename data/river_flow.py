from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

RIVER_FLOW_DIR = Path("data/river_flow")
RIVER_FLOW_DIR.mkdir(parents=True, exist_ok=True)


def make_river_flow(
    num_clients: int = 10,
    iid: bool = True,
    seed: int = 7,
    num_tasks: int = 8,
) -> dict:
    torch.manual_seed(seed)

    data_path = RIVER_FLOW_DIR / "river_flow.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"River Flow data not found at {data_path}. "
            "Please download from UCI and place the CSV file there. "
            "Expected format: rows=samples, columns=features+targets. "
            "Last {num_tasks} columns are the 8 river flow targets."
        )

    import numpy as np
    raw = np.loadtxt(str(data_path), delimiter=",", skiprows=1)
    n_features = raw.shape[1] - num_tasks

    X = raw[:, :n_features]
    Y = raw[:, n_features:n_features + num_tasks]

    n_samples = len(X)
    n_train = int(n_samples * 0.8)
    train_X, test_X = X[:n_train], X[n_train:]
    train_Y, test_Y = Y[:n_train], Y[n_train:]

    train_mean = train_X.mean(axis=0)
    train_std = train_X.std(axis=0) + 1e-8
    train_X = (train_X - train_mean) / train_std
    test_X = (test_X - train_mean) / train_std

    target_mean = train_Y.mean(axis=0)
    target_std = train_Y.std(axis=0) + 1e-8
    train_Y = (train_Y - target_mean) / target_std
    test_Y = (test_Y - target_mean) / target_std

    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_Y = torch.tensor(train_Y, dtype=torch.float32)
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_Y = torch.tensor(test_Y, dtype=torch.float32)

    if iid:
        indices = torch.randperm(len(train_X))
        per_client = len(train_X) // num_clients
        client_indices = [indices[i*per_client:(i+1)*per_client] for i in range(num_clients)]
    else:
        rng = np.random.RandomState(seed)
        sorted_idx = np.argsort(train_Y[:, 0].numpy())
        chunks = np.array_split(sorted_idx, num_clients)
        rng.shuffle(chunks)
        client_indices = [torch.tensor(c) for c in chunks]

    client_datasets = []
    for idx in client_indices:
        client_datasets.append(TensorDataset(train_X[idx], train_Y[idx]))

    test_dataset = TensorDataset(test_X, test_Y)

    return {
        "client_datasets": client_datasets,
        "test_dataset": test_dataset,
        "input_dim": n_features,
        "num_objectives": num_tasks,
        "num_clients": num_clients,
        "is_iid": iid,
        "target_mean": target_mean,
        "target_std": target_std,
    }
