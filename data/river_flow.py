from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

RIVER_FLOW_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00444/river_flow.zip"
RIVER_FLOW_DIR = Path("data/river_flow")


def _download_river_flow(data_dir: Path):
    import zipfile
    import urllib.request

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "river_flow.zip"

    if not zip_path.exists():
        logger.info("Downloading River Flow dataset from UCI...")
        urllib.request.urlretrieve(RIVER_FLOW_URL, str(zip_path))
        logger.info("Download complete: %s", zip_path)

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(data_dir))
    logger.info("Extracted to %s", data_dir)


def _load_csv_robust(data_path: Path) -> np.ndarray:
    try:
        import pandas as pd
        df = pd.read_csv(str(data_path))
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna()
        if len(df) == 0:
            raise ValueError(f"No valid rows after cleaning {data_path}")
        return df.values.astype(np.float64)
    except ImportError:
        pass

    raw = np.genfromtxt(str(data_path), delimiter=",", skip_header=1)
    mask = ~np.isnan(raw).any(axis=1)
    raw = raw[mask]
    if len(raw) == 0:
        raise ValueError(f"No valid rows after cleaning {data_path}")
    return raw


def make_river_flow(
    num_clients: int = 10,
    iid: bool = True,
    seed: int = 7,
    num_tasks: int = 8,
    download: bool = True,
) -> dict:
    torch.manual_seed(seed)

    data_path = RIVER_FLOW_DIR / "river_flow.csv"
    if not data_path.exists():
        alt_paths = list(RIVER_FLOW_DIR.glob("**/*.csv"))
        if alt_paths:
            rf1_candidates = [p for p in alt_paths if "rf1" in p.name.lower()]
            if rf1_candidates:
                data_path = rf1_candidates[0]
                logger.info("Using RF1 CSV: %s", data_path)
            else:
                data_path = alt_paths[0]
                logger.info("Using alternative CSV: %s", data_path)
        elif download:
            _download_river_flow(RIVER_FLOW_DIR)
            alt_paths = list(RIVER_FLOW_DIR.glob("**/*.csv"))
            if alt_paths:
                rf1_candidates = [p for p in alt_paths if "rf1" in p.name.lower()]
                if rf1_candidates:
                    data_path = rf1_candidates[0]
                else:
                    data_path = alt_paths[0]
            else:
                raise FileNotFoundError(
                    f"River Flow CSV not found after download in {RIVER_FLOW_DIR}. "
                    "Please manually download RF1 from "
                    "https://www.kaggle.com/datasets/samanemami/river-flowrf1 "
                    "and place the CSV file at data/river_flow/RF1.csv. "
                    "Expected format: rows=samples, columns=features+targets. "
                    f"Last {num_tasks} columns are the {num_tasks} river flow targets."
                )
        else:
            raise FileNotFoundError(
                f"River Flow data not found at {data_path}. "
                "Please download RF1 from "
                "https://www.kaggle.com/datasets/samanemami/river-flowrf1 "
                "and place the CSV file at data/river_flow/RF1.csv. "
                "Expected format: rows=samples, columns=features+targets. "
                f"Last {num_tasks} columns are the {num_tasks} river flow targets."
            )

    raw = _load_csv_robust(data_path)

    if raw.shape[1] < num_tasks + 1:
        raise ValueError(
            f"CSV has {raw.shape[1]} columns but need at least {num_tasks + 1} "
            f"({num_tasks} targets + at least 1 feature). "
            f"File: {data_path}"
        )

    n_features = raw.shape[1] - num_tasks

    X = raw[:, :n_features]
    Y = raw[:, n_features:n_features + num_tasks]

    n_samples = len(X)
    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.1)
    train_X, val_X, test_X = X[:n_train], X[n_train:n_train + n_val], X[n_train + n_val:]
    train_Y, val_Y, test_Y = Y[:n_train], Y[n_train:n_train + n_val], Y[n_train + n_val:]

    train_mean = train_X.mean(axis=0)
    train_std = train_X.std(axis=0) + 1e-8
    train_X = (train_X - train_mean) / train_std
    val_X = (val_X - train_mean) / train_std
    test_X = (test_X - train_mean) / train_std

    target_mean = train_Y.mean(axis=0)
    target_std = train_Y.std(axis=0) + 1e-8
    train_Y = (train_Y - target_mean) / target_std
    val_Y = (val_Y - target_mean) / target_std
    test_Y = (test_Y - target_mean) / target_std

    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_Y = torch.tensor(train_Y, dtype=torch.float32)
    val_X = torch.tensor(val_X, dtype=torch.float32)
    val_Y = torch.tensor(val_Y, dtype=torch.float32)
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

    val_dataset = TensorDataset(val_X, val_Y)
    test_dataset = TensorDataset(test_X, test_Y)

    return {
        "client_datasets": client_datasets,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "input_dim": n_features,
        "num_objectives": num_tasks,
        "num_clients": num_clients,
        "is_iid": iid,
        "target_mean": target_mean,
        "target_std": target_std,
    }
