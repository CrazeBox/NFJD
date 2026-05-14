from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from fedjd.paths import resolve_project_path


EMNIST_BYCLASS_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
EMNIST_ASCII_TO_INDEX = {ord(char): idx for idx, char in enumerate(EMNIST_BYCLASS_ALPHABET)}


@dataclass
class VisionFederatedData:
    client_train_datasets: list[Dataset]
    client_test_datasets: list[Dataset]
    global_test_dataset: Dataset
    num_classes: int
    input_channels: int
    dataset_note: str
    num_tasks: int = 1
    task_type: str = "multiclass"


class TargetColumnDataset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: Iterable[int]) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        x, y = self.base_dataset[self.indices[item]]
        return x, torch.tensor([int(y)], dtype=torch.long)


def _normalize_emnist_byclass_label(label) -> int:
    if isinstance(label, str) and len(label) == 1 and not label.isdigit():
        label = ord(label)
    else:
        label = int(label)
    if 0 <= label < len(EMNIST_BYCLASS_ALPHABET):
        return label
    if label in EMNIST_ASCII_TO_INDEX:
        return EMNIST_ASCII_TO_INDEX[label]
    raise ValueError(f"Unsupported FEMNIST/EMNIST byclass label: {label}")


def _emnist_raw_to_upright_tensor(x_tensor: torch.Tensor) -> torch.Tensor:
    """Apply the same EMNIST orientation fix used for torchvision EMNIST."""
    return torch.rot90(torch.flip(x_tensor, dims=[-2]), k=-1, dims=[-2, -1])


def _split_indices(indices: list[int], test_fraction: float, rng: random.Random) -> tuple[list[int], list[int]]:
    copied = list(indices)
    rng.shuffle(copied)
    if len(copied) <= 1:
        return copied, copied
    test_count = max(1, int(round(len(copied) * test_fraction)))
    test_count = min(test_count, len(copied) - 1)
    return copied[test_count:], copied[:test_count]


def _cap_indices(indices: list[int], max_samples: int | None, rng: random.Random) -> list[int]:
    if max_samples is None or len(indices) <= max_samples:
        return list(indices)
    copied = list(indices)
    rng.shuffle(copied)
    return copied[:max_samples]


def _load_cifar10(root: str):
    try:
        from torchvision import datasets, transforms
    except ModuleNotFoundError as exc:
        raise ImportError("CIFAR10 experiments require torchvision.") from exc

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    root_path = resolve_project_path(root)
    train_set = datasets.CIFAR10(root=str(root_path), train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=str(root_path), train=False, download=True, transform=transform)
    return train_set, test_set


def _load_emnist_byclass_global_test(root: str) -> Dataset:
    try:
        from torchvision import datasets, transforms
    except ModuleNotFoundError as exc:
        raise ImportError("FEMNIST global IID evaluation requires torchvision EMNIST.") from exc

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(_emnist_raw_to_upright_tensor),
        transforms.Normalize((0.1736,), (0.3248,)),
    ])
    root_path = resolve_project_path(root)
    emnist_test = datasets.EMNIST(root=str(root_path), split="byclass", train=False, download=True, transform=transform)
    return TargetColumnDataset(emnist_test, range(len(emnist_test)))


def _dirichlet_partition(
    labels: list[int],
    num_clients: int,
    num_classes: int,
    alpha: float,
    rng: np.random.Generator,
    min_samples_per_client: int,
) -> list[list[int]]:
    by_class = [np.where(np.asarray(labels) == class_id)[0] for class_id in range(num_classes)]
    for _ in range(200):
        client_indices = [[] for _ in range(num_clients)]
        for class_indices in by_class:
            shuffled = np.array(class_indices, copy=True)
            rng.shuffle(shuffled)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            cuts = (np.cumsum(proportions)[:-1] * len(shuffled)).astype(int)
            chunks = np.split(shuffled, cuts)
            for client_id, chunk in enumerate(chunks):
                client_indices[client_id].extend(int(i) for i in chunk.tolist())
        if min(len(indices) for indices in client_indices) >= min_samples_per_client:
            for indices in client_indices:
                rng.shuffle(indices)
            return client_indices
    raise RuntimeError(
        f"Failed to create CIFAR10 Dirichlet(alpha={alpha}) split with "
        f"{num_clients} clients and min_samples_per_client={min_samples_per_client}."
    )


def make_cifar10_dirichlet(
    num_clients: int = 50,
    alpha: float = 0.5,
    seed: int = 7,
    root: str = "data/torchvision",
    test_fraction: float = 0.2,
    max_train_samples: int | None = None,
    min_samples_per_client: int = 20,
) -> VisionFederatedData:
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    train_set, test_set = _load_cifar10(root)
    all_indices = _cap_indices(list(range(len(train_set))), max_train_samples, rng_py)
    labels = [int(train_set.targets[i]) for i in all_indices]
    relative_partitions = _dirichlet_partition(
        labels=labels,
        num_clients=num_clients,
        num_classes=10,
        alpha=alpha,
        rng=rng_np,
        min_samples_per_client=min_samples_per_client,
    )
    partitions = [[all_indices[i] for i in rel] for rel in relative_partitions]

    client_train = []
    client_test = []
    for client_id, indices in enumerate(partitions):
        train_indices, test_indices = _split_indices(indices, test_fraction, random.Random(seed * 1009 + client_id))
        client_train.append(TargetColumnDataset(train_set, train_indices))
        client_test.append(TargetColumnDataset(train_set, test_indices))

    return VisionFederatedData(
        client_train_datasets=client_train,
        client_test_datasets=client_test,
        global_test_dataset=TargetColumnDataset(test_set, range(len(test_set))),
        num_classes=10,
        input_channels=3,
        dataset_note=f"torchvision_cifar10_dirichlet_alpha_{alpha}",
    )


def _leaf_json_files(root: Path, split: str | None = None) -> list[Path]:
    if split is None:
        candidates = [root, root / "data" / "train", root / "data" / "test", root / "train", root / "test"]
    else:
        candidates = [root / "data" / split, root / split]
    files: list[Path] = []
    for candidate in candidates:
        if candidate.exists():
            files.extend(sorted(candidate.glob("*.json")))
    deduped = []
    seen = set()
    for path in files:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def _prepare_leaf_femnist(root: str, preprocess_kind: str = "sample") -> Path:
    root_path = resolve_project_path(root)
    if _leaf_json_files(root_path):
        return root_path

    git_path = shutil.which("git")
    bash_path = shutil.which("bash")
    if git_path is None or bash_path is None:
        raise FileNotFoundError(
            "FEMNIST writer JSON files were not found and automatic LEAF setup requires both git and bash. "
            f"Expected LEAF/FEMNIST data under {root_path}."
        )

    root_path.parent.mkdir(parents=True, exist_ok=True)
    leaf_repo = root_path.parent / "leaf"
    if not leaf_repo.exists():
        subprocess.run(
            [git_path, "clone", "--depth", "1", "https://github.com/TalwalkarLab/leaf.git", str(leaf_repo)],
            check=True,
        )

    femnist_dir = leaf_repo / "data" / "femnist"
    preprocess = femnist_dir / "preprocess.sh"
    if not preprocess.exists():
        raise FileNotFoundError(f"LEAF FEMNIST preprocess script not found: {preprocess}")

    data_to_json = femnist_dir / "preprocess" / "data_to_json.py"
    if data_to_json.exists():
        text = data_to_json.read_text(encoding="utf-8")
        legacy_token = "Image.ANTIALIAS"
        if legacy_token in text:
            text = text.replace(
                legacy_token,
                "(Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)",
            )
            data_to_json.write_text(text, encoding="utf-8")

    if preprocess_kind not in {"sample", "full"}:
        raise ValueError("preprocess_kind must be 'sample' or 'full'.")
    # LEAF FEMNIST uses writer identities as users under the non-IID split. The
    # FedMGDA+ paper protocol requires the full split; sample remains the default
    # for smoke tests and lightweight local checks.
    subprocess.run(
        [bash_path, str(preprocess.name), "-s", "niid", "--sf", "1.0", "-k", "0", "-t", preprocess_kind],
        cwd=str(femnist_dir),
        check=True,
    )

    for candidate in (root_path, femnist_dir):
        if _leaf_json_files(candidate):
            return candidate
    raise FileNotFoundError(
        "Automatic LEAF FEMNIST setup finished but no writer JSON files were produced. "
        f"Checked {root_path} and {femnist_dir}."
    )


def _load_leaf_femnist_users(
    root: str,
    auto_prepare: bool = True,
    split: str | None = None,
    preprocess_kind: str = "sample",
) -> dict[str, tuple[list, list[int]]]:
    root_path = resolve_project_path(root)
    if auto_prepare:
        root_path = _prepare_leaf_femnist(root, preprocess_kind=preprocess_kind)
    files = _leaf_json_files(root_path, split=split)
    if not files:
        raise FileNotFoundError(
            "Strict FEMNIST requires LEAF/FEMNIST writer JSON files. "
            "Pass --femnist-leaf-root pointing to a directory containing LEAF data/train and data/test JSON files, "
            "or keep automatic preparation enabled on a server with git and bash."
        )

    users: dict[str, tuple[list, list[int]]] = {}
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for user in payload.get("users", []):
            entry = payload.get("user_data", {}).get(user, {})
            xs = entry.get("x", [])
            ys = [_normalize_emnist_byclass_label(y) for y in entry.get("y", [])]
            if not xs or not ys:
                continue
            if user not in users:
                users[user] = ([], [])
            users[user][0].extend(xs)
            users[user][1].extend(ys)
    return users


def _femnist_tensor_dataset(xs: list, ys: list[int], apply_emnist_orientation_fix: bool = True) -> TensorDataset:
    x_tensor = torch.tensor(xs, dtype=torch.float32)
    if x_tensor.ndim == 2 and x_tensor.shape[1] == 28 * 28:
        x_tensor = x_tensor.view(-1, 1, 28, 28)
    elif x_tensor.ndim == 3 and x_tensor.shape[1:] == (28, 28):
        x_tensor = x_tensor.unsqueeze(1)
    elif x_tensor.ndim == 4 and x_tensor.shape[-1] == 1:
        x_tensor = x_tensor.permute(0, 3, 1, 2)
    if x_tensor.max().item() > 1.5:
        x_tensor = x_tensor / 255.0
    if apply_emnist_orientation_fix:
        x_tensor = _emnist_raw_to_upright_tensor(x_tensor)
    x_tensor = (x_tensor - 0.1736) / 0.3248
    y_tensor = torch.tensor([_normalize_emnist_byclass_label(y) for y in ys], dtype=torch.long).view(-1, 1)
    return TensorDataset(x_tensor.contiguous(), y_tensor)


def make_femnist_writers(
    num_clients: int = 50,
    seed: int = 7,
    leaf_root: str = "data/femnist",
    torchvision_root: str = "data/torchvision",
    test_fraction: float = 0.2,
    min_samples_per_writer: int = 20,
    max_samples_per_writer: int | None = None,
    use_emnist_global_test: bool = True,
    auto_prepare: bool = True,
    apply_emnist_orientation_fix: bool = True,
    use_leaf_train_test_split: bool = False,
    leaf_preprocess_kind: str = "sample",
) -> VisionFederatedData:
    rng = random.Random(seed)
    if use_leaf_train_test_split:
        train_users = _load_leaf_femnist_users(
            leaf_root,
            auto_prepare=auto_prepare,
            split="train",
            preprocess_kind=leaf_preprocess_kind,
        )
        test_users = _load_leaf_femnist_users(
            leaf_root,
            auto_prepare=False,
            split="test",
            preprocess_kind=leaf_preprocess_kind,
        )
        eligible = [
            user for user in sorted(set(train_users) & set(test_users))
            if len(train_users[user][0]) >= min_samples_per_writer
            and len(train_users[user][0]) == len(train_users[user][1])
            and len(test_users[user][0]) == len(test_users[user][1])
            and len(test_users[user][0]) > 0
        ]
        users = train_users
    else:
        users = _load_leaf_femnist_users(leaf_root, auto_prepare=auto_prepare, preprocess_kind=leaf_preprocess_kind)
        eligible = [user for user, (xs, ys) in users.items() if len(xs) >= min_samples_per_writer and len(xs) == len(ys)]
    if len(eligible) < num_clients:
        raise RuntimeError(
            f"FEMNIST has only {len(eligible)} eligible writers, fewer than requested num_clients={num_clients}."
        )
    if num_clients < len(eligible):
        rng.shuffle(eligible)
    selected = eligible[:num_clients]

    client_train = []
    client_test = []
    global_test_parts = []
    for client_id, user in enumerate(selected):
        if use_leaf_train_test_split:
            train_x, train_y = train_users[user]
            test_x, test_y = test_users[user]
            if max_samples_per_writer is not None and len(train_x) > max_samples_per_writer:
                rng_user_cap = random.Random(seed * 917 + client_id)
                train_indices = _cap_indices(list(range(len(train_x))), max_samples_per_writer, rng_user_cap)
                train_x = [train_x[i] for i in train_indices]
                train_y = [train_y[i] for i in train_indices]
        else:
            xs, ys = users[user]
            indices = list(range(len(xs)))
            if max_samples_per_writer is not None and len(indices) > max_samples_per_writer:
                rng_user_cap = random.Random(seed * 917 + client_id)
                indices = _cap_indices(indices, max_samples_per_writer, rng_user_cap)
            train_indices, test_indices = _split_indices(indices, test_fraction, random.Random(seed * 1009 + client_id))
            train_x = [xs[i] for i in train_indices]
            train_y = [ys[i] for i in train_indices]
            test_x = [xs[i] for i in test_indices]
            test_y = [ys[i] for i in test_indices]
        client_train.append(_femnist_tensor_dataset(train_x, train_y, apply_emnist_orientation_fix))
        client_test_ds = _femnist_tensor_dataset(test_x, test_y, apply_emnist_orientation_fix)
        client_test.append(client_test_ds)
        global_test_parts.append(client_test_ds)

    if use_leaf_train_test_split:
        global_x = torch.cat([dataset.tensors[0] for dataset in global_test_parts], dim=0)
        global_y = torch.cat([dataset.tensors[1] for dataset in global_test_parts], dim=0)
        global_test_dataset = TensorDataset(global_x, global_y)
        dataset_note = "leaf_femnist_writer_partition_original_train_test_split"
    elif use_emnist_global_test:
        global_test_dataset: Dataset = _load_emnist_byclass_global_test(torchvision_root)
        dataset_note = "leaf_femnist_writer_partition_emnist_byclass_label_aligned_orientation_aligned_global_test"
    else:
        global_x = torch.cat([dataset.tensors[0] for dataset in global_test_parts], dim=0)
        global_y = torch.cat([dataset.tensors[1] for dataset in global_test_parts], dim=0)
        global_test_dataset = TensorDataset(global_x, global_y)
        dataset_note = "leaf_femnist_writer_partition_client_test_union_global_test"
    return VisionFederatedData(
        client_train_datasets=client_train,
        client_test_datasets=client_test,
        global_test_dataset=global_test_dataset,
        num_classes=62,
        input_channels=1,
        dataset_note=dataset_note,
    )


def bottom_fraction_mean(values: list[float], fraction: float = 0.1) -> float:
    if not values:
        return math.nan
    count = max(1, int(math.ceil(len(values) * fraction)))
    return float(np.mean(sorted(values)[:count]))
