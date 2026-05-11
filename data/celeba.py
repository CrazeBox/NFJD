from __future__ import annotations

import csv
import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from fedjd.paths import resolve_project_path

logger = logging.getLogger(__name__)


def _ensure_optional_dependency(import_name: str, package_name: str) -> None:
    if importlib.util.find_spec(import_name) is not None:
        return
    logger.warning("Missing optional dependency '%s'; attempting to install package '%s'.", import_name, package_name)
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", package_name], check=True)
    except Exception as exc:
        raise ImportError(
            f"CelebA experiments require '{package_name}'. Automatic installation failed; "
            f"install it manually with: {sys.executable} -m pip install {package_name}"
        ) from exc
    if importlib.util.find_spec(import_name) is None:
        raise ImportError(f"Installed '{package_name}', but Python still cannot import '{import_name}'.")


def _import_celeba_transforms():
    _ensure_optional_dependency("torchvision", "torchvision")
    from torchvision import transforms
    return transforms


def _import_pil_image():
    _ensure_optional_dependency("PIL", "Pillow")
    from PIL import Image
    return Image


def _has_celeba_files(root: str) -> bool:
    root_path = resolve_project_path(root)
    file_candidates = [
        root_path / "list_attr_celeba.txt",
        root_path / "list_attr_celeba.csv",
        root_path / "celeba" / "list_attr_celeba.txt",
        root_path / "celeba" / "list_attr_celeba.csv",
    ]
    img_candidates = [
        root_path / "img_align_celeba" / "img_align_celeba",
        root_path / "img_align_celeba",
        root_path / "img_celeba",
        root_path / "celeba" / "img_align_celeba" / "img_align_celeba",
        root_path / "celeba" / "img_align_celeba",
        root_path / "celeba" / "img_celeba",
    ]
    return any(path.is_file() for path in file_candidates) and any(path.is_dir() for path in img_candidates)


def _prepare_torchvision_celeba(root: str) -> None:
    root_path = resolve_project_path(root)
    if _has_celeba_files(root_path):
        return
    _ensure_optional_dependency("torchvision", "torchvision")
    from torchvision import datasets

    try:
        datasets.CelebA(root=str(root_path), split="train", target_type="attr", download=True)
    except Exception as exc:
        raise RuntimeError(
            "Automatic CelebA download failed. CelebA is hosted behind Google Drive and torchvision downloads "
            "can fail due to quota or confirmation limits. Manually place list_attr_celeba.txt, "
            "list_eval_partition.txt, and img_align_celeba under the CelebA root, or pass --no-auto-prepare-celeba."
        ) from exc


class LocalCelebA(Dataset):
    def __init__(self, root: str, split: str, transform=None):
        root_path = resolve_project_path(root)
        self.root = str(root_path)
        self.split = split
        self.transform = transform

        attr_path = self._find_file(root_path, ["list_attr_celeba.txt", "list_attr_celeba.csv"])
        partition_path = self._find_file(root_path, ["list_eval_partition.txt", "list_eval_partition.csv"])
        img_dir = self._find_img_dir(root_path)

        self.attr_names, attr_data = self._load_attrs(attr_path)
        partition = self._load_partition(partition_path)

        split_map = {"train": 0, "valid": 1, "test": 2}
        target_split = split_map.get(split, 0)

        self.filenames = []
        self.attr_list = []
        for fname, s in partition.items():
            if s == target_split and fname in attr_data:
                self.filenames.append(fname)
                self.attr_list.append(attr_data[fname])

        self.attr = torch.tensor(self.attr_list, dtype=torch.float32)
        self.img_dir = img_dir

        logger.info(
            "LocalCelebA: split=%s, %d images, %d attributes from %s",
            split, len(self.filenames), len(self.attr_names), root_path,
        )

    @staticmethod
    def _find_file(root: str | Path, candidates: list[str]) -> str:
        root_path = Path(root)
        for name in candidates:
            path = root_path / name
            if path.is_file():
                return str(path)
            path = root_path / "celeba" / name
            if path.is_file():
                return str(path)
        raise FileNotFoundError(
            f"Cannot find any of {candidates} under {root} or {root}/celeba"
        )

    @staticmethod
    def _find_img_dir(root: str | Path) -> str:
        root_path = Path(root)
        for d in [
            root_path / "img_align_celeba" / "img_align_celeba",
            root_path / "img_align_celeba",
            root_path / "img_celeba",
            root_path / "celeba" / "img_align_celeba" / "img_align_celeba",
            root_path / "celeba" / "img_align_celeba",
            root_path / "celeba" / "img_celeba",
        ]:
            if d.is_dir():
                return str(d)
        raise FileNotFoundError(
            f"Cannot find image directory under {root}"
        )

    @staticmethod
    def _load_attrs(path: str) -> tuple[list[str], dict[str, list[int]]]:
        is_csv = path.endswith(".csv")
        attr_data = {}
        attr_names = []

        if is_csv:
            with open(path, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                if header[0].strip().isdigit():
                    attr_names = [h.strip() for h in header[1:]]
                    num_imgs = int(header[0].strip())
                else:
                    attr_names = [h.strip() for h in header[1:]]
                for row in reader:
                    if len(row) < 2:
                        continue
                    filename = row[0].strip()
                    attrs = [(1 if int(x.strip()) == 1 else 0) for x in row[1:]]
                    attr_data[filename] = attrs
        else:
            with open(path, "r") as f:
                lines = f.readlines()
            num_imgs = int(lines[0].strip())
            attr_names = lines[1].strip().split()
            for line in lines[2:]:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                filename = parts[0]
                attrs = [(1 if int(x) == 1 else 0) for x in parts[1:]]
                attr_data[filename] = attrs

        return attr_names, attr_data

    @staticmethod
    def _load_partition(path: str) -> dict[str, int]:
        is_csv = path.endswith(".csv")
        partition = {}

        if is_csv:
            with open(path, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    if len(row) < 2:
                        continue
                    filename = row[0].strip()
                    s = int(row[1].strip())
                    partition[filename] = s
        else:
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    filename = parts[0]
                    s = int(parts[1])
                    partition[filename] = s

        return partition

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        Image = _import_pil_image()
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        attrs = self.attr[idx]
        return image, attrs


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
    if download:
        _prepare_torchvision_celeba(root)

    transforms = _import_celeba_transforms()
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = LocalCelebA(root=root, split="train", transform=transform)
    val_dataset = LocalCelebA(root=root, split="valid", transform=transform)
    test_dataset = LocalCelebA(root=root, split="test", transform=transform)

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
