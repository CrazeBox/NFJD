from __future__ import annotations

import json
import logging
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Dataset, TensorDataset

from fedjd.paths import data_path, resolve_project_path


logger = logging.getLogger(__name__)

SHAKESPEARE_DIR = data_path("shakespeare")


@dataclass
class ShakespeareFederatedData:
    client_train_datasets: list[Dataset]
    client_test_datasets: list[Dataset]
    global_test_dataset: Dataset
    vocab: dict[str, int]
    sequence_length: int
    num_clients: int
    dataset_note: str


def _leaf_json_files(root: Path) -> list[Path]:
    candidates = [root, root / "data" / "train", root / "data" / "test", root / "train", root / "test"]
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


def _prepare_leaf_shakespeare(root: Path) -> Path:
    if _leaf_json_files(root):
        return root

    git_path = shutil.which("git")
    bash_path = shutil.which("bash")
    if git_path is None or bash_path is None:
        raise FileNotFoundError(
            "Shakespeare LEAF JSON files were not found and automatic setup requires both git and bash. "
            f"Expected data under {root}. See docs/shakespeare_client_objectives.md for manual preparation."
        )

    root.parent.mkdir(parents=True, exist_ok=True)
    leaf_repo = root.parent / "leaf"
    if not leaf_repo.exists():
        subprocess.run(
            [git_path, "clone", "--depth", "1", "https://github.com/TalwalkarLab/leaf.git", str(leaf_repo)],
            check=True,
        )

    shakespeare_dir = leaf_repo / "data" / "shakespeare"
    preprocess = shakespeare_dir / "preprocess.sh"
    if not preprocess.exists():
        raise FileNotFoundError(f"LEAF Shakespeare preprocess script not found: {preprocess}")

    try:
        subprocess.run(
            [bash_path, str(preprocess.name), "-s", "niid", "--sf", "1.0", "-k", "0", "-t", "sample"],
            cwd=str(shakespeare_dir),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Automatic LEAF Shakespeare preprocessing failed. Prepare the dataset manually with LEAF's "
            "data/shakespeare/preprocess.sh and pass --no-auto-prepare-shakespeare. "
            "See docs/shakespeare_client_objectives.md for details."
        ) from exc

    for candidate in (root, shakespeare_dir):
        if _leaf_json_files(candidate):
            return candidate
    raise FileNotFoundError(
        "Automatic LEAF Shakespeare setup finished but no JSON files were produced. "
        f"Checked {root} and {shakespeare_dir}. See docs/shakespeare_client_objectives.md for manual preparation."
    )


def _load_leaf_users(root: Path, auto_prepare: bool) -> dict[str, tuple[list[str], list[str]]]:
    data_root = _prepare_leaf_shakespeare(root) if auto_prepare else root
    files = _leaf_json_files(data_root)
    if not files:
        raise FileNotFoundError(
            "Shakespeare requires LEAF JSON files with users/user_data fields. "
            f"Pass --shakespeare-root pointing to prepared LEAF data. Checked {data_root}. "
            "See docs/shakespeare_client_objectives.md for manual preparation."
        )

    users: dict[str, tuple[list[str], list[str]]] = {}
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for user in payload.get("users", []):
            entry = payload.get("user_data", {}).get(user, {})
            xs = entry.get("x", [])
            ys = entry.get("y", [])
            if not xs or not ys:
                continue
            if user not in users:
                users[user] = ([], [])
            users[user][0].extend(str(x) for x in xs)
            users[user][1].extend(str(y) for y in ys)
    return users


def _build_vocab(users: dict[str, tuple[list[str], list[str]]]) -> dict[str, int]:
    chars = set()
    for xs, ys in users.values():
        for x in xs:
            chars.update(x)
        for y in ys:
            chars.update(y)
    vocab = {"<unk>": 0}
    for char in sorted(chars):
        if char not in vocab:
            vocab[char] = len(vocab)
    return vocab


def _encode_user_dataset(
    xs: list[str],
    ys: list[str],
    vocab: dict[str, int],
    client_id: int,
    sequence_length: int,
) -> TensorDataset:
    unk = vocab["<unk>"]
    encoded_x = []
    encoded_y = []
    for x, y in zip(xs, ys):
        seq = [vocab.get(char, unk) for char in x[:sequence_length]]
        if len(seq) < sequence_length:
            seq.extend([unk] * (sequence_length - len(seq)))
        target_char = y[0] if y else ""
        encoded_x.append(seq)
        encoded_y.append([vocab.get(target_char, unk), client_id])
    inputs = torch.tensor(encoded_x, dtype=torch.long)
    targets = torch.tensor(encoded_y, dtype=torch.long)
    return TensorDataset(inputs, targets)


def make_shakespeare(
    num_clients: int = 20,
    seed: int = 7,
    root: str | Path | None = None,
    auto_prepare: bool = True,
    min_samples_per_client: int = 64,
    max_samples_per_client: int | None = 2000,
    test_fraction: float = 0.2,
    sequence_length: int = 80,
) -> ShakespeareFederatedData:
    rng = random.Random(seed)
    root_path = resolve_project_path(root) if root is not None else SHAKESPEARE_DIR
    users = _load_leaf_users(root_path, auto_prepare=auto_prepare)
    eligible = [user for user, (xs, ys) in users.items() if len(xs) >= min_samples_per_client and len(xs) == len(ys)]
    if len(eligible) < num_clients:
        raise RuntimeError(
            f"Shakespeare has only {len(eligible)} eligible clients, fewer than requested num_clients={num_clients}."
        )
    rng.shuffle(eligible)
    selected = eligible[:num_clients]
    selected_users = {user: users[user] for user in selected}
    vocab = _build_vocab(selected_users)

    train_datasets: list[Dataset] = []
    test_datasets: list[Dataset] = []
    for client_id, user in enumerate(selected):
        xs, ys = users[user]
        indices = list(range(len(xs)))
        rng_client = random.Random(seed * 1009 + client_id)
        rng_client.shuffle(indices)
        if max_samples_per_client is not None and len(indices) > max_samples_per_client:
            indices = indices[:max_samples_per_client]
        if len(indices) <= 1:
            train_idx = test_idx = indices
        else:
            test_count = max(1, int(round(len(indices) * test_fraction)))
            test_count = min(test_count, len(indices) - 1)
            test_idx = indices[:test_count]
            train_idx = indices[test_count:]

        train_datasets.append(
            _encode_user_dataset([xs[i] for i in train_idx], [ys[i] for i in train_idx], vocab, client_id, sequence_length)
        )
        test_datasets.append(
            _encode_user_dataset([xs[i] for i in test_idx], [ys[i] for i in test_idx], vocab, client_id, sequence_length)
        )

    logger.info(
        "Loaded Shakespeare: %d clients, vocab=%d, seq_len=%d, root=%s",
        num_clients, len(vocab), sequence_length, root_path,
    )
    return ShakespeareFederatedData(
        client_train_datasets=train_datasets,
        client_test_datasets=test_datasets,
        global_test_dataset=ConcatDataset(test_datasets),
        vocab=vocab,
        sequence_length=sequence_length,
        num_clients=num_clients,
        dataset_note="leaf_shakespeare_speaker_client_objectives",
    )
