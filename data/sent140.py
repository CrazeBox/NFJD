from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Dataset, TensorDataset

from fedjd.paths import resolve_project_path


TOKEN_RE = re.compile(r"[a-z0-9_@#']+|[!?.,;:]")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass
class Sent140FederatedData:
    client_train_datasets: list[Dataset]
    client_test_datasets: list[Dataset]
    global_test_dataset: Dataset
    vocab: dict[str, int]
    dataset_note: str
    num_classes: int = 1
    num_tasks: int = 1
    task_type: str = "binary_multitask"


def _leaf_json_files(root: Path, split: str | None = None) -> list[Path]:
    candidates: list[Path]
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


def _prepare_leaf_sent140(root: str) -> Path:
    root_path = resolve_project_path(root)
    if _leaf_json_files(root_path):
        return root_path

    git_path = shutil.which("git")
    bash_path = shutil.which("bash")
    if git_path is None or bash_path is None:
        raise FileNotFoundError(
            "Sent140 LEAF JSON files were not found and automatic setup requires both git and bash. "
            f"Expected LEAF Sent140 data under {root_path}."
        )

    root_path.parent.mkdir(parents=True, exist_ok=True)
    leaf_repo = root_path.parent / "leaf"
    if not leaf_repo.exists():
        subprocess.run(
            [git_path, "clone", "--depth", "1", "https://github.com/TalwalkarLab/leaf.git", str(leaf_repo)],
            check=True,
        )

    sent_dir = leaf_repo / "data" / "sent140"
    preprocess = sent_dir / "preprocess.sh"
    if not preprocess.exists():
        raise FileNotFoundError(f"LEAF Sent140 preprocess script not found: {preprocess}")

    # LEAF Sent140 may require downloading the original Sentiment140 CSV. The
    # sample split is the safest automatic target for smoke tests; full runs can
    # reuse preprocessed JSON copied into --sent140-root.
    subprocess.run(
        [bash_path, str(preprocess.name), "-s", "niid", "--sf", "1.0", "-k", "0", "-t", "sample"],
        cwd=str(sent_dir),
        check=True,
    )

    for candidate in (root_path, sent_dir):
        if _leaf_json_files(candidate):
            return candidate
    raise FileNotFoundError(
        "Automatic LEAF Sent140 setup finished but no JSON files were produced. "
        f"Checked {root_path} and {sent_dir}."
    )


def _extract_text(sample) -> str:
    if isinstance(sample, str):
        return sample
    if isinstance(sample, dict):
        for key in ("text", "tweet", "sentence", "x"):
            value = sample.get(key)
            if isinstance(value, str):
                return value
        return " ".join(str(value) for value in sample.values())
    if isinstance(sample, (list, tuple)):
        # LEAF Sent140 x entries usually contain tweet metadata with text last.
        for value in reversed(sample):
            if isinstance(value, str) and value.strip():
                return value
        return " ".join(str(value) for value in sample)
    return str(sample)


def _normalize_label(label) -> int:
    if isinstance(label, str):
        stripped = label.strip().lower()
        if stripped in {"positive", "pos"}:
            return 1
        if stripped in {"negative", "neg"}:
            return 0
        label = float(stripped)
    value = int(label)
    if value == 4:
        return 1
    if value in {0, 1}:
        return value
    raise ValueError(f"Unsupported Sent140 label: {label!r}")


def _load_split(files: list[Path]) -> dict[str, tuple[list[str], list[int]]]:
    users: dict[str, tuple[list[str], list[int]]] = {}
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        user_data = payload.get("user_data", {})
        for user in payload.get("users", list(user_data.keys())):
            entry = user_data.get(user, {})
            xs_raw = entry.get("x", [])
            ys_raw = entry.get("y", [])
            if not xs_raw or not ys_raw:
                continue
            xs = [_extract_text(sample) for sample in xs_raw]
            ys = [_normalize_label(label) for label in ys_raw]
            if len(xs) != len(ys):
                continue
            if user not in users:
                users[user] = ([], [])
            users[user][0].extend(xs)
            users[user][1].extend(ys)
    return users


def _load_leaf_sent140_users(root: str, auto_prepare: bool = True):
    root_path = resolve_project_path(root)
    if auto_prepare:
        root_path = _prepare_leaf_sent140(root)

    train_files = _leaf_json_files(root_path, "train")
    test_files = _leaf_json_files(root_path, "test")
    if train_files or test_files:
        return _load_split(train_files), _load_split(test_files)

    files = _leaf_json_files(root_path)
    if not files:
        raise FileNotFoundError(
            "Sent140 requires LEAF JSON files. Pass --sent140-root pointing to a directory containing "
            "LEAF data/train and data/test JSON files, or keep automatic preparation enabled."
        )
    combined = _load_split(files)
    return combined, {}


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _build_vocab(texts: list[str], vocab_size: int, min_token_freq: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(_tokenize(text))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in counter.most_common(max(vocab_size - len(vocab), 0)):
        if count < min_token_freq:
            continue
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def _encode_text(text: str, vocab: dict[str, int], sequence_length: int) -> list[int]:
    tokens = _tokenize(text)
    encoded = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens[:sequence_length]]
    if len(encoded) < sequence_length:
        encoded.extend([vocab[PAD_TOKEN]] * (sequence_length - len(encoded)))
    return encoded


def _tensor_dataset(texts: list[str], labels: list[int], vocab: dict[str, int], sequence_length: int) -> TensorDataset:
    x = torch.tensor([_encode_text(text, vocab, sequence_length) for text in texts], dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    return TensorDataset(x, y)


def _split_indices(indices: list[int], test_fraction: float, rng: random.Random) -> tuple[list[int], list[int]]:
    copied = list(indices)
    rng.shuffle(copied)
    if len(copied) <= 1:
        return copied, copied
    test_count = max(1, int(round(len(copied) * test_fraction)))
    test_count = min(test_count, len(copied) - 1)
    return copied[test_count:], copied[:test_count]


def _cap_indices(indices: list[int], max_samples: int | None, rng: random.Random) -> list[int]:
    if max_samples is None or max_samples <= 0 or len(indices) <= max_samples:
        return list(indices)
    copied = list(indices)
    rng.shuffle(copied)
    return copied[:max_samples]


def make_sent140_clients(
    num_clients: int = 100,
    seed: int = 7,
    leaf_root: str = "data/sent140",
    test_fraction: float = 0.2,
    min_samples_per_client: int = 20,
    max_samples_per_client: int | None = None,
    vocab_size: int = 10000,
    min_token_freq: int = 2,
    sequence_length: int = 32,
    auto_prepare: bool = True,
) -> Sent140FederatedData:
    rng = random.Random(seed)
    train_users, test_users = _load_leaf_sent140_users(leaf_root, auto_prepare=auto_prepare)
    all_users = sorted(set(train_users) | set(test_users))
    eligible = []
    for user in all_users:
        train_count = len(train_users.get(user, ([], []))[0])
        test_count = len(test_users.get(user, ([], []))[0])
        total_count = train_count + test_count
        if train_count > 0 and (test_count > 0 or total_count >= min_samples_per_client) and total_count >= min_samples_per_client:
            eligible.append(user)
    if len(eligible) < num_clients:
        raise RuntimeError(
            f"Sent140 has only {len(eligible)} eligible users, fewer than requested num_clients={num_clients}."
        )
    rng.shuffle(eligible)
    selected = eligible[:num_clients]

    selected_train_texts: list[str] = []
    per_user_parts = []
    for client_id, user in enumerate(selected):
        train_x, train_y = train_users.get(user, ([], []))
        test_x, test_y = test_users.get(user, ([], []))
        if not test_x:
            combined_x = list(train_x)
            combined_y = list(train_y)
            indices = _cap_indices(list(range(len(combined_x))), max_samples_per_client, random.Random(seed * 811 + client_id))
            train_idx, test_idx = _split_indices(indices, test_fraction, random.Random(seed * 1009 + client_id))
            train_x = [combined_x[i] for i in train_idx]
            train_y = [combined_y[i] for i in train_idx]
            test_x = [combined_x[i] for i in test_idx]
            test_y = [combined_y[i] for i in test_idx]
        else:
            train_indices = _cap_indices(list(range(len(train_x))), max_samples_per_client, random.Random(seed * 811 + client_id))
            train_x = [train_x[i] for i in train_indices]
            train_y = [train_y[i] for i in train_indices]
        selected_train_texts.extend(train_x)
        per_user_parts.append((train_x, train_y, test_x, test_y))

    vocab = _build_vocab(selected_train_texts, vocab_size=vocab_size, min_token_freq=min_token_freq)
    client_train = []
    client_test = []
    for train_x, train_y, test_x, test_y in per_user_parts:
        client_train.append(_tensor_dataset(train_x, train_y, vocab, sequence_length))
        client_test.append(_tensor_dataset(test_x, test_y, vocab, sequence_length))

    global_test_dataset: Dataset = ConcatDataset(client_test)
    return Sent140FederatedData(
        client_train_datasets=client_train,
        client_test_datasets=client_test,
        global_test_dataset=global_test_dataset,
        vocab=vocab,
        dataset_note="leaf_sent140_user_partition",
    )
