from __future__ import annotations

import json
import logging
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Dataset, TensorDataset

from fedjd.paths import data_path, resolve_project_path


logger = logging.getLogger(__name__)

SHAKESPEARE_DIR = data_path("shakespeare")
_SPEAKER_RE = re.compile(r"^[A-Z][A-Z .'-]{1,48}\.?$")
_NON_SPEAKER_HEADINGS = {
    "ACT", "SCENE", "THE END", "DRAMATIS PERSONAE", "THE SONNETS", "A LOVER'S COMPLAINT",
    "VENUS AND ADONIS", "THE RAPE OF LUCRECE", "THE PASSIONATE PILGRIM",
}


@dataclass
class ShakespeareFederatedData:
    client_train_datasets: list[Dataset]
    client_test_datasets: list[Dataset]
    global_test_dataset: Dataset
    vocab: dict[str, int]
    sequence_length: int
    num_clients: int
    dataset_note: str
    client_names: list[str]
    train_sizes: list[int]
    test_sizes: list[int]


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


def _raw_text_candidates(root: Path) -> list[Path]:
    return [
        root / "data" / "raw_data" / "raw_data.txt",
        root / "raw_data" / "raw_data.txt",
        root / "data" / "raw_data.txt",
        root / "raw_data.txt",
    ]


def _is_speaker_line(line: str) -> bool:
    text = line.strip()
    if not text or len(text) > 50:
        return False
    stripped = text.rstrip(".")
    if stripped in _NON_SPEAKER_HEADINGS:
        return False
    if stripped.startswith(("ACT ", "SCENE ", "THE ", "PROJECT GUTENBERG")):
        return False
    if any(char.isdigit() for char in stripped):
        return False
    return bool(_SPEAKER_RE.match(text)) and any(char.isalpha() for char in stripped)


def _load_raw_text_users(root: Path, sequence_length: int, stride: int) -> dict[str, tuple[list[str], list[str]]]:
    raw_path = next((path for path in _raw_text_candidates(root) if path.is_file()), None)
    if raw_path is None:
        return {}

    logger.warning("Building custom Shakespeare speaker clients from raw text: %s", raw_path)
    text = raw_path.read_text(encoding="utf-8", errors="replace")
    speaker_chunks: dict[str, list[str]] = {}
    current_speaker: str | None = None
    current_lines: list[str] = []

    def flush_current() -> None:
        nonlocal current_lines
        if current_speaker is None or not current_lines:
            current_lines = []
            return
        chunk = " ".join(line.strip() for line in current_lines if line.strip())
        if chunk:
            speaker_chunks.setdefault(current_speaker, []).append(chunk)
        current_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("[", "(")) and stripped.endswith(("]", ")")):
            continue
        if _is_speaker_line(stripped):
            flush_current()
            current_speaker = stripped.rstrip(".").replace(" ", "_")
            continue
        if current_speaker is not None:
            current_lines.append(stripped)
    flush_current()

    users: dict[str, tuple[list[str], list[str]]] = {}
    for speaker, chunks in speaker_chunks.items():
        speaker_text = " ".join(chunks)
        speaker_text = re.sub(r"\s+", " ", speaker_text).strip()
        if len(speaker_text) <= sequence_length:
            continue
        xs = []
        ys = []
        for start in range(0, len(speaker_text) - sequence_length, max(stride, 1)):
            xs.append(speaker_text[start:start + sequence_length])
            ys.append(speaker_text[start + sequence_length])
        if xs:
            users[speaker] = (xs, ys)
    logger.warning("Built %d custom Shakespeare speaker clients from raw text.", len(users))
    return users


def _load_leaf_users(root: Path, auto_prepare: bool) -> dict[str, tuple[list[str], list[str]]]:
    data_root = _prepare_leaf_shakespeare(root) if auto_prepare else root
    files = _leaf_json_files(data_root)
    if not files:
        raise FileNotFoundError(
            "Shakespeare requires LEAF JSON files with users/user_data fields. "
            f"Checked {data_root}. See docs/shakespeare_client_objectives.md for manual preparation."
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
    if users:
        return users
    return {}


def _load_shakespeare_users(
    root: Path,
    auto_prepare: bool,
    sequence_length: int,
    stride: int,
    source: str,
) -> tuple[dict[str, tuple[list[str], list[str]]], str]:
    source = source.lower()
    if source not in {"auto", "leaf", "custom"}:
        raise ValueError("source must be one of: auto, leaf, custom")

    if source in {"auto", "leaf"}:
        try:
            users = _load_leaf_users(root, auto_prepare=auto_prepare)
        except FileNotFoundError:
            if source == "leaf":
                raise
            users = {}
        if users:
            return users, "leaf_json"
        if source == "leaf":
            raise RuntimeError("LEAF Shakespeare JSON files were found but contained zero usable clients.")
        logger.warning("LEAF Shakespeare JSON is empty or missing; falling back to custom raw-text construction.")

    users = _load_raw_text_users(root, sequence_length=sequence_length, stride=stride)
    if not users:
        raise FileNotFoundError(
            "Could not build Shakespeare clients. Provide LEAF JSON files or raw_data.txt under "
            f"{root}/data/raw_data/raw_data.txt."
        )
    return users, "custom_raw_speaker_stride"


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
    dataset = TensorDataset(inputs, targets)
    dataset.objective_indices = [client_id]
    return dataset


def make_shakespeare(
    num_clients: int = 20,
    seed: int = 7,
    root: str | Path | None = None,
    auto_prepare: bool = True,
    min_samples_per_client: int = 64,
    max_samples_per_client: int | None = 2000,
    test_fraction: float = 0.2,
    sequence_length: int = 80,
    stride: int = 5,
    source: str = "auto",
    select_top_clients: bool = True,
) -> ShakespeareFederatedData:
    rng = random.Random(seed)
    root_path = resolve_project_path(root) if root is not None else SHAKESPEARE_DIR
    users, source_note = _load_shakespeare_users(
        root_path,
        auto_prepare=auto_prepare,
        sequence_length=sequence_length,
        stride=stride,
        source=source,
    )
    eligible = [user for user, (xs, ys) in users.items() if len(xs) >= min_samples_per_client and len(xs) == len(ys)]
    if len(eligible) < num_clients:
        largest = sorted((len(users[user][0]), user) for user in users)[-10:]
        raise RuntimeError(
            f"Shakespeare has only {len(eligible)} eligible clients with at least {min_samples_per_client} samples, "
            f"fewer than requested num_clients={num_clients}. Largest clients: {largest}"
        )
    if select_top_clients:
        selected = sorted(eligible, key=lambda user: len(users[user][0]), reverse=True)[:num_clients]
    else:
        rng.shuffle(eligible)
        selected = eligible[:num_clients]
    rng.shuffle(selected)
    selected_users = {user: users[user] for user in selected}
    vocab = _build_vocab(selected_users)

    train_datasets: list[Dataset] = []
    test_datasets: list[Dataset] = []
    train_sizes: list[int] = []
    test_sizes: list[int] = []
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
        train_sizes.append(len(train_idx))
        test_sizes.append(len(test_idx))

    logger.info(
        "Loaded Shakespeare: source=%s, clients=%d, vocab=%d, seq_len=%d, stride=%d, "
        "train min/median/max=%d/%d/%d, test min/median/max=%d/%d/%d, root=%s",
        source_note,
        num_clients,
        len(vocab),
        sequence_length,
        stride,
        min(train_sizes),
        sorted(train_sizes)[len(train_sizes) // 2],
        max(train_sizes),
        min(test_sizes),
        sorted(test_sizes)[len(test_sizes) // 2],
        max(test_sizes),
        root_path,
    )
    return ShakespeareFederatedData(
        client_train_datasets=train_datasets,
        client_test_datasets=test_datasets,
        global_test_dataset=ConcatDataset(test_datasets),
        vocab=vocab,
        sequence_length=sequence_length,
        num_clients=num_clients,
        dataset_note=f"shakespeare_{source_note}_client_objectives",
        client_names=selected,
        train_sizes=train_sizes,
        test_sizes=test_sizes,
    )
