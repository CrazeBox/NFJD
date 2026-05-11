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
_LEAF_CHARACTER_RE = re.compile(r"^  ([a-zA-Z][a-zA-Z ]*)\. (.*)")
_LEAF_CONT_RE = re.compile(r"^    (.*)")
_LEAF_COE_CHARACTER_RE = re.compile(r"^([a-zA-Z][a-zA-Z ]*)\. (.*)")
_LEAF_COE_CONT_RE = re.compile(r"^(.*)")
_SPEAKER_RE = re.compile(r"^[A-Z][A-Z .'-]{1,48}\.?$")
_PLAY_TITLE_RE = re.compile(r"^(THE )?(TRAGEDY|COMEDY|HISTORY|LIFE|FIRST PART|SECOND PART|THIRD PART|MERRY|TAMING|TEMPEST|WINTER|TWO|TWELFTH|MERCHANT|MUCH ADO|MEASURE|ALL'S WELL|AS YOU LIKE|LOVE'S LABOUR|MIDSUMMER|OTHELLO|HAMLET|MACBETH|KING|RICHARD|HENRY|ROMEO|JULIUS|ANTONY|CORIOLANUS|CYMBELINE|PERICLES|TIMON|TITUS|TROILUS).*")
_NON_SPEAKER_HEADINGS = {
    "ACT", "SCENE", "THE END", "DRAMATIS PERSONAE", "THE SONNETS", "A LOVER'S COMPLAINT",
    "VENUS AND ADONIS", "THE RAPE OF LUCRECE", "THE PASSIONATE PILGRIM",
}
_NON_DIALOGUE_MARKERS = (
    "PROJECT GUTENBERG", "START OF THE PROJECT", "END OF THE PROJECT", "CONTENTS", "THE COMPLETE WORKS",
)


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
            [bash_path, str(preprocess.name), "-s", "niid", "--sf", "1.0", "-k", "0", "-t", "sample", "-tf", "0.8"],
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


def _normalize_key(text: str) -> str:
    text = text.strip().rstrip(".")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_") or "UNKNOWN"


def _is_play_title(line: str) -> bool:
    text = line.strip().rstrip(".")
    if not text or len(text) > 90:
        return False
    if any(marker in text for marker in _NON_DIALOGUE_MARKERS):
        return False
    if any(char.isdigit() for char in text):
        return False
    if text in _NON_SPEAKER_HEADINGS:
        return False
    if text.startswith(("ACT ", "SCENE ", "Enter ", "Exit ", "Exeunt ")):
        return False
    return text.isupper() and bool(_PLAY_TITLE_RE.match(text))


def _make_examples(text: str, sequence_length: int, stride: int) -> tuple[list[str], list[str]]:
    normalized = text.replace("\n", " ")
    normalized = re.sub(r"   *", " ", normalized).strip()
    xs: list[str] = []
    ys: list[str] = []
    if len(normalized) <= sequence_length:
        return xs, ys
    for start in range(0, len(normalized) - sequence_length, max(stride, 1)):
        xs.append(normalized[start:start + sequence_length])
        ys.append(normalized[start + sequence_length])
    return xs, ys


def _remove_nonalphanumerics(text: str) -> str:
    return re.sub(r"\W+", "_", text)


def _leaf_play_and_character(play: str, character: str) -> str:
    return _remove_nonalphanumerics((play + "_" + character).replace(" ", "_"))


def _leaf_match_character(line: str, comedy_of_errors: bool):
    return (_LEAF_COE_CHARACTER_RE if comedy_of_errors else _LEAF_CHARACTER_RE).match(line)


def _leaf_match_continuation(line: str, comedy_of_errors: bool):
    return (_LEAF_COE_CONT_RE if comedy_of_errors else _LEAF_CONT_RE).match(line)


def _split_into_plays_leaf_style(shakespeare_full: str) -> list[tuple[str, dict[str, list[str]]]]:
    plays: list[tuple[str, dict[str, list[str]]]] = []
    slines = shakespeare_full.splitlines(True)[1:]

    author_count = 0
    start_i = 0
    for i, line in enumerate(slines):
        if "by William Shakespeare" in line:
            author_count += 1
        if author_count == 2:
            start_i = max(i - 5, 0)
            break
    slines = slines[start_i:]

    current_character = None
    characters: dict[str, list[str]] | None = None
    comedy_of_errors = False
    for i, line in enumerate(slines):
        if i > 124195 - start_i:
            break
        if "by William Shakespeare" in line:
            current_character = None
            characters = {}
            title = ""
            for back in range(2, 8):
                if i - back >= 0 and slines[i - back].strip():
                    title = slines[i - back].strip()
                    break
            if not title:
                continue
            comedy_of_errors = title == "THE COMEDY OF ERRORS"
            plays.append((title, characters))
            continue

        if characters is None:
            continue

        match = _leaf_match_character(line, comedy_of_errors)
        if match:
            character, snippet = match.group(1).upper(), match.group(2)
            if not (comedy_of_errors and character.startswith("ACT ")):
                characters.setdefault(character, []).append(snippet)
                current_character = character
                continue
            current_character = None
            continue

        if current_character:
            match = _leaf_match_continuation(line, comedy_of_errors)
            if match:
                if comedy_of_errors and match.group(1).startswith("<"):
                    current_character = None
                    continue
                characters[current_character].append(match.group(1))
                continue

    return [(play, chars) for play, chars in plays if len(chars) > 1]


def _build_leaf_style_users_from_text(text: str, sequence_length: int, stride: int) -> dict[str, tuple[list[str], list[str]]]:
    users: dict[str, tuple[list[str], list[str]]] = {}
    plays = _split_into_plays_leaf_style(text)
    for play, characters in plays:
        for character, sound_bites in characters.items():
            if len(sound_bites) <= 2:
                continue
            user = _leaf_play_and_character(play, character)
            xs, ys = _make_examples("\n".join(sound_bites), sequence_length, stride)
            if xs:
                users[user] = (xs, ys)
    return users


def _build_heading_style_users_from_text(text: str, sequence_length: int, stride: int) -> dict[str, tuple[list[str], list[str]]]:
    speaker_chunks: dict[str, list[str]] = {}
    current_play = "UNKNOWN_PLAY"
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
        if any(marker in stripped.upper() for marker in _NON_DIALOGUE_MARKERS):
            continue
        if stripped.startswith(("[", "(")) and stripped.endswith(("]", ")")):
            continue
        if _is_play_title(stripped):
            flush_current()
            current_play = _normalize_key(stripped)
            current_speaker = None
            continue
        if _is_speaker_line(stripped):
            flush_current()
            current_speaker = f"{current_play}::{_normalize_key(stripped)}"
            continue
        if current_speaker is not None:
            current_lines.append(stripped)
    flush_current()

    users: dict[str, tuple[list[str], list[str]]] = {}
    for speaker, chunks in speaker_chunks.items():
        xs, ys = _make_examples(" ".join(chunks), sequence_length, stride)
        if xs:
            users[speaker] = (xs, ys)
    logger.warning("Built %d heading-based Shakespeare clients from raw text.", len(users))
    return users


def _load_raw_text_users(root: Path, sequence_length: int, stride: int) -> dict[str, tuple[list[str], list[str]]]:
    raw_path = next((path for path in _raw_text_candidates(root) if path.is_file()), None)
    if raw_path is None:
        return {}

    logger.warning("Building LEAF-style Shakespeare clients from raw text: %s", raw_path)
    text = raw_path.read_text(encoding="utf-8", errors="replace")
    users = _build_leaf_style_users_from_text(text, sequence_length, stride)
    if users:
        logger.warning("Built %d LEAF-style play-character clients from raw text.", len(users))
        return users

    logger.warning("LEAF-style parser produced no clients; falling back to heading-based parser.")
    return _build_heading_style_users_from_text(text, sequence_length, stride)


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
    return users, "custom_raw_leafstyle_stride"


def _leaf_niid_sample_users(
    users: dict[str, tuple[list[str], list[str]]],
    fraction: float,
    seed: int,
) -> dict[str, tuple[list[str], list[str]]]:
    ordered_users = list(users.keys())
    rng = random.Random(seed)
    rng.shuffle(ordered_users)
    if fraction >= 1.0:
        return {user: users[user] for user in ordered_users}
    target_samples = int(max(fraction, 0.0) * sum(len(users[user][1]) for user in ordered_users))
    sampled: dict[str, tuple[list[str], list[str]]] = {}
    total = 0
    for user in ordered_users:
        xs, ys = users[user]
        remaining = target_samples - total
        if remaining <= 0:
            break
        if len(ys) > remaining:
            # Match LEAF's niid sampling behavior: the final selected user may
            # contribute only the first remaining samples.
            sampled[user] = (xs[:remaining], ys[:remaining])
            total += remaining
        else:
            sampled[user] = (xs, ys)
            total += len(ys)
        if total >= target_samples:
            break
    return sampled


def _leaf_sample_split_indices(total_samples: int, train_fraction: float, sequence_length: int) -> tuple[list[int], list[int]]:
    if total_samples < 2:
        return [], []
    num_train = max(1, int(train_fraction * total_samples))
    if total_samples == 2:
        num_train = 1
    num_train = min(num_train, total_samples - 1)
    train_indices = list(range(num_train))
    test_start = num_train + sequence_length - 1
    test_indices = list(range(test_start, total_samples))
    if not test_indices:
        test_indices = list(range(num_train, total_samples))
    return train_indices, test_indices


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
    stride: int = 1,
    source: str = "auto",
    select_top_clients: bool = True,
    client_selection: str = "leaf",
    sample_fraction: float = 1.0,
    vocab_scope: str = "all",
) -> ShakespeareFederatedData:
    rng = random.Random(seed)
    root_path = resolve_project_path(root) if root is not None else SHAKESPEARE_DIR
    if max_samples_per_client is not None and max_samples_per_client <= 0:
        max_samples_per_client = None
    users, source_note = _load_shakespeare_users(
        root_path,
        auto_prepare=auto_prepare,
        sequence_length=sequence_length,
        stride=stride,
        source=source,
    )
    sampled_user_data = _leaf_niid_sample_users(users, sample_fraction, seed)
    eligible = []
    for user, (xs, ys) in sampled_user_data.items():
        if len(xs) < min_samples_per_client or len(xs) != len(ys):
            continue
        cap_len = min(len(xs), max_samples_per_client) if max_samples_per_client is not None else len(xs)
        train_idx, test_idx = _leaf_sample_split_indices(cap_len, 1.0 - test_fraction, sequence_length)
        if train_idx and test_idx:
            eligible.append(user)
    if len(eligible) < num_clients:
        largest = sorted((len(sampled_user_data[user][0]), user) for user in sampled_user_data)[-10:]
        raise RuntimeError(
            f"Shakespeare has only {len(eligible)} eligible clients with at least {min_samples_per_client} samples, "
            f"fewer than requested num_clients={num_clients}. Largest clients: {largest}"
        )
    if client_selection not in {"top", "random", "leaf"}:
        raise ValueError("client_selection must be one of: top, random, leaf")
    if not select_top_clients and client_selection == "top":
        client_selection = "random"
    if client_selection == "top":
        selected = sorted(eligible, key=lambda user: len(sampled_user_data[user][0]), reverse=True)[:num_clients]
    elif client_selection == "random":
        rng.shuffle(eligible)
        selected = eligible[:num_clients]
    else:
        selected = eligible[:num_clients]
    if vocab_scope not in {"all", "sampled", "selected"}:
        raise ValueError("vocab_scope must be one of: all, sampled, selected")
    if vocab_scope == "all":
        vocab = _build_vocab(users)
    elif vocab_scope == "sampled":
        vocab = _build_vocab(sampled_user_data)
    else:
        vocab = _build_vocab({user: sampled_user_data[user] for user in selected})

    train_datasets: list[Dataset] = []
    test_datasets: list[Dataset] = []
    train_sizes: list[int] = []
    test_sizes: list[int] = []
    for client_id, user in enumerate(selected):
        xs, ys = sampled_user_data[user]
        total_count = len(xs)
        if max_samples_per_client is not None and total_count > max_samples_per_client:
            total_count = max_samples_per_client
        train_idx, test_idx = _leaf_sample_split_indices(total_count, 1.0 - test_fraction, sequence_length)

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
