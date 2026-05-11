from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_project_path(path: str | os.PathLike[str]) -> Path:
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def data_path(*parts: str | os.PathLike[str]) -> Path:
    base = os.environ.get("FEDJD_DATA_ROOT")
    root = resolve_project_path(base) if base else PROJECT_ROOT / "data"
    return root.joinpath(*parts)


def results_path(*parts: str | os.PathLike[str]) -> Path:
    base = os.environ.get("FEDJD_RESULTS_ROOT")
    root = resolve_project_path(base) if base else PROJECT_ROOT / "results"
    return root.joinpath(*parts)
