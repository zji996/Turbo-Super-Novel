from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache
def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "AGENTS.md").is_file() and (parent / "apps").is_dir() and (parent / "libs").is_dir():
            return parent

    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        if (parent / "AGENTS.md").is_file() and (parent / "apps").is_dir() and (parent / "libs").is_dir():
            return parent

    return cwd


def _dir_from_env(env_key: str, default_name: str) -> Path:
    value = os.getenv(env_key)
    if value:
        return Path(value).expanduser().resolve()
    return (repo_root() / default_name).resolve()


def models_dir() -> Path:
    return _dir_from_env("MODELS_DIR", "models")


def data_dir() -> Path:
    return _dir_from_env("DATA_DIR", "data")


def logs_dir() -> Path:
    return _dir_from_env("LOG_DIR", "logs")


def paths_summary() -> dict[str, str]:
    return {
        "models_dir": str(models_dir()),
        "data_dir": str(data_dir()),
        "logs_dir": str(logs_dir()),
    }
