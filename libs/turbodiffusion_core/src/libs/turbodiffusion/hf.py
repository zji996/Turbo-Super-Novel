from __future__ import annotations

import os
from pathlib import Path

from libs.pycore.paths import data_dir, models_dir


def env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def configure_hf_home() -> Path:
    """
    Ensure HuggingFace cache root is inside repo-local DATA_DIR by default.

    This affects both `transformers` and `huggingface_hub` caching behavior.
    """
    if os.getenv("HF_HOME") is None:
        os.environ["HF_HOME"] = str((data_dir() / "hf").resolve())
    return Path(os.environ["HF_HOME"]).expanduser().resolve()


def configure_hf_offline() -> bool:
    """
    Enable HF/Transformers offline mode when requested.

    Opt-in env vars:
    - `TD_HF_OFFLINE=1` or `TD_LOCAL_ONLY=1`
    """
    offline = env_bool("TD_HF_OFFLINE", False) or env_bool("TD_LOCAL_ONLY", False)
    if not offline:
        return False

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    return True


def default_umt5_tokenizer_dir() -> Path:
    candidates = [
        (models_dir() / "text-encoder" / "umt5-xxl-tokenizer").resolve(),
        (models_dir() / "text-encoder-df11" / "umt5-xxl-tokenizer").resolve(),
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def umt5_tokenizer_id() -> str:
    """
    Tokenizer identifier used by `transformers.AutoTokenizer.from_pretrained`.

    Priority:
    1) `TD_UMT5_TOKENIZER_DIR` (local path)
    2) `TD_UMT5_TOKENIZER` (local path)
    3) `<MODELS_DIR>/text-encoder/umt5-xxl-tokenizer`

    Note: inference intentionally does not auto-download tokenizer files. If the directory is
    missing, we fail fast and ask the user to run `scripts/cache_umt5_tokenizer.py`.
    """
    tokenizer_dir = (os.getenv("TD_UMT5_TOKENIZER_DIR") or "").strip()
    if tokenizer_dir:
        resolved = Path(tokenizer_dir).expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(str(resolved))
        return str(resolved)

    tokenizer_id = (os.getenv("TD_UMT5_TOKENIZER") or "").strip()
    if tokenizer_id:
        maybe_path = Path(tokenizer_id).expanduser()
        if maybe_path.is_dir():
            return str(maybe_path.resolve())
        raise FileNotFoundError(str(maybe_path.resolve()))

    default_dir = default_umt5_tokenizer_dir()
    return str(default_dir)


def require_local_tokenizer(tokenizer_id: str) -> None:
    """
    Fail early if tokenizer directory is missing.
    """
    maybe_path = Path(tokenizer_id).expanduser()
    if maybe_path.is_dir():
        return

    default_te = (models_dir() / "text-encoder" / "umt5-xxl-tokenizer").resolve()
    default_df11 = (models_dir() / "text-encoder-df11" / "umt5-xxl-tokenizer").resolve()
    raise RuntimeError(
        "UMT5 tokenizer is missing. This project does not auto-download tokenizers at runtime. "
        "Fix: download it with `uv run --project apps/worker scripts/cache_umt5_tokenizer.py` "
        "and point `TD_UMT5_TOKENIZER_DIR` to one of:\n"
        f"- {default_te}\n"
        f"- {default_df11}"
    )
