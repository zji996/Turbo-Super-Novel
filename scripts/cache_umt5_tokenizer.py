from __future__ import annotations

import argparse
from pathlib import Path

from libs.turbodiffusion.paths import turbodiffusion_models_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache UMT5 tokenizer files locally for offline inference")
    parser.add_argument("--repo-id", default="google/umt5-xxl", help="HuggingFace repo id (default: google/umt5-xxl)")
    parser.add_argument(
        "--out-dir",
        default=str((turbodiffusion_models_root() / "text-encoder" / "umt5-xxl-tokenizer").resolve()),
        help="Output directory for tokenizer files (default: <MODELS_DIR>/2v/text-encoder/umt5-xxl-tokenizer)",
    )
    parser.add_argument("--revision", default=None, help="Optional HF revision (tag/commit)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `huggingface_hub`. Install via worker env: `uv sync --project apps/worker`."
        ) from exc

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = [
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "added_tokens.json",
    ]

    snapshot_download(
        repo_id=str(args.repo_id),
        repo_type="model",
        revision=args.revision,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )

    print(f"Cached tokenizer to: {out_dir}")
    print("Next: set `TD_UMT5_TOKENIZER_DIR` to this directory and optionally `TD_HF_OFFLINE=1`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
