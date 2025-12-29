from __future__ import annotations

import argparse
import sys
from pathlib import Path

import urllib.error
import urllib.request

from core.paths import models_dir, repo_root

DEFAULT_MODEL_ID = "ZhipuAI/GLM-TTS"
DEFAULT_CAMPPLUS_COMMIT = "c5dc7aecc3b4032032d631b271e767893984f821"


def _default_output_dir() -> Path:
    return (models_dir() / "tts" / "GLM-TTS").resolve()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Download GLM-TTS model files from ModelScope into MODELS_DIR/tts.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"ModelScope model id (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Where to place the model files (default: <MODELS_DIR>/tts/GLM-TTS)",
    )
    parser.add_argument(
        "--revision", default=None, help="Optional ModelScope revision/tag"
    )
    args = parser.parse_args(argv)

    output_dir: Path = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "modelscope is not installed.\n"
            "Recommended (API tools group):\n"
            "  uv sync --project apps/api --group tools\n"
            "  uv run --project apps/api --group tools scripts/download_glm_tts_models.py\n"
            f"import error: {exc}"
        ) from exc

    model_id: str = str(args.model_id)
    print(f"MODELS_DIR={models_dir().resolve()}")
    print(f"Downloading modelscope model={model_id}")
    print(f"Output dir={output_dir}")

    try:
        model_path = snapshot_download(
            model_id,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            revision=args.revision,
        )
    except TypeError:
        model_path = snapshot_download(
            model_id,
            local_dir=str(output_dir),
            revision=args.revision,
        )
    print(f"Done. model_path={model_path}")

    # Optional: copy frontend assets from submodule for local inference.
    # (ModelScope weights do not always include campplus.onnx.)
    frontend_dir = output_dir / "frontend"
    if not (frontend_dir / "campplus.onnx").is_file():
        frontend_dir.mkdir(parents=True, exist_ok=True)
        dst = frontend_dir / "campplus.onnx"

        # 1) Try to download from upstream GitHub at a pinned commit.
        url = f"https://raw.githubusercontent.com/zai-org/GLM-TTS/{DEFAULT_CAMPPLUS_COMMIT}/frontend/campplus.onnx"
        tmp = dst.with_suffix(".onnx.part")
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Turbo-Super-Novel/glm-tts-downloader"}
            )  # noqa: S310
            with urllib.request.urlopen(req) as resp:  # noqa: S310
                tmp.write_bytes(resp.read())
            tmp.replace(dst)
            print(f"Downloaded frontend asset: {dst}")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            if tmp.exists():
                tmp.unlink()

            # 2) Fallback: copy from local submodule if present.
            src = repo_root() / "third_party" / "GLM-TTS" / "frontend" / "campplus.onnx"
            if src.is_file():
                dst.write_bytes(src.read_bytes())
                print(f"Copied frontend asset from submodule: {dst}")
            else:
                print(f"Note: campplus.onnx not found (download failed: {exc}).")
                print(
                    "Provide it under <MODELS_DIR>/tts/GLM-TTS/frontend/campplus.onnx if inference needs it."
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
