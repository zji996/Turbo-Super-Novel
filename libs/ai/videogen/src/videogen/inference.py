from __future__ import annotations

import os
from pathlib import Path

from core.paths import data_dir
from .paths import Wan22I2VModelPaths, wan22_i2v_text_encoder_df11_path
from .hf import default_umt5_tokenizer_dir, require_local_tokenizer, umt5_tokenizer_id


class TurboDiffusionInferenceError(RuntimeError):
    pass


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(str(path))


def _check_wan22_i2v_assets(
    *, image_path: Path, model_paths: Wan22I2VModelPaths
) -> None:
    missing: list[Path] = []

    if not image_path.is_file():
        missing.append(image_path)
    if not model_paths.vae_path.is_file():
        missing.append(model_paths.vae_path)
    if not model_paths.high_noise_dit_path.is_file():
        missing.append(model_paths.high_noise_dit_path)
    if not model_paths.low_noise_dit_path.is_file():
        missing.append(model_paths.low_noise_dit_path)

    text_encoder_format = (
        str(os.getenv("TD_TEXT_ENCODER_FORMAT", "df11")).strip().lower()
    )
    df11_path = wan22_i2v_text_encoder_df11_path()
    if text_encoder_format == "df11":
        if not df11_path.is_file() and not model_paths.text_encoder_path.is_file():
            missing.append(df11_path)
            missing.append(model_paths.text_encoder_path)
    else:
        if not model_paths.text_encoder_path.is_file():
            missing.append(model_paths.text_encoder_path)

    tokenizer_dir = default_umt5_tokenizer_dir()
    resolved = umt5_tokenizer_id()
    try:
        require_local_tokenizer(resolved)
    except Exception as exc:
        raise RuntimeError(
            "UMT5 tokenizer directory is missing. This project does not auto-download tokenizers at runtime.\n"
            f"Expected (default): {tokenizer_dir}\n"
            "Fix: `uv run --project apps/api --group tools scripts/cache_umt5_tokenizer.py` "
            "then set `TD_UMT5_TOKENIZER_DIR` (or keep the default path)."
        ) from exc

    if missing:
        unique = sorted({p.resolve() for p in missing})
        lines = "\n".join(f"- {p}" for p in unique)
        raise FileNotFoundError(
            "Required TurboDiffusion model files are missing:\n"
            f"{lines}\n\n"
            "Fix (base+dit weights): `uv run --project apps/api scripts/download_turbodiffusion_models.py`\n"
            "Fix (DF11 text encoder): provide your quantized file under "
            f"`{df11_path}` (download is intentionally not automated yet)."
        )


def run_wan22_i2v(
    *,
    image_path: Path,
    prompt: str,
    output_path: Path,
    model_paths: Wan22I2VModelPaths,
    num_frames: int = 77,
    fps: float = 16,
    num_steps: int = 4,
    seed: int = 0,
    attention_type: str = "sagesla",
    sla_topk: float = 0.1,
    resolution: str = "720p",
    aspect_ratio: str = "16:9",
    adaptive_resolution: bool = True,
    ode: bool = True,
    boundary: float = 0.9,
    sigma_max: float = 200.0,
    **_: object,
) -> Path:
    """
    Run Wan2.2 I2V inference.

    Dev default uses a vendored wrapper that avoids building TurboDiffusion custom CUDA ops.
    """
    _check_wan22_i2v_assets(image_path=image_path, model_paths=model_paths)
    _require_file(image_path)
    _require_file(model_paths.vae_path)
    text_encoder_format = (
        str(os.getenv("TD_TEXT_ENCODER_FORMAT", "df11")).strip().lower()
    )
    if text_encoder_format == "df11":
        df11_path = wan22_i2v_text_encoder_df11_path()
        if df11_path.is_file():
            _require_file(df11_path)
        else:
            _require_file(model_paths.text_encoder_path)
    else:
        _require_file(model_paths.text_encoder_path)
    _require_file(model_paths.high_noise_dit_path)
    _require_file(model_paths.low_noise_dit_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from .vendor.wan22_i2v_infer import generate_wan22_i2v

        out = generate_wan22_i2v(
            image_path=image_path,
            high_noise_model_path=model_paths.high_noise_dit_path,
            low_noise_model_path=model_paths.low_noise_dit_path,
            vae_path=model_paths.vae_path,
            text_encoder_path=model_paths.text_encoder_path,
            prompt=prompt,
            save_path=output_path,
            num_frames=int(num_frames),
            fps=float(fps),
            boundary=boundary,
            num_steps=num_steps,
            sigma_max=sigma_max,
            seed=seed,
            attention_type=attention_type,
            sla_topk=sla_topk,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            adaptive_resolution=adaptive_resolution,
            ode=ode,
        )
        _require_file(out)
        return out
    except Exception as exc:
        log_path = (data_dir() / "turbodiffusion" / "last_infer_error.txt").resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(str(exc), encoding="utf-8", errors="replace")
        raise TurboDiffusionInferenceError(
            f"TurboDiffusion inference failed; see {log_path}"
        ) from exc
