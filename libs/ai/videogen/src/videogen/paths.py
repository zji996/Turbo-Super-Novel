from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from core.paths import models_dir

TurboDiffusionModelName = Literal["TurboWan2.2-I2V-A14B-720P"]


@dataclass(frozen=True, slots=True)
class Wan22I2VModelPaths:
    vae_path: Path
    text_encoder_path: Path
    high_noise_dit_path: Path
    low_noise_dit_path: Path


def turbodiffusion_models_root() -> Path:
    """
    Root directory that contains TurboDiffusion-related model artifacts.

    Repo convention (2025-12): keep TurboDiffusion artifacts under `models/2v/`
    to make it explicit that these weights produce "to-video" outputs.

    Backward compatible: if `models/2v/` does not exist but legacy directories are
    found under `models/`, we keep using the legacy layout.
    """
    root = models_dir().resolve()
    v2_root = (root / "2v").resolve()

    legacy_markers = (
        root / "vae",
        root / "text-encoder",
        root / "text-encoder-df11",
        root / "wan2.2-i2v-quant",
        root / "wan2.2-i2v",
    )

    v2_markers = (
        v2_root / "vae",
        v2_root / "text-encoder",
        v2_root / "text-encoder-df11",
        v2_root / "wan2.2-i2v-quant",
        v2_root / "wan2.2-i2v",
    )

    # Prefer the new layout only when it actually contains TurboDiffusion artifacts, so that
    # an empty pre-created `models/2v/` directory doesn't break existing installs.
    if v2_root.is_dir():
        if any(p.exists() for p in v2_markers):
            return v2_root
        if any(p.exists() for p in legacy_markers):
            return root
        return v2_root
    if any(p.exists() for p in legacy_markers):
        return root

    return v2_root


def wan22_i2v_text_encoder_df11_dir() -> Path:
    return (turbodiffusion_models_root() / "text-encoder-df11").resolve()


def wan22_i2v_text_encoder_df11_path() -> Path:
    return (
        wan22_i2v_text_encoder_df11_dir() / "models_t5_umt5-xxl-enc-df11.safetensors"
    ).resolve()


def wan22_i2v_model_paths(*, quantized: bool = True) -> Wan22I2VModelPaths:
    root = turbodiffusion_models_root()
    dit_dir = root / ("wan2.2-i2v-quant" if quantized else "wan2.2-i2v")
    suffix = "-quant" if quantized else ""
    return Wan22I2VModelPaths(
        vae_path=(root / "vae" / "Wan2.1_VAE.pth").resolve(),
        text_encoder_path=(
            root / "text-encoder" / "models_t5_umt5-xxl-enc-bf16.pth"
        ).resolve(),
        high_noise_dit_path=(
            dit_dir / f"TurboWan2.2-I2V-A14B-high-720P{suffix}.pth"
        ).resolve(),
        low_noise_dit_path=(
            dit_dir / f"TurboWan2.2-I2V-A14B-low-720P{suffix}.pth"
        ).resolve(),
    )
