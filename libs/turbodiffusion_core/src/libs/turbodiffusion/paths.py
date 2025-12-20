from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from libs.pycore.paths import models_dir, repo_root


TurboDiffusionModelName = Literal["TurboWan2.2-I2V-A14B-720P"]


@dataclass(frozen=True, slots=True)
class Wan22I2VModelPaths:
    vae_path: Path
    text_encoder_path: Path
    high_noise_dit_path: Path
    low_noise_dit_path: Path


def turbodiffusion_models_root() -> Path:
    return (models_dir() / "turbodiffusion").resolve()


def wan22_i2v_model_paths(*, quantized: bool = True) -> Wan22I2VModelPaths:
    root = turbodiffusion_models_root()
    suffix = "-quant" if quantized else ""
    return Wan22I2VModelPaths(
        vae_path=(root / "checkpoints" / "Wan2.1_VAE.pth").resolve(),
        text_encoder_path=(root / "checkpoints" / "models_t5_umt5-xxl-enc-bf16.pth").resolve(),
        high_noise_dit_path=(
            root
            / "TurboWan2.2-I2V-A14B-720P"
            / f"TurboWan2.2-I2V-A14B-high-720P{suffix}.pth"
        ).resolve(),
        low_noise_dit_path=(
            root
            / "TurboWan2.2-I2V-A14B-720P"
            / f"TurboWan2.2-I2V-A14B-low-720P{suffix}.pth"
        ).resolve(),
    )


def turbodiffusion_repo_dir() -> Path:
    return (repo_root() / "third_party" / "TurboDiffusion").resolve()

