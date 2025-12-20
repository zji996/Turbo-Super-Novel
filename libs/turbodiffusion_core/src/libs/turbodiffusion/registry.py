from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


TurboDiffusionModelName = Literal["TurboWan2.2-I2V-A14B-720P"]
TurboDiffusionArtifactGroup = Literal["base", "dit"]


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    name: str
    url: str
    group: TurboDiffusionArtifactGroup
    relative_path: str


def list_artifacts(
    model: TurboDiffusionModelName = "TurboWan2.2-I2V-A14B-720P",
    *,
    quantized: bool = True,
) -> list[ModelArtifact]:
    if model != "TurboWan2.2-I2V-A14B-720P":
        raise ValueError(f"Unknown model: {model}")

    base = [
        ModelArtifact(
            name="Wan2.1_VAE.pth",
            url="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth",
            group="base",
            relative_path="turbodiffusion/checkpoints/Wan2.1_VAE.pth",
        ),
        ModelArtifact(
            name="models_t5_umt5-xxl-enc-bf16.pth",
            url="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth",
            group="base",
            relative_path="turbodiffusion/checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
        ),
    ]

    suffix = "-quant" if quantized else ""
    dit = [
        ModelArtifact(
            name=f"TurboWan2.2-I2V-A14B-high-720P{suffix}.pth",
            url=f"https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P/resolve/main/TurboWan2.2-I2V-A14B-high-720P{suffix}.pth",
            group="dit",
            relative_path=f"turbodiffusion/TurboWan2.2-I2V-A14B-720P/TurboWan2.2-I2V-A14B-high-720P{suffix}.pth",
        ),
        ModelArtifact(
            name=f"TurboWan2.2-I2V-A14B-low-720P{suffix}.pth",
            url=f"https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P/resolve/main/TurboWan2.2-I2V-A14B-low-720P{suffix}.pth",
            group="dit",
            relative_path=f"turbodiffusion/TurboWan2.2-I2V-A14B-720P/TurboWan2.2-I2V-A14B-low-720P{suffix}.pth",
        ),
    ]

    return [*base, *dit]


def iter_artifacts(
    model: TurboDiffusionModelName = "TurboWan2.2-I2V-A14B-720P",
    *,
    quantized: bool = True,
    groups: Iterable[TurboDiffusionArtifactGroup] = ("base", "dit"),
) -> Iterable[ModelArtifact]:
    allowed = set(groups)
    for artifact in list_artifacts(model, quantized=quantized):
        if artifact.group in allowed:
            yield artifact

