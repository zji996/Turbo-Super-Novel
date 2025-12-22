from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from libs.pycore.paths import repo_root
from libs.turbodiffusion.paths import turbodiffusion_repo_dir, wan22_i2v_model_paths


def _ensure_upstream_on_syspath() -> None:
    repo_dir = turbodiffusion_repo_dir()
    upstream_pkg = (repo_dir / "turbodiffusion").resolve()
    if str(upstream_pkg) not in sys.path:
        sys.path.insert(0, str(upstream_pkg))


def _ensure_dfloat11_on_syspath() -> None:
    dfloat11_repo = (repo_root() / "third_party" / "DFloat11").resolve()
    if str(dfloat11_repo) not in sys.path:
        sys.path.insert(0, str(dfloat11_repo))


def _load_state_dict(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        return obj
    raise TypeError(f"Unsupported checkpoint type: {type(obj)}")


def _build_pattern_dict_for_umt5_encoder() -> dict[str, list[str]]:
    return {
        "token_embedding$": [],
        "blocks\\.\\d+$": [
            "attn.q",
            "attn.k",
            "attn.v",
            "attn.o",
            "ffn.gate.0",
            "ffn.fc1",
            "ffn.fc2",
        ],
    }


def _load_umt5_xxl_encoder_model(*, ckpt_path: Path) -> torch.nn.Module:
    _ensure_upstream_on_syspath()

    from rcm.utils.umt5 import umt5_xxl

    state_dict = _load_state_dict(ckpt_path)
    model = umt5_xxl(encoder_only=True, dtype=torch.bfloat16, device="meta")

    try:
        model.load_state_dict(state_dict, strict=True, assign=True)
        return model.to("cpu").eval()
    except TypeError:
        model.load_state_dict(state_dict, strict=True)
        return model.eval()
    except RuntimeError:
        cpu_model = umt5_xxl(encoder_only=True, dtype=torch.bfloat16, device="cpu")
        cpu_model.load_state_dict(state_dict, strict=True)
        return cpu_model.eval()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_ckpt = wan22_i2v_model_paths().text_encoder_path
    default_save_dir = (default_ckpt.parent.parent / "text-encoder-df11").resolve()
    default_output_name = "models_t5_umt5-xxl-enc-df11.safetensors"

    parser = argparse.ArgumentParser(description="Compress TurboDiffusion UMT5-XXL encoder (BF16 .pth) into DF11.")
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=default_ckpt,
        help="Path to BF16 UMT5 encoder checkpoint (.pth).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=default_save_dir,
        help="Output directory for DF11 (writes config.json + a .safetensors file).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=default_output_name,
        help="Output safetensors file name (inside --save-dir).",
    )
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="Enable GPU bit-exact check during compression (requires CUDA + cupy).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ckpt_path: Path = args.ckpt_path
    save_dir: Path = args.save_dir
    output_name: str = str(args.output_name)

    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    try:
        from dfloat11 import compress_model  # type: ignore[import-not-found]
    except Exception:
        _ensure_dfloat11_on_syspath()
        from dfloat11 import compress_model  # type: ignore[import-not-found]

    model = _load_umt5_xxl_encoder_model(ckpt_path=ckpt_path)

    pattern_dict = _build_pattern_dict_for_umt5_encoder()
    compress_model(
        model=model,
        pattern_dict=pattern_dict,
        save_path=str(save_dir),
        save_single_file=True,
        check_correctness=bool(args.check_correctness),
    )

    generated = (save_dir / "model.safetensors").resolve()
    final_path = (save_dir / output_name).resolve()
    if generated.is_file() and generated != final_path:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            final_path.unlink()
        generated.replace(final_path)

    print(f"done: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
