from __future__ import annotations

import argparse
import math
import os
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as T
from einops import repeat
from PIL import Image
from tqdm import tqdm

from ..hf import configure_hf_home, configure_hf_offline
from ..paths import wan22_i2v_text_encoder_df11_path


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wan2.2 I2V inference (vendored wrapper)"
    )
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--high_noise_model_path", type=str, required=True)
    parser.add_argument("--low_noise_model_path", type=str, required=True)
    parser.add_argument("--boundary", type=float, default=0.9)
    parser.add_argument("--model", choices=["Wan2.2-A14B"], default="Wan2.2-A14B")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument("--sigma_max", type=float, default=200)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--text_encoder_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=77)
    parser.add_argument("--fps", type=float, default=16)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--resolution", default="720p", type=str)
    parser.add_argument("--aspect_ratio", default="16:9", type=str)
    parser.add_argument("--adaptive_resolution", action="store_true")
    parser.add_argument("--ode", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="output/generated_video.mp4")
    parser.add_argument(
        "--attention_type", choices=["original", "sla", "sagesla"], default="sagesla"
    )
    parser.add_argument("--sla_topk", type=float, default=0.1)
    parser.add_argument("--quant_linear", action="store_true")
    parser.add_argument("--default_norm", action="store_true")
    return parser.parse_args(argv)


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _gpu_mode_defaults() -> tuple[bool, bool]:
    """
    Return (resident_gpu_default, cuda_cleanup_default) derived from GPU_MODE.

    GPU_MODE is a higher-level switch typically expanded by `scripts/tsn_manage.sh`. When users
    only export GPU_MODE (e.g. via IDE envFile) but not the derived TD_* vars, we still want
    predictable behavior.
    """
    mode = str(os.getenv("GPU_MODE", "")).strip().lower()
    if mode == "fast":
        return True, False
    if mode == "balanced":
        return False, True
    if mode == "lowvram":
        return False, True
    return False, True


_RESIDENT_LOCK = threading.Lock()
_TEXT_ENCODER_LOCK = threading.Lock()
_RESIDENT_DIT: (
    tuple[Path, Path, str, float, torch.nn.Module, torch.nn.Module] | None
) = None
_RESIDENT_VAE: tuple[Path, object] | None = None
_RESIDENT_DF11_TEXT: tuple[Path, str, int, object] | None = None


def _get_resident_dit_models(
    *,
    high_noise_model_path: Path,
    low_noise_model_path: Path,
    attention_type: str,
    sla_topk: float,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    global _RESIDENT_DIT
    with _RESIDENT_LOCK:
        if _RESIDENT_DIT is not None:
            hi_path, lo_path, cached_attn, cached_topk, hi_model, lo_model = (
                _RESIDENT_DIT
            )
            if (
                hi_path == high_noise_model_path
                and lo_path == low_noise_model_path
                and cached_attn == attention_type
                and float(cached_topk) == float(sla_topk)
            ):
                return hi_model, lo_model

        from .modify_model import create_model  # noqa: PLC0415

        args = argparse.Namespace(
            model="Wan2.2-A14B",
            attention_type=attention_type,
            sla_topk=sla_topk,
            quant_linear=False,
            default_norm=True,
        )

        hi_model = create_model(dit_path=str(high_noise_model_path), args=args).eval()
        lo_model = create_model(dit_path=str(low_noise_model_path), args=args).eval()
        _RESIDENT_DIT = (
            high_noise_model_path,
            low_noise_model_path,
            attention_type,
            float(sla_topk),
            hi_model,
            lo_model,
        )
        return hi_model, lo_model


def _get_resident_vae_tokenizer(*, vae_path: Path) -> object:
    global _RESIDENT_VAE
    with _RESIDENT_LOCK:
        if _RESIDENT_VAE is not None:
            cached_path, cached = _RESIDENT_VAE
            if cached_path == vae_path:
                return cached

        from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface  # noqa: PLC0415

        tokenizer = Wan2pt1VAEInterface(vae_pth=str(vae_path))
        _RESIDENT_VAE = (vae_path, tokenizer)
        return tokenizer


def _get_text_embedding(
    *,
    checkpoint_path: Path,
    prompt: str,
    device: str,
    max_length: int,
    resident: bool,
) -> torch.Tensor:
    from imaginaire.utils import log  # noqa: PLC0415

    text_encoder_format = (
        str(os.getenv("TD_TEXT_ENCODER_FORMAT", "df11")).strip().lower()
    )
    if text_encoder_format == "df11":
        df11_path = wan22_i2v_text_encoder_df11_path()
        if not df11_path.is_file():
            print(
                f"[warn] DF11 text encoder file missing; falling back to bf16: {df11_path}",
                file=sys.stderr,
            )
            text_encoder_format = "bf16"
        else:
            log.info(
                f"Using DF11 text encoder (umt5-xxl, device={device}, max_length={max_length})"
            )

    if text_encoder_format == "df11":
        try:
            from ..df11_umt5_encoder import UMT5DF11Encoder  # noqa: PLC0415
        except Exception as exc:
            print(
                f"[warn] DF11 text encoder requested but helper import failed; falling back to bf16: {exc}",
                file=sys.stderr,
            )
        else:
            global _RESIDENT_DF11_TEXT
            with _TEXT_ENCODER_LOCK:
                if _RESIDENT_DF11_TEXT is not None:
                    cached_path, cached_device, cached_len, encoder = (
                        _RESIDENT_DF11_TEXT
                    )
                    if (
                        cached_path == df11_path
                        and cached_device == device
                        and int(cached_len) == int(max_length)
                    ):
                        log.info("Reusing resident DF11 text encoder")
                        return encoder(prompt)  # type: ignore[call-arg]

                try:
                    log.info(f"Loading DF11 text encoder weights: {df11_path}")
                    encoder = UMT5DF11Encoder(
                        df11_dir=df11_path.parent,
                        df11_safetensors_path=df11_path,
                        device=device,
                        max_length=max_length,
                    )
                except Exception as exc:
                    print(
                        f"[warn] DF11 text encoder init failed; falling back to bf16: {exc}",
                        file=sys.stderr,
                    )
                else:
                    if resident:
                        _RESIDENT_DF11_TEXT = (
                            df11_path,
                            device,
                            int(max_length),
                            encoder,
                        )
                    return encoder(prompt)

    from ..umt5_bf16_encoder import clear_umt5_memory, get_umt5_embedding  # noqa: PLC0415

    emb = get_umt5_embedding(
        checkpoint_path=str(checkpoint_path),
        prompts=prompt,
        device=device,
        max_length=max_length,
    )
    if not resident:
        clear_umt5_memory()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return emb


def generate_wan22_i2v(
    *,
    image_path: Path,
    high_noise_model_path: Path,
    low_noise_model_path: Path,
    vae_path: Path,
    text_encoder_path: Path,
    prompt: str,
    save_path: Path,
    num_frames: int = 77,
    fps: float = 16,
    boundary: float = 0.9,
    num_steps: int = 4,
    sigma_max: float = 200.0,
    seed: int = 0,
    attention_type: str = "sagesla",
    sla_topk: float = 0.1,
    resolution: str = "720p",
    aspect_ratio: str = "16:9",
    adaptive_resolution: bool = True,
    ode: bool = True,
) -> Path:
    from imaginaire.utils.io import save_image_or_video  # noqa: PLC0415
    from imaginaire.utils import log  # noqa: PLC0415
    from rcm.datasets.utils import VIDEO_RES_SIZE_INFO  # noqa: PLC0415
    from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface  # noqa: PLC0415

    from .modify_model import tensor_kwargs, create_model  # noqa: PLC0415

    configure_hf_home()
    configure_hf_offline()

    resident_default, _ = _gpu_mode_defaults()
    resident_gpu = _env_bool("TD_RESIDENT_GPU", resident_default)

    log.info(f"Computing embedding for prompt: {prompt}")
    umt5_device = os.getenv("TD_UMT5_DEVICE", "cuda")
    text_emb = _get_text_embedding(
        checkpoint_path=text_encoder_path,
        prompt=prompt,
        device=str(umt5_device),
        max_length=512,
        resident=resident_gpu,
    ).to(**tensor_kwargs)

    if resident_gpu:
        log.info("Loading DiT models (resident GPU mode).")
        high_noise_model, low_noise_model = _get_resident_dit_models(
            high_noise_model_path=high_noise_model_path,
            low_noise_model_path=low_noise_model_path,
            attention_type=attention_type,
            sla_topk=sla_topk,
        )
        log.success("Successfully loaded DiT models (resident GPU mode).")
        tokenizer = _get_resident_vae_tokenizer(vae_path=vae_path)
    else:
        args = argparse.Namespace(
            model="Wan2.2-A14B",
            attention_type=attention_type,
            sla_topk=sla_topk,
            quant_linear=False,
            default_norm=True,
        )

        log.info("Loading DiT models (dev fallback: no custom CUDA ops).")
        high_noise_model = create_model(
            dit_path=str(high_noise_model_path), args=args
        ).cpu()
        torch.cuda.empty_cache()
        low_noise_model = create_model(
            dit_path=str(low_noise_model_path), args=args
        ).cpu()
        torch.cuda.empty_cache()
        log.success("Successfully loaded DiT models.")

        tokenizer = Wan2pt1VAEInterface(vae_pth=str(vae_path))

    log.info(f"Loading and preprocessing image from: {image_path}")
    input_image = Image.open(image_path).convert("RGB")

    if adaptive_resolution:
        base_w, base_h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
        max_resolution_area = base_w * base_h

        orig_w, orig_h = input_image.size
        image_aspect_ratio = orig_h / orig_w

        ideal_w = np.sqrt(max_resolution_area / image_aspect_ratio)
        ideal_h = np.sqrt(max_resolution_area * image_aspect_ratio)

        stride = tokenizer.spatial_compression_factor * 2
        lat_h = round(ideal_h / stride)
        lat_w = round(ideal_w / stride)
        h = lat_h * stride
        w = lat_w * stride
    else:
        w, h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]

    requested_f = int(num_frames)
    if requested_f <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}")
    if requested_f == 2:
        raise ValueError(
            "num_frames=2 is unsupported due to VAE temporal kernel; use 1 or >=3"
        )
    # The upstream VAE encoder chunks frames with a fixed window and relies on cache.
    # Certain `num_frames` values can lead to Conv3D kernel>input errors in downstream
    # temporal downsample blocks; padding to `1 + 4k` keeps chunk boundaries stable.
    if requested_f > 1 and (requested_f - 1) % 4 != 0:
        padded_f = 1 + 4 * ((requested_f - 1 + 3) // 4)
        log.warning(
            f"num_frames={requested_f} is not aligned to 1+4k; padding to {padded_f} frames for VAE stability"
        )
        F = int(padded_f)
    else:
        F = int(requested_f)
    lat_h = h // tokenizer.spatial_compression_factor
    lat_w = w // tokenizer.spatial_compression_factor
    lat_t = tokenizer.get_latent_num_frames(F)

    image_transforms = T.Compose(
        [
            T.ToImage(),
            T.Resize(size=(h, w), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image_tensor = (
        image_transforms(input_image)
        .unsqueeze(0)
        .to(device=tensor_kwargs["device"], dtype=torch.float32)
    )

    with torch.no_grad():
        frames_to_encode = torch.cat(
            [
                image_tensor.unsqueeze(2),
                torch.zeros(1, 3, F - 1, h, w, device=image_tensor.device),
            ],
            dim=2,
        )
        encoded_latents = tokenizer.encode(frames_to_encode)

    msk = torch.zeros(
        1,
        4,
        lat_t,
        lat_h,
        lat_w,
        device=tensor_kwargs["device"],
        dtype=tensor_kwargs["dtype"],
    )
    msk[:, :, 0, :, :] = 1.0

    y = torch.cat([msk, encoded_latents.to(**tensor_kwargs)], dim=1).repeat(
        1, 1, 1, 1, 1
    )
    condition = {
        "crossattn_emb": repeat(
            text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=1
        ),
        "y_B_C_T_H_W": y,
    }

    state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]

    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(seed)

    init_noise = torch.randn(
        1,
        *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )

    mid_t = [1.5, 1.4, 1.0][: num_steps - 1]
    t_steps = torch.tensor(
        [math.atan(sigma_max), *mid_t, 0], dtype=torch.float64, device=init_noise.device
    )
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)

    total_steps = t_steps.shape[0] - 1
    if not resident_gpu:
        high_noise_model.cuda()
    net = high_noise_model
    switched = False
    for t_cur, t_next in tqdm(
        list(zip(t_steps[:-1], t_steps[1:])), desc="Sampling", total=total_steps
    ):
        if t_cur.item() < boundary and not switched:
            if not resident_gpu:
                high_noise_model.cpu()
                torch.cuda.empty_cache()
                low_noise_model.cuda()
            net = low_noise_model
            switched = True
        with torch.no_grad():
            v_pred = net(
                x_B_C_T_H_W=x.to(**tensor_kwargs),
                timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                **condition,
            ).to(torch.float64)
            if ode:
                x = x - (t_cur - t_next) * v_pred
            else:
                x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                    *x.shape,
                    dtype=torch.float32,
                    device=tensor_kwargs["device"],
                    generator=generator,
                )

    samples = x.float()
    if not resident_gpu:
        low_noise_model.cpu()
        torch.cuda.empty_cache()

    video = tokenizer.decode(samples)
    if requested_f != F:
        video = video[:, :, : min(requested_f, int(video.shape[2])), :, :]
    to_show = (1.0 + video.float().cpu().clamp(-1, 1)) / 2.0

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_image_or_video(to_show[0], str(save_path), fps=float(fps))

    return save_path


def main() -> int:
    args = parse_arguments()
    out = generate_wan22_i2v(
        image_path=Path(args.image_path),
        high_noise_model_path=Path(args.high_noise_model_path),
        low_noise_model_path=Path(args.low_noise_model_path),
        vae_path=Path(args.vae_path),
        text_encoder_path=Path(args.text_encoder_path),
        prompt=args.prompt,
        save_path=Path(args.save_path),
        num_frames=int(args.num_frames),
        fps=float(args.fps),
        boundary=float(args.boundary),
        num_steps=int(args.num_steps),
        sigma_max=float(args.sigma_max),
        seed=int(args.seed),
        resolution=str(args.resolution),
        aspect_ratio=str(args.aspect_ratio),
        adaptive_resolution=bool(args.adaptive_resolution),
        ode=bool(args.ode),
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
