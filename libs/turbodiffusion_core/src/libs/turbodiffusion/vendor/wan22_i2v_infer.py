from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as T
from einops import rearrange, repeat
from PIL import Image
from tqdm import tqdm

from libs.pycore.paths import data_dir
from libs.turbodiffusion.paths import turbodiffusion_repo_dir


def _ensure_upstream_on_syspath() -> None:
    repo_dir = turbodiffusion_repo_dir()
    upstream_pkg = (repo_dir / "turbodiffusion").resolve()
    if str(upstream_pkg) not in sys.path:
        sys.path.insert(0, str(upstream_pkg))


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan2.2 I2V inference (vendored wrapper)")
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
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--resolution", default="720p", type=str)
    parser.add_argument("--aspect_ratio", default="16:9", type=str)
    parser.add_argument("--adaptive_resolution", action="store_true")
    parser.add_argument("--ode", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="output/generated_video.mp4")
    parser.add_argument("--attention_type", choices=["original", "sla", "sagesla"], default="original")
    parser.add_argument("--sla_topk", type=float, default=0.1)
    parser.add_argument("--quant_linear", action="store_true")
    parser.add_argument("--default_norm", action="store_true")
    return parser.parse_args(argv)


def generate_wan22_i2v(
    *,
    image_path: Path,
    high_noise_model_path: Path,
    low_noise_model_path: Path,
    vae_path: Path,
    text_encoder_path: Path,
    prompt: str,
    save_path: Path,
    boundary: float = 0.9,
    num_steps: int = 4,
    sigma_max: float = 200.0,
    seed: int = 0,
    attention_type: str = "original",
    sla_topk: float = 0.1,
    resolution: str = "720p",
    aspect_ratio: str = "16:9",
    adaptive_resolution: bool = True,
    ode: bool = True,
) -> Path:
    _ensure_upstream_on_syspath()

    from imaginaire.utils.io import save_image_or_video  # noqa: PLC0415
    from imaginaire.utils import log  # noqa: PLC0415
    from rcm.datasets.utils import VIDEO_RES_SIZE_INFO  # noqa: PLC0415
    from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding  # noqa: PLC0415
    from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface  # noqa: PLC0415

    from .modify_model import tensor_kwargs, create_model  # noqa: PLC0415

    if os.getenv("HF_HOME") is None:
        os.environ["HF_HOME"] = str((data_dir() / "hf").resolve())

    log.info(f"Computing embedding for prompt: {prompt}")
    umt5_device = os.getenv("TD_UMT5_DEVICE", "cpu")
    text_emb = get_umt5_embedding(checkpoint_path=str(text_encoder_path), prompts=prompt, device=umt5_device).to(
        **tensor_kwargs
    )
    clear_umt5_memory()

    args = argparse.Namespace(
        model="Wan2.2-A14B",
        attention_type=attention_type,
        sla_topk=sla_topk,
        quant_linear=False,
        default_norm=True,
    )

    log.info("Loading DiT models (dev fallback: no custom CUDA ops).")
    high_noise_model = create_model(dit_path=str(high_noise_model_path), args=args).cpu()
    torch.cuda.empty_cache()
    low_noise_model = create_model(dit_path=str(low_noise_model_path), args=args).cpu()
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

    F = 77
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
    image_tensor = image_transforms(input_image).unsqueeze(0).to(device=tensor_kwargs["device"], dtype=torch.float32)

    with torch.no_grad():
        frames_to_encode = torch.cat(
            [image_tensor.unsqueeze(2), torch.zeros(1, 3, F - 1, h, w, device=image_tensor.device)],
            dim=2,
        )
        encoded_latents = tokenizer.encode(frames_to_encode)

    msk = torch.zeros(1, 4, lat_t, lat_h, lat_w, device=tensor_kwargs["device"], dtype=tensor_kwargs["dtype"])
    msk[:, :, 0, :, :] = 1.0

    y = torch.cat([msk, encoded_latents.to(**tensor_kwargs)], dim=1).repeat(1, 1, 1, 1, 1)
    condition = {
        "crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=1),
        "y_B_C_T_H_W": y,
    }

    state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]

    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(seed)

    init_noise = torch.randn(1, *state_shape, dtype=torch.float32, device=tensor_kwargs["device"], generator=generator)

    mid_t = [1.5, 1.4, 1.0][: num_steps - 1]
    t_steps = torch.tensor([math.atan(sigma_max), *mid_t, 0], dtype=torch.float64, device=init_noise.device)
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)

    total_steps = t_steps.shape[0] - 1
    high_noise_model.cuda()
    net = high_noise_model
    switched = False
    for t_cur, t_next in tqdm(list(zip(t_steps[:-1], t_steps[1:])), desc="Sampling", total=total_steps):
        if t_cur.item() < boundary and not switched:
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
                    *x.shape, dtype=torch.float32, device=tensor_kwargs["device"], generator=generator
                )

    samples = x.float()
    low_noise_model.cpu()
    torch.cuda.empty_cache()

    video = tokenizer.decode(samples)
    to_show = (1.0 + video.float().cpu().clamp(-1, 1)) / 2.0

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_image_or_video(to_show[0], str(save_path), fps=16)

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
