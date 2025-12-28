from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
import importlib

import torch

tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}

_SUPPORTED_ATTENTION_TYPES = {"original", "sla", "sagesla"}


def _ensure_torch_libs_on_ld_library_path() -> None:
    torch_lib = (Path(torch.__file__).resolve().parent / "lib").resolve()
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(":") if p]
    if str(torch_lib) in parts:
        return
    os.environ["LD_LIBRARY_PATH"] = f"{torch_lib}:{current}" if current else str(torch_lib)


def _require_turbo_diffusion_ops() -> None:
    """
    Ensure `turbo_diffusion_ops` is importable.

    We intentionally do not load `.so` files from `third_party/` at runtime; the extension must be
    built/installed into the worker environment (see `docs/turbodiffusion_i2v_runbook.md`).
    """
    try:
        import turbo_diffusion_ops  # noqa: F401
        return
    except ModuleNotFoundError:
        raise RuntimeError(
            "`turbo_diffusion_ops` is required for quantized Wan2.2 checkpoints, but it is not available. "
            "Fix: build and install the extension into the worker venv (see `docs/turbodiffusion_i2v_runbook.md`)."
        ) from None


def _torch_apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool = True,
    inplace: bool = False,
    **_: object,
) -> torch.Tensor:
    """
    FlashAttention-compatible fallback for `flash_attn.layers.rotary.apply_rotary_emb`.

    Upstream TurboDiffusion uses FlashAttention's rotary embedding function but does not
    guard against `flash_attn` being absent. This pure-PyTorch implementation is slower
    but keeps inference functional when `flash_attn` isn't installed.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x to have shape [B, S, H, D], got {tuple(x.shape)}")

    batch_size, seq_len, n_heads, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError(f"Expected head_dim to be even, got {head_dim}")

    if cos.ndim != 2 or sin.ndim != 2:
        raise ValueError(f"Expected cos/sin to have shape [S, D/2], got {tuple(cos.shape)} / {tuple(sin.shape)}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin shapes must match, got {tuple(cos.shape)} vs {tuple(sin.shape)}")
    if cos.shape[0] != seq_len:
        raise ValueError(f"cos/sin seq_len must match x, got {cos.shape[0]} vs {seq_len}")

    rotary_dim = int(cos.shape[1]) * 2
    if rotary_dim > head_dim:
        raise ValueError(f"rotary_dim ({rotary_dim}) exceeds head_dim ({head_dim})")

    cos_b = cos.to(device=x.device, dtype=x.dtype).reshape(1, seq_len, 1, rotary_dim // 2)
    sin_b = sin.to(device=x.device, dtype=x.dtype).reshape(1, seq_len, 1, rotary_dim // 2)

    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    if interleaved:
        x_pair = x_rot.reshape(batch_size, seq_len, n_heads, rotary_dim // 2, 2)
        x0 = x_pair[..., 0]
        x1 = x_pair[..., 1]
        out0 = x0 * cos_b - x1 * sin_b
        out1 = x0 * sin_b + x1 * cos_b
        out_rot = torch.stack((out0, out1), dim=-1).reshape(batch_size, seq_len, n_heads, rotary_dim)
    else:
        x0, x1 = x_rot.chunk(2, dim=-1)
        out_rot = torch.cat((x0 * cos_b - x1 * sin_b, x0 * sin_b + x1 * cos_b), dim=-1)

    out = torch.cat((out_rot, x_pass), dim=-1) if x_pass.numel() else out_rot
    if inplace:
        x.copy_(out)
        return x
    return out


def _ensure_flash_rope_available() -> None:
    """
    TurboDiffusion upstream's `rcm.networks.wan2pt{1,2}` uses FlashAttention's rotary embedding.

    Those modules currently set `flash_apply_rotary_emb = None` when `flash_attn` is missing,
    but still call it unconditionally, crashing with:
      TypeError: 'NoneType' object is not callable
    """
    try:
        from flash_attn.layers.rotary import apply_rotary_emb  # noqa: F401
    except Exception:
        apply_rotary_emb = None  # type: ignore[assignment]

    if apply_rotary_emb is not None:
        return

    for module_name in ("rcm.networks.wan2pt1", "rcm.networks.wan2pt2"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        if getattr(module, "flash_apply_rotary_emb", None) is None:
            setattr(module, "flash_apply_rotary_emb", _torch_apply_rotary_emb)


def _normalize_state_dict_keys(state_dict: dict[str, object]) -> dict[str, object]:
    renamed: dict[str, object] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        renamed[key.replace("._checkpoint_wrapped_module.", ".")] = value
    return renamed


def _state_dict_uses_int8_linear(state_dict: dict[str, object]) -> bool:
    for key, value in state_dict.items():
        if isinstance(key, str) and key.endswith(".int8_weight") and isinstance(value, torch.Tensor):
            return True
    return False


def select_model(model_name: str) -> torch.nn.Module:
    _ensure_flash_rope_available()
    from rcm.utils.selective_activation_checkpoint import SACConfig  # noqa: PLC0415
    from rcm.networks.wan2pt2 import WanModel as WanModel2pt2  # noqa: PLC0415

    if model_name != "Wan2.2-A14B":
        raise ValueError(f"Unsupported model name: {model_name}")

    return WanModel2pt2(
        dim=5120,
        eps=1e-06,
        ffn_dim=13824,
        freq_dim=256,
        in_dim=36,
        model_type="i2v",
        num_heads=40,
        num_layers=40,
        out_dim=16,
        text_len=512,
        sac_config=SACConfig(mode="none"),
    )


def _resolve_attention_type(attention_type: str) -> str:
    if attention_type not in _SUPPORTED_ATTENTION_TYPES:
        raise ValueError(f"Unsupported attention_type: {attention_type}")

    if attention_type != "sagesla":
        return attention_type

    try:
        import spas_sage_attn  # noqa: F401
    except Exception:
        warnings.warn(
            "Requested attention_type='sagesla' but `spas_sage_attn` is not installed; "
            "falling back to 'sla'. Install the worker `sagesla` group to enable SageSLA.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "sla"

    return "sagesla"


def _replace_attention(*, model: torch.nn.Module, attention_type: str, sla_topk: float) -> torch.nn.Module:
    attention_type = _resolve_attention_type(attention_type)
    if attention_type == "original":
        return model

    from SLA import SparseLinearAttention as SLA  # noqa: PLC0415
    SageSLA = None
    if attention_type == "sagesla":
        from SLA import SageSparseLinearAttention as SageSLA  # noqa: PLC0415

    from rcm.networks.wan2pt2 import WanSelfAttention as WanSelfAttention2pt2  # noqa: PLC0415

    for module in model.modules():
        if type(module) is WanSelfAttention2pt2:
            head_dim = module.dim // module.num_heads
            if attention_type == "sla":
                module.attn_op.local_attn = SLA(head_dim=head_dim, topk=sla_topk, BLKQ=128, BLKK=64)
                continue

            assert SageSLA is not None
            try:
                module.attn_op.local_attn = SageSLA(head_dim=head_dim, topk=sla_topk)
            except AssertionError:
                warnings.warn(
                    "Failed to initialize SageSLA; falling back to SLA. "
                    "This typically means `spas_sage_attn` is missing or failed to load.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                module.attn_op.local_attn = SLA(head_dim=head_dim, topk=sla_topk, BLKQ=128, BLKK=64)

    return model


def _cast_rmsnorm_weights(*, model: torch.nn.Module, dtype: torch.dtype) -> None:
    """
    Upstream `WanRMSNorm.forward()` multiplies by `self.weight` outside `type_as(x)`,
    so keeping `weight` in fp32 can make q/k float32 while v is bf16/fp16, tripping dtype asserts.
    """
    from rcm.networks.wan2pt2 import WanRMSNorm as WanRMSNorm2pt2  # noqa: PLC0415

    for module in model.modules():
        if isinstance(module, WanRMSNorm2pt2) and module.weight.dtype != dtype:
            module.weight.data = module.weight.data.to(dtype)


def _replace_linear_with_int8(*, model: torch.nn.Module, skip_layer: str = "proj_l") -> torch.nn.Module:
    """
    Replace `torch.nn.Linear` layers inside transformer blocks with int8 GEMM modules.

    This is required to load quantized checkpoints that store `*.int8_weight` and `*.scale` buffers.
    """
    from turbo_diffusion_ops import gemm_cuda, quant_cuda  # noqa: PLC0415

    replacements: dict[str, torch.nn.Module] = {}

    class _SafeInt8Linear(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int, *, bias: bool = True, dtype: torch.dtype = torch.bfloat16):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

            row_blocks = (out_features + 127) // 128
            col_blocks = (in_features + 127) // 128

            self.register_buffer("int8_weight", torch.empty((out_features, in_features), dtype=torch.int8))
            self.register_buffer("scale", torch.empty((row_blocks, col_blocks), dtype=torch.float32))
            if bias:
                self.register_buffer("bias", torch.empty(out_features, dtype=dtype))
            else:
                self.bias = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
            shape = x.shape
            x2 = x.reshape(-1, shape[-1])
            if x2.dtype == torch.float32:
                x2 = x2.to(torch.bfloat16)

            m = x2.shape[0]
            n = self.int8_weight.shape[0]
            y = torch.zeros(m, n, dtype=x2.dtype, device=x2.device)

            x_q, x_s = quant_cuda(x2, None, None)
            gemm_cuda(x_q, x_s, self.int8_weight, self.scale, y)
            y = y.reshape(*shape[:-1], n)
            if self.bias is not None:
                y = y + self.bias
            return y

        @classmethod
        def from_linear(cls, original_linear: torch.nn.Linear) -> "_SafeInt8Linear":
            return cls(
                original_linear.in_features,
                original_linear.out_features,
                bias=original_linear.bias is not None,
                dtype=original_linear.weight.dtype,
            )

    for name, module in model.blocks.named_modules():
        if isinstance(module, torch.nn.Linear) and skip_layer not in name:
            replacements[name] = _SafeInt8Linear.from_linear(module)

    for name, new_module in replacements.items():
        parent: torch.nn.Module = model.blocks
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    return model


def create_model(dit_path: str, args: argparse.Namespace) -> torch.nn.Module:
    """
    Create Wan2.2 I2V model without TurboDiffusion custom CUDA ops.

    This is a dev-friendly fallback to avoid building `turbo_diffusion_ops` which requires a
    CUDA toolkit version matching the PyTorch build (e.g., cu128).
    """
    _ensure_flash_rope_available()
    from rcm.utils.model_utils import load_state_dict  # noqa: PLC0415

    with torch.device("meta"):
        net = select_model(args.model)

    attention_type = getattr(args, "attention_type", "original")
    sla_topk = float(getattr(args, "sla_topk", 0.1))
    net = _replace_attention(model=net, attention_type=attention_type, sla_topk=sla_topk)

    state_dict_raw = load_state_dict(dit_path)
    state_dict = _normalize_state_dict_keys(state_dict_raw)

    if _state_dict_uses_int8_linear(state_dict):
        _ensure_torch_libs_on_ld_library_path()
        try:
            _require_turbo_diffusion_ops()
            import turbo_diffusion_ops  # noqa: F401
        except Exception:
            raise RuntimeError(
                "Quantized TurboDiffusion checkpoint detected (int8 Linear), but `turbo_diffusion_ops` "
                "is not available. This typically happens when the local CUDA toolkit `nvcc` major version "
                "doesn't match PyTorch's CUDA version (e.g., nvcc 13.x with torch +cu128). "
                "Fix: install a matching CUDA toolkit (see docs/turbodiffusion_i2v_runbook.md) and rebuild "
                "TurboDiffusion ops."
            )
        else:
            net = _replace_linear_with_int8(model=net)

    net.load_state_dict(state_dict, assign=True)
    _cast_rmsnorm_weights(model=net, dtype=tensor_kwargs["dtype"])
    net = net.to(tensor_kwargs["device"]).eval()
    del state_dict_raw, state_dict
    return net
