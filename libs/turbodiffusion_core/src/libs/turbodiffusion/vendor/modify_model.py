from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import torch

from rcm.utils.model_utils import load_state_dict
from rcm.utils.selective_activation_checkpoint import SACConfig
from rcm.networks.wan2pt2 import WanModel as WanModel2pt2
from rcm.networks.wan2pt2 import WanRMSNorm as WanRMSNorm2pt2
from rcm.networks.wan2pt2 import WanSelfAttention as WanSelfAttention2pt2


tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}

_SUPPORTED_ATTENTION_TYPES = {"original", "sla", "sagesla"}


def _ensure_torch_libs_on_ld_library_path() -> None:
    torch_lib = (Path(torch.__file__).resolve().parent / "lib").resolve()
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(":") if p]
    if str(torch_lib) in parts:
        return
    os.environ["LD_LIBRARY_PATH"] = f"{torch_lib}:{current}" if current else str(torch_lib)


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


def _dequantize_int8_linear_state_dict(
    state_dict: dict[str, object],
    *,
    dtype: torch.dtype,
    block_size: int = 128,
) -> dict[str, object]:
    """
    Convert TurboDiffusion int8-linear checkpoints into standard `nn.Linear` weights.

    This is a dev fallback when `turbo_diffusion_ops` isn't available (e.g., CUDA toolkit mismatch),
    allowing smoke tests to run without compiling custom CUDA extensions.
    """

    def dequantize_weight(int8_weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if int8_weight.dtype != torch.int8:
            raise TypeError(f"Expected int8_weight dtype=int8, got {int8_weight.dtype}")
        if scale.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"Expected scale dtype=float, got {scale.dtype}")
        if int8_weight.ndim != 2 or scale.ndim != 2:
            raise ValueError("Expected int8_weight and scale to be 2D tensors")

        out_features, in_features = int8_weight.shape
        row_blocks, col_blocks = scale.shape
        if out_features != row_blocks * block_size or in_features != col_blocks * block_size:
            raise ValueError(
                "Unexpected int8 Linear shapes: "
                f"int8_weight={tuple(int8_weight.shape)} scale={tuple(scale.shape)} block_size={block_size}"
            )

        weight = int8_weight.reshape(row_blocks, block_size, col_blocks, block_size).to(torch.float16)
        scale_f16 = scale.to(torch.float16).reshape(row_blocks, 1, col_blocks, 1)
        return (weight * scale_f16).reshape(out_features, in_features).to(dtype)

    prefixes: list[str] = []
    for key in state_dict:
        if isinstance(key, str) and key.endswith(".int8_weight"):
            prefixes.append(key[: -len(".int8_weight")])

    drop_keys: set[str] = set()
    for prefix in prefixes:
        drop_keys.add(f"{prefix}.int8_weight")
        drop_keys.add(f"{prefix}.scale")
        drop_keys.add(f"{prefix}.bias")

    new_state_dict: dict[str, object] = {
        key: value for key, value in state_dict.items() if isinstance(key, str) and key not in drop_keys
    }

    for prefix in prefixes:
        int8_weight = state_dict.get(f"{prefix}.int8_weight")
        scale = state_dict.get(f"{prefix}.scale")
        bias = state_dict.get(f"{prefix}.bias")
        if not isinstance(int8_weight, torch.Tensor) or not isinstance(scale, torch.Tensor):
            raise KeyError(f"Missing tensors for quantized linear: {prefix}")

        new_state_dict[f"{prefix}.weight"] = dequantize_weight(int8_weight, scale)
        if isinstance(bias, torch.Tensor):
            new_state_dict[f"{prefix}.bias"] = bias.to(dtype)

    return new_state_dict


def select_model(model_name: str) -> torch.nn.Module:
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
            import turbo_diffusion_ops  # noqa: F401
        except Exception:
            if os.getenv("TD_DEQUANT_INT8_FALLBACK") in {"1", "true", "TRUE", "yes", "YES"}:
                state_dict = _dequantize_int8_linear_state_dict(state_dict, dtype=tensor_kwargs["dtype"])
            else:
                raise RuntimeError(
                    "Quantized TurboDiffusion checkpoint detected (int8 Linear), but `turbo_diffusion_ops` "
                    "is not available. This typically happens when the local CUDA toolkit `nvcc` major version "
                    "doesn't match PyTorch's CUDA version (e.g., nvcc 13.x with torch +cu128). "
                    "Fix options: (1) install CUDA 12.8 toolkit and rebuild TurboDiffusion ops; "
                    "(2) use non-quantized checkpoints on a >40GB GPU. "
                    "If you still want a dev fallback that dequantizes to bf16 (may OOM on 32GB GPUs), "
                    "set TD_DEQUANT_INT8_FALLBACK=1."
                )
        else:
            net = _replace_linear_with_int8(model=net)

    net.load_state_dict(state_dict, assign=True)
    _cast_rmsnorm_weights(model=net, dtype=tensor_kwargs["dtype"])
    net = net.to(tensor_kwargs["device"]).eval()
    del state_dict_raw, state_dict
    return net
