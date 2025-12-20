from __future__ import annotations

import argparse
import warnings

import torch

from rcm.utils.model_utils import load_state_dict
from rcm.networks.wan2pt2 import WanModel as WanModel2pt2
from rcm.networks.wan2pt2 import WanSelfAttention as WanSelfAttention2pt2


tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}

_SUPPORTED_ATTENTION_TYPES = {"original", "sla", "sagesla"}


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

    state_dict = load_state_dict(dit_path)
    net.load_state_dict(state_dict, assign=True)
    net = net.to(tensor_kwargs["device"]).eval()
    del state_dict
    return net
