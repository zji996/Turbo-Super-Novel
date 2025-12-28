from __future__ import annotations

import gc
import threading
from typing import List, Optional, Union

import torch

from libs.turbodiffusion.hf import configure_hf_offline, require_local_tokenizer, umt5_tokenizer_id


_BF16_LOCK = threading.Lock()
_BF16_ENCODER: tuple[str, str, int, object] | None = None


def get_umt5_embedding(
    *,
    checkpoint_path: str,
    prompts: Union[str, List[str]],
    device: str = "cuda",
    max_length: int = 512,
    tokenizer_path: Optional[str] = None,
) -> torch.Tensor:
    """
    BF16 UMT5 embedding with configurable tokenizer location.

    The upstream helper uses a fixed `google/umt5-xxl` tokenizer id, which may trigger
    network access; this wrapper lets the caller provide a local tokenizer directory.
    """
    from rcm.utils.umt5 import UMT5EncoderModel  # noqa: PLC0415

    configure_hf_offline()
    resolved_tokenizer = tokenizer_path or umt5_tokenizer_id()
    require_local_tokenizer(resolved_tokenizer)

    global _BF16_ENCODER
    with _BF16_LOCK:
        if _BF16_ENCODER is not None:
            cached_ckpt, cached_tok, cached_len, encoder = _BF16_ENCODER
            if cached_ckpt == checkpoint_path and cached_tok == resolved_tokenizer and int(cached_len) == int(max_length):
                return encoder(prompts, device=device)  # type: ignore[call-arg]

        encoder = UMT5EncoderModel(
            text_len=int(max_length),
            device=device,
            checkpoint_path=str(checkpoint_path),
            tokenizer_path=str(resolved_tokenizer),
        )
        _BF16_ENCODER = (str(checkpoint_path), str(resolved_tokenizer), int(max_length), encoder)
        return encoder(prompts, device=device)  # type: ignore[call-arg]


def clear_umt5_memory() -> None:
    global _BF16_ENCODER
    with _BF16_LOCK:
        _BF16_ENCODER = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
