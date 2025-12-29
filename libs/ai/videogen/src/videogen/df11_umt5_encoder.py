from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import torch

from .hf import configure_hf_offline, require_local_tokenizer, umt5_tokenizer_id


def _resolve_module(root: torch.nn.Module, dotted: str) -> torch.nn.Module:
    module: torch.nn.Module = root
    if not dotted:
        return module
    for part in dotted.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module


def _resolve_parent_and_attr(
    root: torch.nn.Module, dotted: str
) -> tuple[torch.nn.Module, str]:
    if "." not in dotted:
        return root, dotted
    parent_path, attr = dotted.rsplit(".", 1)
    return _resolve_module(root, parent_path), attr


def _set_param(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    if name in module._parameters and module._parameters[name] is not None:
        module._parameters[name] = torch.nn.Parameter(value, requires_grad=False)
        return
    raise KeyError(f"parameter not found: {module.__class__.__name__}.{name}")


def _set_buffer(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    if name in module._buffers:
        module._buffers[name] = value
        return
    module.register_buffer(name, value)


@dataclass(frozen=True, slots=True)
class DF11Config:
    threads_per_block: tuple[int, ...]
    bytes_per_thread: int
    pattern_dict: dict[str, list[str]]

    @classmethod
    def from_dir(cls, df11_dir: Path) -> "DF11Config":
        config_path = (df11_dir / "config.json").resolve()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        cfg = data["dfloat11_config"]
        return cls(
            threads_per_block=tuple(int(x) for x in cfg["threads_per_block"]),
            bytes_per_thread=int(cfg["bytes_per_thread"]),
            pattern_dict={
                str(k): [str(x) for x in v]
                for k, v in dict(cfg["pattern_dict"]).items()
            },
        )


class UMT5DF11Encoder:
    def __init__(
        self,
        *,
        df11_dir: Path,
        df11_safetensors_path: Path,
        device: str = "cuda",
        max_length: int = 512,
    ) -> None:
        self.df11_dir = df11_dir.resolve()
        self.df11_safetensors_path = df11_safetensors_path.resolve()
        self.device = device
        self.max_length = int(max_length)

        self._cfg = DF11Config.from_dir(self.df11_dir)

        from rcm.utils.umt5 import HuggingfaceTokenizer, umt5_xxl  # noqa: PLC0415

        configure_hf_offline()
        tokenizer_name = umt5_tokenizer_id()
        require_local_tokenizer(tokenizer_name)
        self._tokenizer = HuggingfaceTokenizer(
            name=tokenizer_name,
            seq_len=self.max_length,
            clean="whitespace",
        )

        with torch.device("meta"):
            self._model: torch.nn.Module = umt5_xxl(
                encoder_only=True, dtype=torch.bfloat16, device="meta"
            )

        self._load_df11_weights()
        self._model = self._model.to(device=self.device).eval()

    def _load_df11_weights(self) -> None:
        from safetensors.torch import load_file  # noqa: PLC0415

        try:
            from dfloat11.dfloat11 import get_hook  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "DF11 text encoder requested, but `dfloat11` could not be imported. "
                "Fix: `uv sync --project apps/worker` and restart the worker."
            ) from exc

        tensors = load_file(str(self.df11_safetensors_path))

        threads_per_block = self._cfg.threads_per_block
        bytes_per_thread = self._cfg.bytes_per_thread
        pattern_dict = self._cfg.pattern_dict

        for full_key, tensor in tensors.items():
            parent, attr = _resolve_parent_and_attr(self._model, full_key)

            if attr == "split_positions":
                setattr(parent, "split_positions", tensor.tolist())
                continue

            if attr in parent._parameters and parent._parameters[attr] is not None:
                _set_param(parent, attr, tensor)
                continue

            _set_buffer(parent, attr, tensor)

            if attr == "encoded_exponent":
                parent.register_forward_pre_hook(
                    get_hook(threads_per_block, bytes_per_thread)
                )

                full_name = full_key.rsplit(".", 1)[0]
                for pattern, attr_names in pattern_dict.items():
                    if re.fullmatch(pattern, full_name) is None:
                        continue

                    if isinstance(parent, torch.nn.Embedding):
                        # Some upstream modules read `embedding.weight.device` before calling the embedding,
                        # so we keep a tiny placeholder `weight` (as a buffer, not a Parameter) for relative position embeddings.
                        keep_weight = "pos_embedding.embedding" in full_name
                        if keep_weight:
                            if hasattr(parent, "weight"):
                                tmp = parent.weight
                                delattr(parent, "weight")
                                del tmp
                            _set_buffer(
                                parent,
                                "weight",
                                torch.empty(
                                    (parent.num_embeddings, parent.embedding_dim),
                                    dtype=torch.bfloat16,
                                    device="cpu",
                                ),
                            )
                        elif hasattr(parent, "weight"):
                            tmp = parent.weight
                            delattr(parent, "weight")
                            del tmp
                        break

                    if isinstance(parent, torch.nn.Linear):
                        if hasattr(parent, "weight"):
                            tmp = parent.weight
                            delattr(parent, "weight")
                            del tmp
                        break

                    setattr(parent, "weight_injection_modules", [])
                    for inject_path in attr_names:
                        target = _resolve_module(parent, inject_path)
                        if hasattr(target, "weight"):
                            tmp = target.weight
                            delattr(target, "weight")
                            del tmp
                        parent.weight_injection_modules.append(target)  # type: ignore[attr-defined]
                continue

            if attr == "output_positions":
                output_positions_np = tensor.view(torch.uint32).cpu().numpy()
                shared_mem_size = (
                    threads_per_block[0] * 4
                    + 4
                    + (output_positions_np[1:] - output_positions_np[:-1]).max().item()
                    * 2
                )
                setattr(parent, "shared_mem_size", int(shared_mem_size))

    def __call__(self, prompts: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]

        ids, mask = self._tokenizer(prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)

        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self._model(ids, mask)  # type: ignore[call-arg]

        stack_emb = []
        for emb, length in zip(context, seq_lens):
            if length > self.max_length:
                stack_emb.append(emb[: self.max_length])
            else:
                zeros = torch.zeros(
                    self.max_length - length,
                    emb.shape[1],
                    device=emb.device,
                    dtype=emb.dtype,
                )
                stack_emb.append(torch.cat([emb[:length], zeros], dim=0))

        return torch.stack(stack_emb)
