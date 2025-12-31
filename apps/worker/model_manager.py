from __future__ import annotations

import gc
import os
import threading
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from capabilities.base import CapabilityWorker


class GPUMode(str, Enum):
    RESIDENT = "resident"
    ONDEMAND = "ondemand"


def _parse_gpu_mode(raw: str | None) -> GPUMode | None:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"resident", "keep", "cache"}:
        return GPUMode.RESIDENT
    if value in {"ondemand", "on-demand", "unload"}:
        return GPUMode.ONDEMAND
    return None


def _gpu_mode_from_env() -> GPUMode:
    explicit = _parse_gpu_mode(os.getenv("CAP_GPU_MODE"))
    if explicit is not None:
        return explicit
    return GPUMode.ONDEMAND


def _cuda_cleanup() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        return


class ModelManager:
    """Ensure GPU has only one capability's model loaded at a time (single-process)."""

    _lock = threading.Lock()
    _active: str | None = None
    _workers: dict[str, "CapabilityWorker"] = {}

    @classmethod
    def register(cls, worker: "CapabilityWorker") -> None:
        cls._workers[worker.name] = worker

    @classmethod
    def acquire(cls, capability: str) -> "CapabilityWorker":
        with cls._lock:
            if capability not in cls._workers:
                raise ValueError(f"Unknown capability: {capability}")

            worker = cls._workers[capability]

            if cls._active and cls._active != capability:
                old = cls._workers.get(cls._active)
                if old is not None:
                    try:
                        old.unload_models()
                    finally:
                        _cuda_cleanup()
                cls._active = None

            if not worker.is_loaded:
                worker.load_models()

            cls._active = capability
            return worker

    @classmethod
    def release(cls, capability: str) -> None:
        if _gpu_mode_from_env() != GPUMode.ONDEMAND:
            return
        with cls._lock:
            if cls._active != capability:
                return
            worker = cls._workers.get(capability)
            if worker is None:
                cls._active = None
                return
            try:
                worker.unload_models()
            finally:
                _cuda_cleanup()
                cls._active = None

    @classmethod
    def _reset_for_tests(cls) -> None:
        cls._workers = {}
        cls._active = None
