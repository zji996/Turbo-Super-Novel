from __future__ import annotations

import pytest

from capabilities.base import CapabilityWorker
from model_manager import ModelManager


class _DummyWorker(CapabilityWorker):
    def __init__(self, name: str) -> None:
        self.name = name
        self.requires_gpu = True
        self._loaded = False
        self.loaded_calls = 0
        self.unloaded_calls = 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_models(self) -> None:
        self.loaded_calls += 1
        self._loaded = True

    def unload_models(self) -> None:
        self.unloaded_calls += 1
        self._loaded = False

    def execute(self, job: dict) -> dict:  # noqa: ARG002
        return {"ok": True}


def test_model_manager_switching_resident(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAP_GPU_MODE", "resident")
    snapshot = (ModelManager._workers.copy(), ModelManager._active)
    try:
        ModelManager._reset_for_tests()

        a = _DummyWorker("a")
        b = _DummyWorker("b")
        ModelManager.register(a)
        ModelManager.register(b)

        ModelManager.acquire("a")
        ModelManager.release("a")
        assert a.is_loaded is True
        assert a.loaded_calls == 1
        assert a.unloaded_calls == 0

        ModelManager.acquire("b")
        assert a.is_loaded is False
        assert a.unloaded_calls == 1
        assert b.is_loaded is True
    finally:
        ModelManager._workers, ModelManager._active = snapshot


def test_model_manager_release_ondemand(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAP_GPU_MODE", "ondemand")
    snapshot = (ModelManager._workers.copy(), ModelManager._active)
    try:
        ModelManager._reset_for_tests()

        a = _DummyWorker("a")
        ModelManager.register(a)

        ModelManager.acquire("a")
        assert a.is_loaded is True
        ModelManager.release("a")
        assert a.is_loaded is False
        assert a.loaded_calls == 1
        assert a.unloaded_calls == 1
    finally:
        ModelManager._workers, ModelManager._active = snapshot


def test_model_manager_unknown_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAP_GPU_MODE", "ondemand")
    snapshot = (ModelManager._workers.copy(), ModelManager._active)
    try:
        ModelManager._reset_for_tests()

        with pytest.raises(ValueError, match="Unknown capability"):
            ModelManager.acquire("missing")
    finally:
        ModelManager._workers, ModelManager._active = snapshot
