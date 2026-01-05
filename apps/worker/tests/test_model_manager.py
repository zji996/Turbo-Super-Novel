import pytest

from model_manager import GPUMode, ModelManager, _parse_gpu_mode


class MockWorker:
    def __init__(self, name: str):
        self.name = name
        self.is_loaded = False
        self.load_count = 0
        self.unload_count = 0

    def load_models(self):
        self.is_loaded = True
        self.load_count += 1

    def unload_models(self):
        self.is_loaded = False
        self.unload_count += 1


@pytest.fixture(autouse=True)
def reset_manager():
    ModelManager._reset_for_tests()
    yield
    ModelManager._reset_for_tests()


def test_parse_gpu_mode():
    assert _parse_gpu_mode("resident") == GPUMode.RESIDENT
    assert _parse_gpu_mode("ondemand") == GPUMode.ONDEMAND
    assert _parse_gpu_mode("keep") == GPUMode.RESIDENT
    assert _parse_gpu_mode("invalid") is None
    assert _parse_gpu_mode(None) is None


def test_acquire_loads_model():
    worker = MockWorker("tts")
    ModelManager.register(worker)
    result = ModelManager.acquire("tts")
    assert result == worker
    assert worker.is_loaded
    assert worker.load_count == 1


def test_acquire_unloads_previous():
    tts = MockWorker("tts")
    videogen = MockWorker("videogen")
    ModelManager.register(tts)
    ModelManager.register(videogen)

    ModelManager.acquire("tts")
    assert tts.is_loaded

    ModelManager.acquire("videogen")
    assert not tts.is_loaded  # Should be unloaded
    assert videogen.is_loaded
    assert tts.unload_count == 1


def test_acquire_unknown_capability():
    with pytest.raises(ValueError, match="Unknown capability"):
        ModelManager.acquire("unknown")
