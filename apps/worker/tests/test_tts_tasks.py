from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest


class _DummyS3:
    def download_file(self, bucket: str, key: str, filename: str) -> None:  # noqa: ARG002
        raise AssertionError("download_file should not be called in this test")

    def upload_file(self, filename: str, bucket: str, key: str) -> None:  # noqa: ARG002
        return


class _DummyProvider:
    async def synthesize(self, text: str, output_path, **kwargs: Any):  # noqa: ANN001, ARG002
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"dummy")
        return output_path


def test_synthesize_tts_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # noqa: ANN001
    from capabilities.tts import synthesize_tts

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr("capabilities.tts._s3_client", lambda: _DummyS3())
    monkeypatch.setattr("capabilities.tts._db_update_tts_job", lambda **_: None)
    monkeypatch.setattr("capabilities.tts._audio_duration_seconds", lambda _path: None)

    tts_mod = ModuleType("tts")
    tts_mod.TTSError = RuntimeError
    tts_mod.get_tts_provider = lambda *args, **kwargs: _DummyProvider()  # noqa: ARG005
    monkeypatch.setitem(sys.modules, "tts", tts_mod)

    assert synthesize_tts.name == "cap.tts.synthesize"
    out = synthesize_tts(
        job_id="00000000-0000-0000-0000-000000000000",
        text="hello",
        provider="edge",
        output_bucket="b",
        output_key="k.mp3",
    )
    assert out["output_bucket"] == "b"
    assert out["output_key"] == "k.mp3"
