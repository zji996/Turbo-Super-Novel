from __future__ import annotations

from capabilities.config import load_capability_config


def test_load_config_defaults_to_local() -> None:
    cfg = load_capability_config({})
    assert cfg.tts.provider == "local"
    assert cfg.tts.remote_url is None
    assert cfg.llm.provider == "local"
    assert cfg.llm.remote_url is None


def test_load_config_remote_values() -> None:
    cfg = load_capability_config(
        {
            "CAP_TTS_PROVIDER": "remote",
            "CAP_TTS_REMOTE_URL": "http://127.0.0.1:8000",
            "CAP_TTS_REMOTE_API_KEY": "secret",
            "CAP_LLM_PROVIDER": "remote",
            "CAP_LLM_REMOTE_URL": "https://example.com/v1",
            "CAP_LLM_REMOTE_API_KEY": "llm-secret",
        }
    )
    assert cfg.tts.provider == "remote"
    assert cfg.tts.remote_url == "http://127.0.0.1:8000"
    assert cfg.tts.remote_api_key == "secret"
    assert cfg.llm.provider == "remote"
    assert cfg.llm.remote_url == "https://example.com/v1"
    assert cfg.llm.remote_api_key == "llm-secret"
