from __future__ import annotations

import pytest

from capabilities.config import CapabilityConfig, CapabilityEndpoint
from capabilities.providers.llm import LLMProvider
from capabilities.providers.local import LocalProvider
from capabilities.providers.remote import RemoteProvider
from capabilities.router import CapabilityRouter


def test_router_returns_local_provider() -> None:
    router = CapabilityRouter(config=CapabilityConfig(tts=CapabilityEndpoint(provider="local")))
    provider = router.provider_for("tts")
    assert isinstance(provider, LocalProvider)


def test_router_returns_remote_provider() -> None:
    router = CapabilityRouter(
        config=CapabilityConfig(
            tts=CapabilityEndpoint(provider="remote", remote_url="http://example.com")
        )
    )
    provider = router.provider_for("tts")
    assert isinstance(provider, RemoteProvider)
    assert provider.base_url == "http://example.com"


def test_router_remote_requires_url() -> None:
    router = CapabilityRouter(config=CapabilityConfig(tts=CapabilityEndpoint(provider="remote")))
    with pytest.raises(ValueError):
        router.provider_for("tts")


def test_router_llm_requires_remote() -> None:
    router = CapabilityRouter(config=CapabilityConfig(llm=CapabilityEndpoint(provider="local")))
    with pytest.raises(ValueError):
        router.provider_for("llm")


def test_router_llm_returns_llm_provider() -> None:
    router = CapabilityRouter(
        config=CapabilityConfig(llm=CapabilityEndpoint(provider="remote", remote_url="https://example.com/v1"))
    )
    provider = router.provider_for("llm")
    assert isinstance(provider, LLMProvider)
    assert provider.base_url == "https://example.com/v1"
