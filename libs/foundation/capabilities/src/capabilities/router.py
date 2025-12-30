from __future__ import annotations

from typing import Literal

from capabilities.config import CapabilityConfig, load_capability_config
from capabilities.providers.base import BaseProvider
from capabilities.providers.llm import LLMProvider
from capabilities.providers.local import LocalProvider
from capabilities.providers.remote import RemoteProvider

CapabilityName = Literal["tts", "imagegen", "videogen", "llm"]
Provider = BaseProvider | LLMProvider

_global_router: "CapabilityRouter | None" = None


class CapabilityRouter:
    def __init__(self, config: CapabilityConfig | None = None):
        self._config = config or load_capability_config()

    @property
    def config(self) -> CapabilityConfig:
        return self._config

    def provider_for(self, capability: CapabilityName) -> Provider:
        endpoint = getattr(self._config, capability, None)
        if endpoint is None:
            raise ValueError(f"Unknown capability: {capability}")

        if capability == "llm":
            if endpoint.provider != "remote":
                raise ValueError("LLM capability only supports provider=remote")
            if not endpoint.remote_url:
                raise ValueError("CAP_LLM_REMOTE_URL is required")
            return LLMProvider(base_url=endpoint.remote_url, api_key=endpoint.remote_api_key)

        if endpoint.provider == "remote":
            if not endpoint.remote_url:
                raise ValueError(
                    f"CAP_{capability.upper()}_REMOTE_URL is required when provider=remote"
                )
            return RemoteProvider(base_url=endpoint.remote_url, api_key=endpoint.remote_api_key)

        return LocalProvider()


def get_capability_router(capability: CapabilityName) -> Provider:
    """Get the provider for a specific capability (global cached config)."""
    global _global_router
    if _global_router is None:
        _global_router = CapabilityRouter()
    return _global_router.provider_for(capability)
