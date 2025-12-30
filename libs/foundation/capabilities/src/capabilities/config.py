from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Mapping

ProviderType = Literal["local", "remote"]


@dataclass(frozen=True, slots=True)
class CapabilityEndpoint:
    provider: ProviderType = "local"
    remote_url: str | None = None
    remote_api_key: str | None = None


@dataclass(frozen=True, slots=True)
class CapabilityConfig:
    tts: CapabilityEndpoint = field(default_factory=CapabilityEndpoint)
    imagegen: CapabilityEndpoint = field(default_factory=CapabilityEndpoint)
    videogen: CapabilityEndpoint = field(default_factory=CapabilityEndpoint)
    llm: CapabilityEndpoint = field(default_factory=CapabilityEndpoint)


def load_capability_config(environ: Mapping[str, str] | None = None) -> CapabilityConfig:
    """Load capability config from environment variables."""

    env = environ or os.environ

    def _load_endpoint(prefix: str) -> CapabilityEndpoint:
        provider_raw = (env.get(f"CAP_{prefix}_PROVIDER") or "local").strip().lower()
        provider: ProviderType = "remote" if provider_raw == "remote" else "local"
        remote_url = (env.get(f"CAP_{prefix}_REMOTE_URL") or "").strip() or None
        remote_api_key = (env.get(f"CAP_{prefix}_REMOTE_API_KEY") or "").strip() or None
        return CapabilityEndpoint(
            provider=provider,
            remote_url=remote_url,
            remote_api_key=remote_api_key,
        )

    return CapabilityConfig(
        tts=_load_endpoint("TTS"),
        imagegen=_load_endpoint("IMAGEGEN"),
        videogen=_load_endpoint("VIDEOGEN"),
        llm=_load_endpoint("LLM"),
    )
