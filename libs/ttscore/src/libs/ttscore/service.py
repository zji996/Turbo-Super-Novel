"""TTS provider service - factory and registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import TTSProvider, TTSError

if TYPE_CHECKING:
    pass


# Provider registry
_PROVIDERS: dict[str, type[TTSProvider]] = {}


def register_provider(name: str, provider_class: type[TTSProvider]) -> None:
    """Register a TTS provider.
    
    Args:
        name: Provider name (e.g., "edge", "azure").
        provider_class: The provider class.
    """
    _PROVIDERS[name.lower()] = provider_class


def list_available_providers() -> list[str]:
    """List all registered provider names."""
    return list(_PROVIDERS.keys())


def get_tts_provider(provider_name: str = "edge", **kwargs) -> TTSProvider:
    """Get a TTS provider instance.
    
    Args:
        provider_name: Name of the provider (e.g., "edge", "azure").
        **kwargs: Provider-specific configuration.
    
    Returns:
        An instance of the requested provider.
    
    Raises:
        TTSError: If provider is not available.
    """
    name = provider_name.lower()
    
    # Lazy load providers
    if name == "edge" and name not in _PROVIDERS:
        try:
            from .providers.edge_tts import EdgeTTSProvider
            register_provider("edge", EdgeTTSProvider)
        except ImportError as e:
            raise TTSError(
                f"Edge TTS not available. Install with: pip install edge-tts. Error: {e}"
            ) from e
    
    if name == "azure" and name not in _PROVIDERS:
        try:
            from .providers.azure import AzureTTSProvider
            register_provider("azure", AzureTTSProvider)
        except ImportError as e:
            raise TTSError(
                f"Azure TTS not available. Install with: pip install azure-cognitiveservices-speech. Error: {e}"
            ) from e
    
    if name not in _PROVIDERS:
        available = list_available_providers()
        raise TTSError(
            f"Unknown TTS provider: {provider_name}. "
            f"Available providers: {available or ['none registered']}"
        )
    
    return _PROVIDERS[name](**kwargs)
