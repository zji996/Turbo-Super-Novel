"""Image generation provider service - factory and registry."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .base import ImageGenProvider, ImageGenError

if TYPE_CHECKING:
    pass


# Provider registry
_PROVIDERS: dict[str, type[ImageGenProvider]] = {}


def register_provider(name: str, provider_class: type[ImageGenProvider]) -> None:
    """Register an image generation provider.
    
    Args:
        name: Provider name (e.g., "sd_webui", "comfyui").
        provider_class: The provider class.
    """
    _PROVIDERS[name.lower()] = provider_class


def list_available_providers() -> list[str]:
    """List all registered provider names."""
    return list(_PROVIDERS.keys())


def get_imagegen_provider(
    provider_name: str | None = None,
    **kwargs,
) -> ImageGenProvider:
    """Get an image generation provider instance.
    
    Args:
        provider_name: Name of the provider (e.g., "sd_webui", "comfyui").
                       If None, uses IMAGEGEN_PROVIDER env var or defaults to "sd_webui".
        **kwargs: Provider-specific configuration.
    
    Returns:
        An instance of the requested provider.
    
    Raises:
        ImageGenError: If provider is not available.
    """
    if provider_name is None:
        provider_name = os.getenv("IMAGEGEN_PROVIDER", "sd_webui")
    
    name = provider_name.lower()
    
    # Lazy load providers
    if name == "sd_webui" and name not in _PROVIDERS:
        from .providers.sd_webui import SDWebUIProvider
        register_provider("sd_webui", SDWebUIProvider)
    
    if name == "comfyui" and name not in _PROVIDERS:
        try:
            from .providers.comfyui import ComfyUIProvider
            register_provider("comfyui", ComfyUIProvider)
        except ImportError as e:
            raise ImageGenError(f"ComfyUI provider not available: {e}") from e
    
    if name not in _PROVIDERS:
        available = list_available_providers()
        raise ImageGenError(
            f"Unknown image generation provider: {provider_name}. "
            f"Available providers: {available or ['none registered']}"
        )
    
    return _PROVIDERS[name](**kwargs)
