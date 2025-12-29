"""Capabilities API routes - Individual module endpoints."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/v1/capabilities", tags=["capabilities"])


# ============================================================================
# TTS Endpoints
# ============================================================================


@router.get("/tts/voices")
async def list_tts_voices(
    provider: str = "edge",
    language: str | None = None,
) -> dict:
    """List available TTS voices.

    Args:
        provider: TTS provider name (default: "edge").
        language: Filter by language code (e.g., "zh-CN").
    """
    try:
        from tts import get_tts_provider
    except ImportError:
        return {"error": "tts not installed", "voices": []}

    try:
        tts = get_tts_provider(provider)
        voices = await tts.list_voices(language=language)
        return {
            "provider": provider,
            "voices": [v.to_dict() for v in voices],
        }
    except Exception as exc:
        return {"error": str(exc), "voices": []}


@router.get("/tts/providers")
async def list_tts_providers() -> dict:
    """List available TTS providers."""
    try:
        from tts import list_available_providers
    except ImportError:
        return {"providers": [], "error": "tts not installed"}

    return {"providers": list_available_providers()}


# ============================================================================
# Image Generation Endpoints
# ============================================================================


@router.get("/imagegen/models")
async def list_imagegen_models(provider: str | None = None) -> dict:
    """List available image generation models.

    Args:
        provider: Image generation provider name (default from env).
    """
    try:
        from imagegen import get_imagegen_provider
    except ImportError:
        return {"error": "imagegen not installed", "models": []}

    try:
        imagegen = get_imagegen_provider(provider)
        models = await imagegen.list_models()
        if hasattr(imagegen, "close"):
            await imagegen.close()
        return {
            "provider": imagegen.provider_name,
            "models": [m.to_dict() for m in models],
        }
    except Exception as exc:
        return {"error": str(exc), "models": []}


@router.get("/imagegen/providers")
async def list_imagegen_providers() -> dict:
    """List available image generation providers."""
    try:
        from imagegen import list_available_providers
    except ImportError:
        return {"providers": [], "error": "imagegen not installed"}

    return {"providers": list_available_providers()}


@router.get("/imagegen/current-model")
async def get_current_imagegen_model(provider: str | None = None) -> dict:
    """Get the currently loaded model.

    Args:
        provider: Image generation provider name.
    """
    try:
        from imagegen import get_imagegen_provider
    except ImportError:
        return {"error": "imagegen not installed", "model": None}

    try:
        imagegen = get_imagegen_provider(provider)
        model = await imagegen.get_current_model()
        if hasattr(imagegen, "close"):
            await imagegen.close()
        return {
            "provider": imagegen.provider_name,
            "model": model,
        }
    except Exception as exc:
        return {"error": str(exc), "model": None}
