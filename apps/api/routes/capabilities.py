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
        from libs.ttscore import get_tts_provider
    except ImportError:
        return {"error": "ttscore not installed", "voices": []}
    
    try:
        tts = get_tts_provider(provider)
        voices = await tts.list_voices(language=language)
        return {
            "provider": provider,
            "voices": [v.to_dict() for v in voices],
        }
    except Exception as e:
        return {"error": str(e), "voices": []}


@router.get("/tts/providers")
async def list_tts_providers() -> dict:
    """List available TTS providers."""
    try:
        from libs.ttscore import list_available_providers
        return {"providers": list_available_providers()}
    except ImportError:
        return {"providers": [], "error": "ttscore not installed"}


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
        from libs.imagegencore import get_imagegen_provider
    except ImportError:
        return {"error": "imagegencore not installed", "models": []}
    
    try:
        imagegen = get_imagegen_provider(provider)
        models = await imagegen.list_models()
        await imagegen.close() if hasattr(imagegen, 'close') else None
        return {
            "provider": imagegen.provider_name,
            "models": [m.to_dict() for m in models],
        }
    except Exception as e:
        return {"error": str(e), "models": []}


@router.get("/imagegen/providers")
async def list_imagegen_providers() -> dict:
    """List available image generation providers."""
    try:
        from libs.imagegencore import list_available_providers
        return {"providers": list_available_providers()}
    except ImportError:
        return {"providers": [], "error": "imagegencore not installed"}


@router.get("/imagegen/current-model")
async def get_current_imagegen_model(provider: str | None = None) -> dict:
    """Get the currently loaded model.
    
    Args:
        provider: Image generation provider name.
    """
    try:
        from libs.imagegencore import get_imagegen_provider
    except ImportError:
        return {"error": "imagegencore not installed", "model": None}
    
    try:
        imagegen = get_imagegen_provider(provider)
        model = await imagegen.get_current_model()
        await imagegen.close() if hasattr(imagegen, 'close') else None
        return {
            "provider": imagegen.provider_name,
            "model": model,
        }
    except Exception as e:
        return {"error": str(e), "model": None}
