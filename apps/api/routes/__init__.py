"""API Routes package."""

from .novel import router as novel_router
from .capabilities import router as capabilities_router
from .tts import router as tts_router
from .llm import router as llm_router
from .imagegen import router as imagegen_router

__all__ = ["novel_router", "capabilities_router", "tts_router", "llm_router", "imagegen_router"]
