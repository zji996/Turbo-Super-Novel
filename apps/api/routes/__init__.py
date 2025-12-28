"""API Routes package."""

from .novel import router as novel_router
from .capabilities import router as capabilities_router

__all__ = ["novel_router", "capabilities_router"]
