from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

ProviderType = Literal["local", "remote"]


class BaseProvider(ABC):
    provider_type: ProviderType

    @abstractmethod
    async def submit_tts_job(self, *, text: str, provider: str, **kwargs: Any) -> dict:
        """Submit a TTS job and return job info."""

    @abstractmethod
    async def get_tts_job(self, job_id: str) -> dict:
        """Get TTS job status."""

    @abstractmethod
    async def submit_imagegen_job(self, *, prompt: str, **kwargs: Any) -> dict:
        """Submit an image generation job."""

    @abstractmethod
    async def get_imagegen_job(self, job_id: str) -> dict:
        """Get image generation job status."""

    @abstractmethod
    async def submit_videogen_job(self, *, prompt: str, image_path: str, **kwargs: Any) -> dict:
        """Submit a video generation job."""

    @abstractmethod
    async def get_videogen_job(self, job_id: str) -> dict:
        """Get video generation job status."""

