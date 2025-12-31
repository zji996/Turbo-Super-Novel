"""Common response schemas."""

from __future__ import annotations

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
    error_code: str | None = None


class JobStatusResponse(BaseModel):
    """Base class for job status responses."""

    job_id: str
    status: str
    progress: int | None = None
    error: str | None = None
    provider_type: str = "local"


class TTSJobResponse(JobStatusResponse):
    """TTS job response."""

    output_url: str | None = None
    audio_duration_seconds: float | None = None


class ImageGenJobResponse(JobStatusResponse):
    """Image generation job response."""

    image_url: str | None = None
    error_hint: str | None = None


class VideoGenJobResponse(JobStatusResponse):
    """Video generation job response."""

    video_url: str | None = None
    duration_seconds: float | None = None
