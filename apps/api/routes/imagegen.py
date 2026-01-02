from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from capabilities import get_capability_router
from schemas import ImageGenJobResponse

router = APIRouter(prefix="/v1/imagegen", tags=["imagegen"])


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────


class CreateImageGenJobRequest(BaseModel):
    """Request body for creating an image generation job."""

    prompt: str = Field(..., min_length=1, max_length=8000, description="Image prompt")
    enhance_prompt: bool = Field(
        default=False,
        description="Use LLM to enhance the prompt before submission",
    )
    width: int | None = Field(default=None, ge=256, le=2048, description="Image width")
    height: int | None = Field(
        default=None, ge=256, le=2048, description="Image height"
    )
    num_inference_steps: int | None = Field(
        default=None, ge=1, le=100, description="Number of inference steps"
    )
    guidance_scale: float | None = Field(
        default=None, ge=0.0, le=20.0, description="Guidance scale"
    )
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    negative_prompt: str | None = Field(
        default=None, max_length=4000, description="Negative prompt"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────


def _get_remote_provider():
    """Get the imagegen remote provider or raise 501 if not configured."""
    cap = get_capability_router("imagegen")
    if getattr(cap, "provider_type", "local") != "remote":
        raise HTTPException(
            status_code=501,
            detail="imagegen not available; set CAP_IMAGEGEN_PROVIDER=remote",
        )
    return cap


async def _maybe_enhance_prompt(prompt: str, enabled: bool) -> str:
    if not enabled:
        return prompt
    try:
        llm = get_capability_router("llm")
        if not hasattr(llm, "enhance_prompt"):
            raise RuntimeError("LLM provider does not support enhance_prompt")
        return await llm.enhance_prompt(prompt, context="image_generation")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"LLM prompt enhancement unavailable: {exc}",
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@router.post("/jobs", response_model=ImageGenJobResponse)
async def create_imagegen_job(request: CreateImageGenJobRequest) -> dict:
    """Create an image generation job.

    This endpoint forwards the request to the configured remote Z-Image API.
    """
    cap = _get_remote_provider()

    # Build kwargs from optional fields
    kwargs: dict[str, Any] = {}
    if request.width is not None:
        kwargs["width"] = request.width
    if request.height is not None:
        kwargs["height"] = request.height
    if request.num_inference_steps is not None:
        kwargs["num_inference_steps"] = request.num_inference_steps
    if request.guidance_scale is not None:
        kwargs["guidance_scale"] = request.guidance_scale
    if request.seed is not None:
        kwargs["seed"] = request.seed
    if request.negative_prompt is not None:
        kwargs["negative_prompt"] = request.negative_prompt

    prompt = await _maybe_enhance_prompt(request.prompt, request.enhance_prompt)

    try:
        return await cap.submit_imagegen_job(prompt=prompt, **kwargs)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=int(exc.response.status_code),
            detail=(exc.response.text or str(exc)),
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=ImageGenJobResponse)
async def get_imagegen_job(job_id: str) -> dict:
    """Get the status of an image generation job."""
    cap = _get_remote_provider()

    try:
        return await cap.get_imagegen_job(job_id)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=int(exc.response.status_code),
            detail=(exc.response.text or str(exc)),
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/cancel")
async def cancel_imagegen_job(job_id: str) -> dict:
    """Cancel an image generation job."""
    cap = _get_remote_provider()

    try:
        return await cap.cancel_imagegen_job(job_id)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=int(exc.response.status_code),
            detail=(exc.response.text or str(exc)),
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/history")
async def get_imagegen_history(
    limit: int = Query(default=20, ge=1, le=100, description="Number of items to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
) -> list:
    """Get image generation history from the remote Z-Image API."""
    cap = _get_remote_provider()

    try:
        return await cap.get_imagegen_history(limit=limit, offset=offset)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=int(exc.response.status_code),
            detail=(exc.response.text or str(exc)),
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
