from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from capabilities import get_capability_router

router = APIRouter(prefix="/v1/imagegen", tags=["imagegen"])


class CreateImageGenJobRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    params: dict[str, Any] = Field(default_factory=dict)


@router.post("/jobs")
async def create_imagegen_job(request: CreateImageGenJobRequest) -> dict:
    cap = get_capability_router("imagegen")
    if getattr(cap, "provider_type", "local") != "remote":
        raise HTTPException(
            status_code=501,
            detail="imagegen not available in lightweight API; set CAP_IMAGEGEN_PROVIDER=remote",
        )

    try:
        return await cap.request_json(  # type: ignore[attr-defined]
            "POST",
            "/v1/imagegen/jobs",
            json={"prompt": request.prompt, **(request.params or {})},
            timeout=120,
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=int(exc.response.status_code),
            detail=(exc.response.text or str(exc)),
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/jobs/{job_id}")
async def get_imagegen_job(job_id: str) -> dict:
    cap = get_capability_router("imagegen")
    if getattr(cap, "provider_type", "local") != "remote":
        raise HTTPException(
            status_code=501,
            detail="imagegen not available in lightweight API; set CAP_IMAGEGEN_PROVIDER=remote",
        )
    try:
        return await cap.request_json(  # type: ignore[attr-defined]
            "GET",
            f"/v1/imagegen/jobs/{job_id}",
            timeout=60,
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=int(exc.response.status_code),
            detail=(exc.response.text or str(exc)),
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

