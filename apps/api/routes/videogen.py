"""Videogen (TurboDiffusion) API routes."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import uuid
from uuid import uuid4

from celery.result import AsyncResult
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from celery_app import celery_app
from db import (
    VideoGenJob,
    ensure_schema,
    session_scope,
    try_insert_job,
    try_update_job,
)
from s3 import ensure_bucket_exists, presigned_get_url, s3_bucket_name, s3_client
from schemas import VideoGenJobResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/videogen", tags=["videogen"])


class JobObjectRef(BaseModel):
    """Reference to an object stored in S3/MinIO."""

    bucket: str
    key: str


class JobDbInfo(BaseModel):
    """DB persistence status."""

    persisted: bool
    error: str | None = None


class CreateI2VJobResponse(BaseModel):
    """Response for creating an I2V job."""

    job_id: str
    status: str
    input: JobObjectRef
    output: JobObjectRef
    db: JobDbInfo


class JobDbSnapshot(BaseModel):
    """DB snapshot for job status."""

    status: str
    input: JobObjectRef
    output: JobObjectRef
    created_at: str | None = None
    updated_at: str | None = None
    error: str | None = None


@router.get("/models")
def turbodiffusion_models() -> dict:
    try:
        from videogen.paths import turbodiffusion_models_root, wan22_i2v_model_paths
        from videogen.registry import list_artifacts
    except ImportError:
        return {
            "error": "videogen not installed (API is lightweight; run a full TSN API or install tsn-videogen)",
            "available": [],
        }

    root = turbodiffusion_models_root()
    model_paths = wan22_i2v_model_paths(quantized=True)
    artifacts = list_artifacts("TurboWan2.2-I2V-A14B-720P", quantized=True)
    return {
        "models_root": str(root),
        "available": [
            {
                "model": "TurboWan2.2-I2V-A14B-720P",
                "quantized": True,
                "files": [
                    {
                        "name": a.name,
                        "group": a.group,
                        "path": str(root / a.relative_path),
                        "exists": (root / a.relative_path).is_file(),
                    }
                    for a in artifacts
                ],
                "paths": {
                    "vae_path": str(model_paths.vae_path),
                    "text_encoder_path": str(model_paths.text_encoder_path),
                    "high_noise_dit_path": str(model_paths.high_noise_dit_path),
                    "low_noise_dit_path": str(model_paths.low_noise_dit_path),
                },
            }
        ],
    }


@router.post("/wan22-i2v/jobs", response_model=CreateI2VJobResponse)
async def create_wan22_i2v_job(
    prompt: str = Form(..., min_length=1, max_length=8000),
    image: UploadFile = File(...),
    seed: int = Form(0),
    num_steps: int = Form(4, ge=1, le=100),
    quantized: bool = Form(True),
    duration_seconds: float | None = Form(None, gt=0),
) -> dict:
    if duration_seconds is not None:
        fps = float((os.getenv("TD_VIDEO_FPS") or "16").strip() or "16")
        if fps <= 0:
            raise HTTPException(status_code=500, detail=f"Invalid TD_VIDEO_FPS: {fps}")
        num_frames = int(round(float(duration_seconds) * fps))
        if num_frames <= 0:
            min_duration = 0.5 / fps
            raise HTTPException(
                status_code=422,
                detail=(
                    f"duration_seconds too small (computed num_frames={num_frames} with fps={fps}); "
                    f"try duration_seconds>={min_duration:.3f}"
                ),
            )
        if num_frames == 2:
            min_duration = 2.5 / fps
            raise HTTPException(
                status_code=422,
                detail=(
                    "duration_seconds too small for stable VAE encoding "
                    f"(computed num_frames=2 with fps={fps}); try duration_seconds>={min_duration:.3f} or increase TD_VIDEO_FPS"
                ),
            )

    job_uuid = uuid4()
    job_id = str(job_uuid)
    bucket = s3_bucket_name()
    ensure_bucket_exists(bucket)

    suffix = Path(image.filename or "").suffix or ".jpg"
    input_key = f"turbodiffusion/inputs/{job_id}{suffix}"
    output_key = f"turbodiffusion/outputs/{job_id}.mp4"

    client = s3_client()
    client.upload_fileobj(
        image.file,
        bucket,
        input_key,
        ExtraArgs={"ContentType": image.content_type or "application/octet-stream"},
    )

    db_persisted = False
    db_error: str | None = None
    ok, err = try_insert_job(
        VideoGenJob(
            id=job_uuid,
            status="CREATED",
            prompt=prompt,
            seed=int(seed),
            num_steps=int(num_steps),
            quantized=bool(quantized),
            input_bucket=bucket,
            input_key=input_key,
            output_bucket=bucket,
            output_key=output_key,
        )
    )
    if ok:
        db_persisted = True
    else:
        db_error = err
        logger.error("Failed to persist job to DB: %s (%s)", job_id, db_error)

    try:
        celery_app.send_task(
            "cap.videogen.generate",
            task_id=job_id,
            queue="cap.videogen",
            kwargs={
                "job_id": job_id,
                "input_bucket": bucket,
                "input_key": input_key,
                "output_bucket": bucket,
                "output_key": output_key,
                "prompt": prompt,
                "seed": int(seed),
                "num_steps": int(num_steps),
                "quantized": bool(quantized),
                "duration_seconds": float(duration_seconds)
                if duration_seconds is not None
                else None,
            },
        )
    except Exception as exc:
        if db_persisted:
            try_update_job(job_uuid, status="SUBMIT_FAILED", error=str(exc), result=None)
        raise HTTPException(
            status_code=500, detail=f"Failed to submit Celery task: {exc}"
        ) from exc

    if db_persisted:
        ok, err = try_update_job(job_uuid, status="SUBMITTED", error=None, result=None)
        if not ok:
            db_error = err
            logger.error(
                "Failed to update job status to SUBMITTED: %s (%s)", job_id, db_error
            )

    return {
        "job_id": job_id,
        "status": "submitted",
        "input": {"bucket": bucket, "key": input_key},
        "output": {"bucket": bucket, "key": output_key},
        "db": {"persisted": db_persisted, "error": db_error},
    }


@router.get("/jobs/{job_id}", response_model=VideoGenJobResponse)
def get_job(job_id: str) -> dict:
    result: AsyncResult = celery_app.AsyncResult(job_id)
    payload: dict = {"job_id": job_id, "status": result.status, "provider_type": "local"}

    try:
        ensure_schema()
        with session_scope() as session:
            row = session.get(VideoGenJob, uuid.UUID(job_id))
            if row is not None:
                payload["error"] = row.error
    except Exception as exc:
        payload["db_error"] = str(exc)

    if result.successful():
        try:
            value = result.get(timeout=0)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if isinstance(value, dict) and "output_bucket" in value and "output_key" in value:
            payload["video_url"] = presigned_get_url(
                bucket=value["output_bucket"], key=value["output_key"]
            )
            payload["duration_seconds"] = value.get("duration_seconds")
        return payload

    if result.failed():
        try:
            payload["error"] = str(result.result)
        except Exception:
            payload["error"] = "unknown"
        return payload

    return payload
