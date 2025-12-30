"""TTS API routes."""

from __future__ import annotations

import json
from pathlib import Path
import uuid
from uuid import uuid4

from celery.result import AsyncResult
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from celery_app import celery_app
from db import TTSJob, ensure_schema, session_scope
from s3 import ensure_bucket_exists, s3_bucket_name, s3_client

router = APIRouter(prefix="/v1/tts", tags=["tts"])


_PROMPT_AUDIO_PREFIX = "tts/prompt-audios/"


def _normalize_provider(provider: str) -> str:
    raw = (provider or "").strip().lower()
    if raw in {"glm_tts", "glm-tts", "glmtts"}:
        return "glm_tts"
    return raw


def _presigned_get_url(*, bucket: str, key: str, expires_in: int = 3600) -> str:
    client = s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=int(expires_in),
    )


class CreateTTSJobRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    provider: str = Field(default="glm_tts")
    prompt_text: str | None = None
    prompt_audio_id: str | None = None
    sample_rate: int = Field(default=24000)
    config: dict = Field(default_factory=dict)


@router.get("/providers")
async def list_tts_providers() -> dict:
    return {"providers": ["glm_tts", "edge"]}


@router.post("/prompt-audios")
async def upload_prompt_audio(
    name: str = Form(...),
    text: str = Form(...),
    audio: UploadFile = File(...),
) -> dict:
    """Upload a reference audio for zero-shot TTS."""
    if not name.strip():
        raise HTTPException(status_code=422, detail="name cannot be empty")
    if not text.strip():
        raise HTTPException(status_code=422, detail="text cannot be empty")

    suffix = Path(audio.filename or "").suffix.lower()
    if suffix != ".wav":
        raise HTTPException(status_code=422, detail="prompt audio must be a .wav file")

    bucket = s3_bucket_name()
    ensure_bucket_exists(bucket)
    client = s3_client()

    prompt_id = str(uuid4())
    audio_key = f"{_PROMPT_AUDIO_PREFIX}{prompt_id}.wav"
    meta_key = f"{_PROMPT_AUDIO_PREFIX}{prompt_id}.json"

    client.upload_fileobj(
        audio.file,
        bucket,
        audio_key,
        ExtraArgs={"ContentType": audio.content_type or "audio/wav"},
    )

    meta = {"id": prompt_id, "name": name, "text": text, "audio_key": audio_key}
    client.put_object(
        Bucket=bucket,
        Key=meta_key,
        Body=json.dumps(meta, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json; charset=utf-8",
    )

    return {
        "id": prompt_id,
        "name": name,
        "text": text,
        "bucket": bucket,
        "audio_key": audio_key,
        "audio_url": _presigned_get_url(bucket=bucket, key=audio_key),
    }


@router.get("/prompt-audios")
async def list_prompt_audios(limit: int = 100) -> dict:
    """List available prompt audios (best-effort)."""
    bucket = s3_bucket_name()
    client = s3_client()

    resp = client.list_objects_v2(
        Bucket=bucket, Prefix=_PROMPT_AUDIO_PREFIX, MaxKeys=min(int(limit), 1000)
    )
    contents = resp.get("Contents") or []
    meta_keys = [obj["Key"] for obj in contents if str(obj.get("Key", "")).endswith(".json")]

    items: list[dict] = []
    for key in meta_keys:
        try:
            obj = client.get_object(Bucket=bucket, Key=key)
            body = obj["Body"].read()
            meta = json.loads(body.decode("utf-8"))
            audio_key = meta.get("audio_key")
            if not audio_key:
                continue
            items.append(
                {
                    "id": meta.get("id") or Path(key).stem,
                    "name": meta.get("name"),
                    "text": meta.get("text"),
                    "bucket": bucket,
                    "audio_key": audio_key,
                    "audio_url": _presigned_get_url(bucket=bucket, key=audio_key),
                }
            )
        except Exception:
            continue

    return {"prompt_audios": items}


@router.post("/jobs")
async def create_tts_job(request: CreateTTSJobRequest) -> dict:
    provider = _normalize_provider(request.provider)
    if provider not in {"glm_tts", "edge"}:
        raise HTTPException(status_code=422, detail=f"Unsupported provider: {provider}")

    if provider == "glm_tts":
        if not request.prompt_text or not request.prompt_text.strip():
            raise HTTPException(
                status_code=422, detail="prompt_text is required for glm_tts"
            )
        if not request.prompt_audio_id or not request.prompt_audio_id.strip():
            raise HTTPException(
                status_code=422, detail="prompt_audio_id is required for glm_tts"
            )
        if int(request.sample_rate) not in (24000, 32000):
            raise HTTPException(
                status_code=422, detail="glm_tts sample_rate must be 24000 or 32000"
            )

    bucket = s3_bucket_name()
    ensure_bucket_exists(bucket)

    job_uuid = uuid4()
    job_id = str(job_uuid)
    output_ext = ".wav" if provider == "glm_tts" else ".mp3"
    output_key = f"tts/outputs/{job_id}{output_ext}"

    prompt_audio_bucket: str | None = None
    prompt_audio_key: str | None = None
    if request.prompt_audio_id:
        prompt_audio_bucket = bucket
        if "/" in request.prompt_audio_id:
            prompt_audio_key = request.prompt_audio_id
        else:
            prompt_audio_key = f"{_PROMPT_AUDIO_PREFIX}{request.prompt_audio_id}.wav"

    ensure_schema()
    with session_scope() as session:
        session.add(
            TTSJob(
                id=job_uuid,
                status="CREATED",
                text=request.text,
                provider=provider,
                prompt_text=request.prompt_text,
                prompt_audio_bucket=prompt_audio_bucket,
                prompt_audio_key=prompt_audio_key,
                output_bucket=bucket,
                output_key=output_key,
                sample_rate=int(request.sample_rate),
                config=dict(request.config or {}),
            )
        )

    try:
        celery_app.send_task(
            "tts.synthesize",
            task_id=job_id,
            kwargs={
                "job_id": job_id,
                "text": request.text,
                "provider": provider,
                "output_bucket": bucket,
                "output_key": output_key,
                "prompt_text": request.prompt_text,
                "prompt_audio_bucket": prompt_audio_bucket,
                "prompt_audio_key": prompt_audio_key,
                "sample_rate": int(request.sample_rate),
                **(request.config or {}),
            },
        )
    except Exception as exc:
        ensure_schema()
        with session_scope() as session:
            row = session.get(TTSJob, job_uuid)
            if row is not None:
                row.status = "SUBMIT_FAILED"
                row.error = str(exc)
        raise HTTPException(status_code=500, detail=f"Failed to submit task: {exc}") from exc

    ensure_schema()
    with session_scope() as session:
        row = session.get(TTSJob, job_uuid)
        if row is not None:
            row.status = "SUBMITTED"

    return {"job_id": job_id, "status": "submitted", "output": {"bucket": bucket, "key": output_key}}


@router.post("/jobs/with-prompt")
async def create_tts_job_with_prompt(
    text: str = Form(...),
    prompt_text: str = Form(...),
    prompt_audio: UploadFile = File(...),
    provider: str = Form("glm_tts"),
    sample_rate: int = Form(24000),
) -> dict:
    """Create a TTS job with an uploaded prompt audio (multipart)."""
    provider_norm = _normalize_provider(provider)
    if provider_norm != "glm_tts":
        raise HTTPException(
            status_code=422, detail="jobs/with-prompt currently only supports glm_tts"
        )
    if int(sample_rate) not in (24000, 32000):
        raise HTTPException(status_code=422, detail="sample_rate must be 24000 or 32000")

    suffix = Path(prompt_audio.filename or "").suffix.lower()
    if suffix != ".wav":
        raise HTTPException(status_code=422, detail="prompt audio must be a .wav file")

    bucket = s3_bucket_name()
    ensure_bucket_exists(bucket)
    client = s3_client()

    job_uuid = uuid4()
    job_id = str(job_uuid)
    prompt_key = f"tts/jobs/{job_id}/prompt.wav"
    output_key = f"tts/outputs/{job_id}.wav"

    client.upload_fileobj(
        prompt_audio.file,
        bucket,
        prompt_key,
        ExtraArgs={"ContentType": prompt_audio.content_type or "audio/wav"},
    )

    ensure_schema()
    with session_scope() as session:
        session.add(
            TTSJob(
                id=job_uuid,
                status="CREATED",
                text=text,
                provider=provider_norm,
                prompt_text=prompt_text,
                prompt_audio_bucket=bucket,
                prompt_audio_key=prompt_key,
                output_bucket=bucket,
                output_key=output_key,
                sample_rate=int(sample_rate),
                config={},
            )
        )

    try:
        celery_app.send_task(
            "tts.synthesize",
            task_id=job_id,
            kwargs={
                "job_id": job_id,
                "text": text,
                "provider": provider_norm,
                "output_bucket": bucket,
                "output_key": output_key,
                "prompt_text": prompt_text,
                "prompt_audio_bucket": bucket,
                "prompt_audio_key": prompt_key,
                "sample_rate": int(sample_rate),
            },
        )
    except Exception as exc:
        ensure_schema()
        with session_scope() as session:
            row = session.get(TTSJob, job_uuid)
            if row is not None:
                row.status = "SUBMIT_FAILED"
                row.error = str(exc)
        raise HTTPException(status_code=500, detail=f"Failed to submit task: {exc}") from exc

    ensure_schema()
    with session_scope() as session:
        row = session.get(TTSJob, job_uuid)
        if row is not None:
            row.status = "SUBMITTED"

    return {"job_id": job_id, "status": "submitted"}


@router.get("/jobs/{job_id}")
async def get_tts_job(job_id: str) -> dict:
    result: AsyncResult = celery_app.AsyncResult(job_id)
    payload: dict = {"job_id": job_id, "celery_status": result.status}

    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id")

    try:
        ensure_schema()
        with session_scope() as session:
            row = session.get(TTSJob, job_uuid)
            if row is not None:
                payload["db"] = {
                    "status": row.status,
                    "provider": row.provider,
                    "output": {"bucket": row.output_bucket, "key": row.output_key},
                    "prompt_audio": {
                        "bucket": row.prompt_audio_bucket,
                        "key": row.prompt_audio_key,
                    },
                    "audio_duration_seconds": row.audio_duration_seconds,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                    "error": row.error,
                }
                if row.output_bucket and row.output_key:
                    payload["output_url"] = _presigned_get_url(
                        bucket=row.output_bucket, key=row.output_key
                    )
    except Exception as exc:
        payload["db_error"] = str(exc)

    if result.successful():
        try:
            payload["result"] = result.get(timeout=0)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    elif result.failed():
        try:
            payload["error"] = str(result.result)
        except Exception:
            payload["error"] = "unknown"

    return payload

