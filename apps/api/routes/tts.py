"""TTS API routes."""

from __future__ import annotations

import json
from pathlib import Path
import uuid
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

import httpx
from capabilities import get_capability_router
from db import SpeakerProfile, TTSJob, ensure_schema, session_scope
from s3 import ensure_bucket_exists, presigned_get_url, s3_bucket_name, s3_client
from schemas import TTSJobResponse

router = APIRouter(prefix="/v1/tts", tags=["tts"])


def _normalize_provider(provider: str) -> str:
    raw = (provider or "").strip().lower()
    if raw in {"glm_tts", "glm-tts", "glmtts"}:
        return "glm_tts"
    return raw


class CreateTTSJobRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    enhance_prompt: bool = Field(
        default=False,
        description="Use LLM to enhance/segment the text before TTS",
    )
    provider: str = Field(default="glm_tts")
    prompt_text: str | None = None
    prompt_audio_id: str | None = None
    sample_rate: int = Field(default=24000)
    config: dict = Field(default_factory=dict)


class CreateSpeakerProfileRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    provider: str = Field(default="glm_tts")
    sample_rate: int = Field(default=24000)
    prompt_text: str = Field(..., min_length=1)
    prompt_audio_id: str = Field(...)
    config: dict = Field(default_factory=dict)
    is_default: bool = False


class SpeakerProfileResponse(BaseModel):
    id: str
    name: str
    description: str | None
    provider: str
    sample_rate: int
    prompt_text: str
    prompt_audio_url: str | None
    is_default: bool
    created_at: str
    updated_at: str


class CreateJobWithProfileRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    profile_id: str = Field(...)
    enhance_prompt: bool = Field(
        default=False,
        description="Use LLM to enhance/segment the text before TTS",
    )


def _speaker_profile_response(profile: SpeakerProfile) -> SpeakerProfileResponse:
    return SpeakerProfileResponse(
        id=str(profile.id),
        name=str(profile.name),
        description=profile.description,
        provider=str(profile.provider),
        sample_rate=int(profile.sample_rate),
        prompt_text=str(profile.prompt_text),
        prompt_audio_url=presigned_get_url(
            bucket=str(profile.prompt_audio_bucket), key=str(profile.prompt_audio_key)
        ),
        is_default=bool(profile.is_default),
        created_at=profile.created_at.isoformat() if profile.created_at else "",
        updated_at=profile.updated_at.isoformat() if profile.updated_at else "",
    )


def _parse_config_json(value: str | None) -> dict:
    if value is None:
        return {}
    raw = str(value).strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid config JSON: {exc}") from exc
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail="config must be a JSON object")
    return parsed


async def _maybe_enhance_tts_text(text: str, enabled: bool) -> str:
    if not enabled:
        return text
    try:
        llm = get_capability_router("llm")
        if not hasattr(llm, "enhance_prompt"):
            raise RuntimeError("LLM provider does not support enhance_prompt")
        return await llm.enhance_prompt(text, context="tts_text")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"LLM text enhancement unavailable: {exc}",
        ) from exc


@router.get("/providers")
async def list_tts_providers() -> dict:
    cap = get_capability_router("tts")
    if getattr(cap, "provider_type", "local") == "remote" and hasattr(cap, "request_json"):
        return await cap.request_json("GET", "/v1/tts/providers")
    return {"providers": ["glm_tts"]}


@router.post("/speaker-profiles", response_model=SpeakerProfileResponse)
async def create_speaker_profile(
    name: str = Form(...),
    prompt_text: str = Form(...),
    prompt_audio: UploadFile = File(...),
    description: str | None = Form(None),
    provider: str = Form("glm_tts"),
    sample_rate: int = Form(24000),
    is_default: bool = Form(False),
    config: str | None = Form(None),
) -> SpeakerProfileResponse:
    if not str(name).strip():
        raise HTTPException(status_code=422, detail="name cannot be empty")
    if not str(prompt_text).strip():
        raise HTTPException(status_code=422, detail="prompt_text cannot be empty")

    suffix = Path(prompt_audio.filename or "").suffix.lower()
    if suffix != ".wav":
        raise HTTPException(status_code=422, detail="prompt audio must be a .wav file")

    config_obj = _parse_config_json(config)
    provider_norm = _normalize_provider(provider)

    try:
        bucket = s3_bucket_name()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        ensure_bucket_exists(bucket)
        client = s3_client()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    profile_uuid = uuid4()
    profile_id = str(profile_uuid)
    audio_key = f"tts/speaker-profiles/{profile_id}/prompt.wav"

    client.upload_fileobj(
        prompt_audio.file,
        bucket,
        audio_key,
        ExtraArgs={"ContentType": prompt_audio.content_type or "audio/wav"},
    )

    ok, err = ensure_schema()
    if not ok:
        raise HTTPException(status_code=503, detail=f"DB not ready: {err}")

    from sqlalchemy import update

    with session_scope() as session:
        if is_default:
            session.execute(update(SpeakerProfile).values(is_default=False))
        profile = SpeakerProfile(
            id=profile_uuid,
            name=str(name),
            description=description,
            provider=provider_norm,
            sample_rate=int(sample_rate),
            prompt_text=str(prompt_text),
            prompt_audio_bucket=bucket,
            prompt_audio_key=audio_key,
            config=config_obj,
            is_default=bool(is_default),
        )
        session.add(profile)
        session.flush()
        return _speaker_profile_response(profile)


@router.get("/speaker-profiles")
async def list_speaker_profiles() -> dict:
    ok, err = ensure_schema()
    if not ok:
        raise HTTPException(status_code=503, detail=f"DB not ready: {err}")

    from sqlalchemy import select

    with session_scope() as session:
        rows = session.scalars(select(SpeakerProfile).order_by(SpeakerProfile.created_at.desc()))
        return {"speaker_profiles": [_speaker_profile_response(r).model_dump() for r in rows]}


@router.get("/speaker-profiles/{profile_id}", response_model=SpeakerProfileResponse)
async def get_speaker_profile(profile_id: str) -> SpeakerProfileResponse:
    try:
        pid = uuid.UUID(profile_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid profile_id")

    ok, err = ensure_schema()
    if not ok:
        raise HTTPException(status_code=503, detail=f"DB not ready: {err}")

    with session_scope() as session:
        row = session.get(SpeakerProfile, pid)
        if row is None:
            raise HTTPException(status_code=404, detail="SpeakerProfile not found")
        return _speaker_profile_response(row)


@router.put("/speaker-profiles/{profile_id}", response_model=SpeakerProfileResponse)
async def update_speaker_profile(
    profile_id: str,
    name: str | None = Form(None),
    prompt_text: str | None = Form(None),
    prompt_audio: UploadFile | None = File(None),
    description: str | None = Form(None),
    provider: str | None = Form(None),
    sample_rate: int | None = Form(None),
    is_default: bool | None = Form(None),
    config: str | None = Form(None),
) -> SpeakerProfileResponse:
    try:
        pid = uuid.UUID(profile_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid profile_id")

    ok, err = ensure_schema()
    if not ok:
        raise HTTPException(status_code=503, detail=f"DB not ready: {err}")

    config_obj = _parse_config_json(config) if config is not None else None

    from sqlalchemy import update

    with session_scope() as session:
        row = session.get(SpeakerProfile, pid)
        if row is None:
            raise HTTPException(status_code=404, detail="SpeakerProfile not found")

        if name is not None:
            if not str(name).strip():
                raise HTTPException(status_code=422, detail="name cannot be empty")
            row.name = str(name)
        if prompt_text is not None:
            if not str(prompt_text).strip():
                raise HTTPException(status_code=422, detail="prompt_text cannot be empty")
            row.prompt_text = str(prompt_text)
        if description is not None:
            row.description = description
        if provider is not None:
            row.provider = _normalize_provider(provider)
        if sample_rate is not None:
            row.sample_rate = int(sample_rate)
        if config_obj is not None:
            row.config = config_obj

        if is_default is not None:
            if is_default:
                session.execute(update(SpeakerProfile).values(is_default=False))
            row.is_default = bool(is_default)

        if prompt_audio is not None:
            suffix = Path(prompt_audio.filename or "").suffix.lower()
            if suffix != ".wav":
                raise HTTPException(status_code=422, detail="prompt audio must be a .wav file")

            try:
                ensure_bucket_exists(str(row.prompt_audio_bucket))
                client = s3_client()
            except Exception as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            client.upload_fileobj(
                prompt_audio.file,
                str(row.prompt_audio_bucket),
                str(row.prompt_audio_key),
                ExtraArgs={"ContentType": prompt_audio.content_type or "audio/wav"},
            )

        session.flush()
        return _speaker_profile_response(row)


@router.delete("/speaker-profiles/{profile_id}")
async def delete_speaker_profile(profile_id: str) -> dict:
    try:
        pid = uuid.UUID(profile_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid profile_id")

    ok, err = ensure_schema()
    if not ok:
        raise HTTPException(status_code=503, detail=f"DB not ready: {err}")

    with session_scope() as session:
        row = session.get(SpeakerProfile, pid)
        if row is None:
            raise HTTPException(status_code=404, detail="SpeakerProfile not found")

        bucket = str(row.prompt_audio_bucket)
        key = str(row.prompt_audio_key)
        session.delete(row)

    try:
        s3_client().delete_object(Bucket=bucket, Key=key)
    except Exception:
        pass

    return {"deleted": True}


@router.post("/jobs")
async def create_tts_job(request: CreateTTSJobRequest) -> dict:
    cap = get_capability_router("tts")
    provider = _normalize_provider(request.provider)
    text_final = await _maybe_enhance_tts_text(request.text, request.enhance_prompt)

    if getattr(cap, "provider_type", "local") == "remote":
        try:
            return await cap.submit_tts_job(
                text=text_final,
                provider=provider,
                prompt_text=request.prompt_text,
                prompt_audio_id=request.prompt_audio_id,
                sample_rate=int(request.sample_rate),
                config=dict(request.config or {}),
            )
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=int(exc.response.status_code),
                detail=(exc.response.text or str(exc)),
            ) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    if provider != "glm_tts":
        raise HTTPException(status_code=422, detail=f"Unsupported provider: {provider}")
    if not request.prompt_text or not request.prompt_text.strip():
        raise HTTPException(status_code=422, detail="prompt_text is required for glm_tts")
    if not request.prompt_audio_id or not request.prompt_audio_id.strip():
        raise HTTPException(status_code=422, detail="prompt_audio_id is required for glm_tts")
    if int(request.sample_rate) not in (24000, 32000):
        raise HTTPException(status_code=422, detail="glm_tts sample_rate must be 24000 or 32000")

    try:
        bucket = s3_bucket_name()
        ensure_bucket_exists(bucket)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    job_uuid = uuid4()
    job_id = str(job_uuid)
    output_key = f"tts/outputs/{job_id}.wav"

    prompt_audio_bucket: str | None = None
    prompt_audio_key: str | None = None
    if request.prompt_audio_id:
        prompt_audio_bucket = bucket
        if "/" not in request.prompt_audio_id:
            raise HTTPException(
                status_code=422,
                detail="prompt_audio_id must be a full S3 key (e.g. tts/speaker-profiles/<id>/prompt.wav); consider /v1/tts/jobs/with-profile",
            )
        prompt_audio_key = request.prompt_audio_id

    ensure_schema()
    with session_scope() as session:
        session.add(
            TTSJob(
                id=job_uuid,
                status="CREATED",
                text=text_final,
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
        await cap.submit_tts_job(
            job_id=job_id,
            text=text_final,
            provider=provider,
            output_bucket=bucket,
            output_key=output_key,
            prompt_text=request.prompt_text,
            prompt_audio_bucket=prompt_audio_bucket,
            prompt_audio_key=prompt_audio_key,
            sample_rate=int(request.sample_rate),
            **(request.config or {}),
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
    cap = get_capability_router("tts")
    if getattr(cap, "provider_type", "local") == "remote":
        url = f"{getattr(cap, 'base_url', '').rstrip('/')}/v1/tts/jobs/with-prompt"
        if not url.startswith("http"):
            raise HTTPException(status_code=500, detail="Invalid remote base_url for CAP_TTS_REMOTE_URL")
        headers: dict[str, str] = {}
        api_key = getattr(cap, "api_key", None)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                url,
                data={
                    "text": text,
                    "prompt_text": prompt_text,
                    "provider": provider,
                    "sample_rate": str(sample_rate),
                },
                files={
                    "prompt_audio": (
                        prompt_audio.filename or "prompt.wav",
                        prompt_audio.file,
                        prompt_audio.content_type or "audio/wav",
                    )
                },
                headers=headers,
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise HTTPException(
                    status_code=int(exc.response.status_code),
                    detail=(exc.response.text or str(exc)),
                ) from exc
            return resp.json()

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
        await cap.submit_tts_job(
            job_id=job_id,
            text=text,
            provider=provider_norm,
            output_bucket=bucket,
            output_key=output_key,
            prompt_text=prompt_text,
            prompt_audio_bucket=bucket,
            prompt_audio_key=prompt_key,
            sample_rate=int(sample_rate),
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


@router.post("/jobs/with-profile")
async def create_tts_job_with_profile(request: CreateJobWithProfileRequest) -> dict:
    cap = get_capability_router("tts")
    text_final = await _maybe_enhance_tts_text(request.text, request.enhance_prompt)
    if getattr(cap, "provider_type", "local") == "remote" and hasattr(cap, "request_json"):
        try:
            payload = request.model_dump()
            payload["text"] = text_final
            payload.pop("enhance_prompt", None)
            return await cap.request_json("POST", "/v1/tts/jobs/with-profile", json=payload)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=int(exc.response.status_code),
                detail=(exc.response.text or str(exc)),
            ) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        profile_uuid = uuid.UUID(request.profile_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid profile_id")

    ok, err = ensure_schema()
    if not ok:
        raise HTTPException(status_code=503, detail=f"DB not ready: {err}")

    try:
        output_bucket = s3_bucket_name()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        ensure_bucket_exists(output_bucket)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    job_uuid = uuid4()
    job_id = str(job_uuid)
    output_key = f"tts/outputs/{job_id}.wav"

    with session_scope() as session:
        profile = session.get(SpeakerProfile, profile_uuid)
        if profile is None:
            raise HTTPException(status_code=404, detail="SpeakerProfile not found")

        provider = _normalize_provider(str(profile.provider))
        sample_rate = int(profile.sample_rate)
        prompt_text = str(profile.prompt_text)
        prompt_audio_bucket = str(profile.prompt_audio_bucket)
        prompt_audio_key = str(profile.prompt_audio_key)
        config = dict(profile.config or {})

        session.add(
            TTSJob(
                id=job_uuid,
                status="CREATED",
                text=text_final,
                provider=provider,
                speaker_profile_id=profile_uuid,
                prompt_text=prompt_text,
                prompt_audio_bucket=prompt_audio_bucket,
                prompt_audio_key=prompt_audio_key,
                output_bucket=output_bucket,
                output_key=output_key,
                sample_rate=sample_rate,
                config=config,
            )
        )

    try:
        await cap.submit_tts_job(
            job_id=job_id,
            text=text_final,
            provider=provider,
            output_bucket=output_bucket,
            output_key=output_key,
            prompt_text=prompt_text,
            prompt_audio_bucket=prompt_audio_bucket,
            prompt_audio_key=prompt_audio_key,
            sample_rate=sample_rate,
            **config,
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


@router.get("/jobs/{job_id}", response_model=TTSJobResponse)
async def get_tts_job(job_id: str) -> TTSJobResponse:
    cap = get_capability_router("tts")
    provider_type = getattr(cap, "provider_type", "local")

    try:
        payload: dict = await cap.get_tts_job(job_id)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=int(exc.response.status_code),
            detail=(exc.response.text or str(exc)),
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id")

    celery_status = payload.get("celery_status") or payload.get("status")
    status = str(celery_status or "PENDING")
    error = payload.get("error")
    output_url: str | None = None
    audio_duration_seconds: float | None = None

    try:
        ensure_schema()
        with session_scope() as session:
            row = session.get(TTSJob, job_uuid)
            if row is not None:
                status = str(row.status or status)
                error = row.error or error
                audio_duration_seconds = row.audio_duration_seconds
                if row.output_bucket and row.output_key:
                    output_url = presigned_get_url(bucket=row.output_bucket, key=row.output_key)
    except Exception as exc:
        payload["db_error"] = str(exc)

    return TTSJobResponse(
        job_id=job_id,
        status=status,
        error=error,
        provider_type=str(payload.get("provider_type") or provider_type or "local"),
        output_url=output_url or payload.get("output_url"),
        audio_duration_seconds=audio_duration_seconds,
    )
