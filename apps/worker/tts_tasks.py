"""TTS Celery tasks."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import traceback
from typing import Any
import uuid

from core.paths import data_dir
from celery_app import celery_app

logger = logging.getLogger(__name__)


def _db_update_tts_job(
    *,
    job_id: str,
    status: str,
    error: str | None = None,
    result: dict | None = None,
    audio_duration_seconds: float | None = None,
) -> None:
    try:
        from db import TTSJob, ensure_schema, session_scope

        ensure_schema()
        with session_scope() as session:
            row = session.get(TTSJob, uuid.UUID(job_id))
            if row is None:
                return
            row.status = status
            row.error = error
            row.result = result
            if audio_duration_seconds is not None:
                row.audio_duration_seconds = float(audio_duration_seconds)
    except Exception:
        logger.exception("DB update crashed for job_id=%s status=%s", job_id, status)


def _s3_client():
    import os

    import boto3
    from botocore.config import Config

    def _require_env(key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise RuntimeError(f"Missing required env var: {key}")
        return value

    endpoint = _require_env("S3_ENDPOINT")
    access_key = _require_env("S3_ACCESS_KEY")
    secret_key = _require_env("S3_SECRET_KEY")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name=os.getenv("S3_REGION") or "us-east-1",
    )


def _audio_duration_seconds(path: Path) -> float | None:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        try:
            import wave

            with wave.open(str(path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
            if rate <= 0:
                return None
            return float(frames) / float(rate)
        except Exception:
            return None

    try:
        import librosa

        return float(librosa.get_duration(path=str(path)))
    except Exception:
        return None


@celery_app.task(name="tts.synthesize", bind=True)
def synthesize_tts(  # type: ignore[misc]
    self,
    *,
    job_id: str,
    text: str,
    provider: str,
    output_bucket: str,
    output_key: str,
    prompt_text: str | None = None,
    prompt_audio_bucket: str | None = None,
    prompt_audio_key: str | None = None,
    sample_rate: int = 24000,
    **config: Any,
) -> dict:
    """Run TTS synthesis."""
    job_dir = (data_dir() / "tts" / "jobs" / job_id).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    output_path = (job_dir / "output").with_suffix(Path(output_key).suffix or ".wav")
    prompt_wav_path: Path | None = None

    _db_update_tts_job(job_id=job_id, status="STARTED", error=None, result=None)

    try:
        s3 = _s3_client()

        if provider in {"glm_tts", "glm-tts", "glmtts"}:
            if not prompt_text or not prompt_text.strip():
                raise ValueError("glm_tts requires prompt_text")
            if not prompt_audio_bucket or not prompt_audio_key:
                raise ValueError("glm_tts requires prompt_audio_bucket/prompt_audio_key")
            prompt_wav_path = (job_dir / "prompt").with_suffix(
                Path(prompt_audio_key).suffix or ".wav"
            )
            s3.download_file(prompt_audio_bucket, prompt_audio_key, str(prompt_wav_path))

        from tts import TTSError, get_tts_provider

        tts_provider = get_tts_provider(provider, sample_rate=int(sample_rate), **config)

        synth_kwargs: dict[str, Any] = {}
        if prompt_wav_path is not None:
            synth_kwargs["prompt_wav"] = str(prompt_wav_path)
            synth_kwargs["prompt_text"] = prompt_text

        try:
            asyncio.run(tts_provider.synthesize(text=text, output_path=output_path, **synth_kwargs))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    tts_provider.synthesize(text=text, output_path=output_path, **synth_kwargs)
                )
            finally:
                loop.close()

        if not output_path.is_file():
            raise TTSError(f"Output file not created: {output_path}")

        s3.upload_file(str(output_path), output_bucket, output_key)
        duration = _audio_duration_seconds(output_path)
        result = {
            "job_id": job_id,
            "provider": provider,
            "output_bucket": output_bucket,
            "output_key": output_key,
            "audio_duration_seconds": duration,
        }
        _db_update_tts_job(
            job_id=job_id,
            status="SUCCEEDED",
            error=None,
            result=result,
            audio_duration_seconds=duration,
        )
        return result

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("TTS synthesis failed (job_id=%s): %s\n%s", job_id, exc, tb)
        _db_update_tts_job(job_id=job_id, status="FAILED", error=str(exc), result=None)
        raise

