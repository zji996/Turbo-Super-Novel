from __future__ import annotations

import logging
import os
from pathlib import Path
import traceback
from typing import Any
import uuid

import boto3
from botocore.config import Config

from core.paths import data_dir
from videogen.inference import run_wan22_i2v
from videogen.paths import wan22_i2v_model_paths
from celery_app import celery_app

logger = logging.getLogger(__name__)


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return value


def _s3_client():
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


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _gpu_mode_defaults() -> tuple[bool, bool]:
    """
    Return (resident_gpu_default, cuda_cleanup_default) derived from GPU_MODE.

    GPU_MODE is expanded to TD_* vars by `scripts/tsn_manage.sh`, but when the worker is started
    directly (or via IDE envFile) only GPU_MODE may be present.
    """
    mode = str(os.getenv("GPU_MODE", "")).strip().lower()
    if mode == "fast":
        return True, False
    if mode == "balanced":
        return False, True
    if mode == "lowvram":
        return False, True
    return False, True


def _maybe_cuda_cleanup(*, job_id: str) -> None:
    resident_default, cleanup_default = _gpu_mode_defaults()
    resident_gpu = _env_bool("TD_RESIDENT_GPU", resident_default)
    if not _env_bool("TD_CUDA_CLEANUP", cleanup_default):
        return
    if resident_gpu:
        return
    try:
        import gc

        import torch
    except Exception:
        return

    if not torch.cuda.is_available():
        return

    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        logger.exception("CUDA cleanup failed (job_id=%s)", job_id)


def _db_update(
    *, job_id: str, status: str, error: str | None = None, result: dict | None = None
) -> None:
    try:
        from db import ensure_schema, try_update_job

        ensure_schema()
        ok, err = try_update_job(job_id, status=status, error=error, result=result)
        if not ok:
            logger.warning(
                "DB update failed for job_id=%s status=%s: %s", job_id, status, err
            )
    except Exception:
        logger.exception("DB update crashed for job_id=%s status=%s", job_id, status)


def _db_ensure_job_row(
    *,
    job_id: str,
    status: str,
    prompt: str,
    seed: int,
    num_steps: int,
    quantized: bool,
    input_bucket: str,
    input_key: str,
    output_bucket: str,
    output_key: str,
) -> None:
    try:
        job_uuid = uuid.UUID(job_id)
        from db import TurboDiffusionJob, ensure_schema, session_scope

        ensure_schema()
        with session_scope() as session:
            row = session.get(TurboDiffusionJob, job_uuid)
            if row is None:
                session.add(
                    TurboDiffusionJob(
                        id=job_uuid,
                        status=status,
                        prompt=prompt,
                        seed=int(seed),
                        num_steps=int(num_steps),
                        quantized=bool(quantized),
                        input_bucket=input_bucket,
                        input_key=input_key,
                        output_bucket=output_bucket,
                        output_key=output_key,
                    )
                )
    except Exception:
        logger.exception("DB ensure job row failed for job_id=%s", job_id)


@celery_app.task(name="turbodiffusion.wan22_i2v.generate", bind=True)
def generate_wan22_i2v(  # type: ignore[misc]
    self,
    *,
    job_id: str,
    input_bucket: str,
    input_key: str,
    output_bucket: str,
    output_key: str,
    prompt: str,
    seed: int = 0,
    num_steps: int = 4,
    quantized: bool = True,
    duration_seconds: float | None = None,
    attention_type: str = "sagesla",
    sla_topk: float = 0.1,
    ode: bool = True,
    adaptive_resolution: bool = True,
    resolution: str = "720p",
    aspect_ratio: str = "16:9",
    boundary: float = 0.9,
    sigma_max: float = 200.0,
) -> dict[str, Any]:
    s3 = _s3_client()
    job_dir = (data_dir() / "turbodiffusion" / "jobs" / job_id).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)

    fps = float(os.getenv("TD_VIDEO_FPS", "16") or "16")
    if fps <= 0:
        raise ValueError(f"TD_VIDEO_FPS must be > 0, got {fps}")

    num_frames: int | None = None
    if duration_seconds is not None:
        if float(duration_seconds) <= 0:
            raise ValueError(f"duration_seconds must be > 0, got {duration_seconds}")
        max_duration = float(os.getenv("TD_MAX_VIDEO_SECONDS", "10") or "10")
        if float(duration_seconds) > max_duration:
            raise ValueError(
                f"duration_seconds too large (max={max_duration}s), got {duration_seconds}"
            )

        num_frames = int(round(float(duration_seconds) * fps))
        if num_frames <= 0:
            min_duration = 0.5 / fps
            raise ValueError(
                f"duration_seconds too small (computed num_frames={num_frames} with fps={fps}); try duration_seconds>={min_duration:.3f}"
            )
        if num_frames == 2:
            min_duration = 2.5 / fps
            raise ValueError(
                "duration_seconds too small for stable VAE encoding "
                f"(computed num_frames=2 with fps={fps}); try duration_seconds>={min_duration:.3f} or increase TD_VIDEO_FPS"
            )

    _db_ensure_job_row(
        job_id=job_id,
        status="STARTED",
        prompt=prompt,
        seed=seed,
        num_steps=num_steps,
        quantized=quantized,
        input_bucket=input_bucket,
        input_key=input_key,
        output_bucket=output_bucket,
        output_key=output_key,
    )
    _db_update(job_id=job_id, status="STARTED", error=None, result=None)

    input_path = (job_dir / "input").with_suffix(Path(input_key).suffix or ".jpg")
    output_path = (job_dir / "output.mp4").resolve()

    s3.download_file(input_bucket, input_key, str(input_path))
    _db_update(job_id=job_id, status="DOWNLOADED", error=None, result=None)

    model_paths = wan22_i2v_model_paths(quantized=quantized)
    try:
        _db_update(job_id=job_id, status="RUNNING", error=None, result=None)

        run_wan22_i2v(
            image_path=input_path,
            prompt=prompt,
            output_path=output_path,
            model_paths=model_paths,
            num_frames=(num_frames or 77),
            fps=fps,
            seed=seed,
            num_steps=num_steps,
            attention_type=attention_type,
            sla_topk=sla_topk,
            ode=ode,
            adaptive_resolution=adaptive_resolution,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            boundary=boundary,
            sigma_max=sigma_max,
            quant_linear=quantized,
        )
    except Exception as exc:
        tb = "".join(traceback.format_exception(exc))
        (job_dir / "error.txt").write_text(tb, encoding="utf-8", errors="replace")
        _db_update(job_id=job_id, status="FAILED", error=str(exc), result=None)
        raise
    finally:
        _maybe_cuda_cleanup(job_id=job_id)

    s3.upload_file(
        str(output_path),
        output_bucket,
        output_key,
        ExtraArgs={"ContentType": "video/mp4"},
    )
    _db_update(job_id=job_id, status="UPLOADED", error=None, result=None)

    _db_update(
        job_id=job_id,
        status="SUCCEEDED",
        error=None,
        result={
            "output_bucket": output_bucket,
            "output_key": output_key,
            "duration_seconds": float(duration_seconds)
            if duration_seconds is not None
            else None,
        },
    )

    return {
        "job_id": job_id,
        "input_bucket": input_bucket,
        "input_key": input_key,
        "output_bucket": output_bucket,
        "output_key": output_key,
        "duration_seconds": float(duration_seconds)
        if duration_seconds is not None
        else None,
    }
