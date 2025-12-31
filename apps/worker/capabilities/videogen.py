"""Video generation capability tasks (GPU)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import traceback
from typing import Any
import uuid

from celery_app import celery_app
from core.paths import data_dir
from model_manager import ModelManager
from s3_utils import get_s3_client
from videogen.inference import run_wan22_i2v
from videogen.paths import wan22_i2v_model_paths

from .base import CapabilityWorker

logger = logging.getLogger(__name__)


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _db_update(
    *, job_id: str, status: str, error: str | None = None, result: dict | None = None
) -> None:
    try:
        from db import ensure_schema, try_update_job

        ensure_schema()
        ok, err = try_update_job(job_id, status=status, error=error, result=result)
        if not ok:
            logger.warning("DB update failed for job_id=%s status=%s: %s", job_id, status, err)
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
        from db import VideoGenJob, ensure_schema, session_scope

        ensure_schema()
        with session_scope() as session:
            row = session.get(VideoGenJob, job_uuid)
            if row is None:
                session.add(
                    VideoGenJob(
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


class VideoGenWorker(CapabilityWorker):
    name = "videogen"
    requires_gpu = True

    def __init__(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_models(self) -> None:
        # Current videogen implementation loads models inside inference call.
        # Marking loaded enables capability switching semantics (unload on switch/ondemand).
        self._loaded = True

    def unload_models(self) -> None:
        self._loaded = False

    def execute(self, job: dict) -> dict[str, Any]:
        kind = str(job.get("kind", "") or "")
        if kind != "cap.videogen.generate":
            raise ValueError(f"Unknown VideoGen job kind: {kind}")

        job_id: str = str(job["job_id"])
        input_bucket: str = str(job["input_bucket"])
        input_key: str = str(job["input_key"])
        output_bucket: str = str(job["output_bucket"])
        output_key: str = str(job["output_key"])
        prompt: str = str(job["prompt"])
        seed: int = int(job.get("seed", 0))
        num_steps: int = int(job.get("num_steps", 4))
        quantized: bool = bool(job.get("quantized", True))
        duration_seconds: float | None = job.get("duration_seconds")

        s3 = get_s3_client()
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
                    f"duration_seconds too small (computed num_frames={num_frames} with fps={fps}); "
                    f"try duration_seconds>={min_duration:.3f}"
                )
            if num_frames == 2:
                min_duration = 2.5 / fps
                raise ValueError(
                    "duration_seconds too small for stable VAE encoding "
                    f"(computed num_frames=2 with fps={fps}); try duration_seconds>={min_duration:.3f} "
                    "or increase TD_VIDEO_FPS"
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
        attention_type = str(job.get("attention_type", "sagesla") or "sagesla")
        sla_topk = float(job.get("sla_topk", 0.1))
        ode = bool(job.get("ode", True))
        adaptive_resolution = bool(job.get("adaptive_resolution", True))
        resolution = str(job.get("resolution", "720p") or "720p")
        aspect_ratio = str(job.get("aspect_ratio", "16:9") or "16:9")
        boundary = float(job.get("boundary", 0.9))
        sigma_max = float(job.get("sigma_max", 200.0))

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
                "duration_seconds": float(duration_seconds) if duration_seconds is not None else None,
            },
        )

        return {
            "job_id": job_id,
            "input_bucket": input_bucket,
            "input_key": input_key,
            "output_bucket": output_bucket,
            "output_key": output_key,
            "duration_seconds": float(duration_seconds) if duration_seconds is not None else None,
        }


_videogen_worker = VideoGenWorker()
ModelManager.register(_videogen_worker)


@celery_app.task(name="cap.videogen.generate", bind=True)
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
    **extra: Any,
) -> dict[str, Any]:
    worker = ModelManager.acquire("videogen")
    try:
        return worker.execute(
            {
                "kind": "cap.videogen.generate",
                "job_id": job_id,
                "input_bucket": input_bucket,
                "input_key": input_key,
                "output_bucket": output_bucket,
                "output_key": output_key,
                "prompt": prompt,
                "seed": int(seed),
                "num_steps": int(num_steps),
                "quantized": bool(quantized),
                "duration_seconds": float(duration_seconds) if duration_seconds is not None else None,
                "attention_type": attention_type,
                "sla_topk": float(sla_topk),
                "ode": bool(ode),
                "adaptive_resolution": bool(adaptive_resolution),
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "boundary": float(boundary),
                "sigma_max": float(sigma_max),
                **dict(extra),
            }
        )
    finally:
        ModelManager.release("videogen")
