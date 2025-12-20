from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

from libs.pycore.paths import data_dir
from libs.turbodiffusion.inference import run_wan22_i2v
from libs.turbodiffusion.paths import wan22_i2v_model_paths

from celery_app import celery_app


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
    attention_type: str = "sla",
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

    input_path = (job_dir / "input").with_suffix(Path(input_key).suffix or ".jpg")
    output_path = (job_dir / "output.mp4").resolve()

    s3.download_file(input_bucket, input_key, str(input_path))

    model_paths = wan22_i2v_model_paths(quantized=quantized)
    try:
        run_wan22_i2v(
            image_path=input_path,
            prompt=prompt,
            output_path=output_path,
            model_paths=model_paths,
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
        try:
            import uuid

            from libs.dbcore import TurboDiffusionJob, session_scope

            with session_scope() as session:
                row = session.get(TurboDiffusionJob, uuid.UUID(job_id))
                if row is not None:
                    row.status = "FAILED"
                    row.error = str(exc)
        except Exception:
            pass
        raise

    s3.upload_file(
        str(output_path),
        output_bucket,
        output_key,
        ExtraArgs={"ContentType": "video/mp4"},
    )

    try:
        import uuid

        from libs.dbcore import TurboDiffusionJob, session_scope

        with session_scope() as session:
            row = session.get(TurboDiffusionJob, uuid.UUID(job_id))
            if row is not None:
                row.status = "SUCCEEDED"
                row.result = {"output_bucket": output_bucket, "output_key": output_key}
                row.error = None
    except Exception:
        pass

    return {
        "job_id": job_id,
        "input_bucket": input_bucket,
        "input_key": input_key,
        "output_bucket": output_bucket,
        "output_key": output_key,
    }
