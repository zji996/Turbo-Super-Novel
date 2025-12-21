from __future__ import annotations

import logging
from pathlib import Path
import uuid
from uuid import uuid4

from celery.result import AsyncResult
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from libs.pycore.paths import paths_summary
from libs.dbcore import TurboDiffusionJob, ensure_schema, session_scope, try_insert_job, try_update_job
from libs.turbodiffusion.registry import list_artifacts
from libs.turbodiffusion.paths import turbodiffusion_models_root, wan22_i2v_model_paths

from celery_app import celery_app
from s3 import ensure_bucket_exists, s3_bucket_name, s3_client

app = FastAPI(title="Turbo-Super-Novel API", version="0.1.0")
logger = logging.getLogger(__name__)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/paths")
def paths() -> dict:
    return paths_summary()


@app.get("/v1/turbodiffusion/models")
def turbodiffusion_models() -> dict:
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
                    {"name": a.name, "group": a.group, "path": str(root.parent / a.relative_path), "exists": (root.parent / a.relative_path).is_file()}
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


@app.post("/v1/turbodiffusion/wan22-i2v/jobs")
async def create_wan22_i2v_job(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    seed: int = Form(0),
    num_steps: int = Form(4),
    quantized: bool = Form(True),
) -> dict:
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
        TurboDiffusionJob(
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
            "turbodiffusion.wan22_i2v.generate",
            task_id=job_id,
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
            },
        )
    except Exception as exc:
        if db_persisted:
            try_update_job(job_uuid, status="SUBMIT_FAILED", error=str(exc), result=None)
        raise HTTPException(status_code=500, detail=f"Failed to submit Celery task: {exc}") from exc

    if db_persisted:
        ok, err = try_update_job(job_uuid, status="SUBMITTED", error=None, result=None)
        if not ok:
            db_error = err
            logger.error("Failed to update job status to SUBMITTED: %s (%s)", job_id, db_error)

    return {
        "job_id": job_id,
        "status": "submitted",
        "input": {"bucket": bucket, "key": input_key},
        "output": {"bucket": bucket, "key": output_key},
        "db": {"persisted": db_persisted, "error": db_error},
    }


@app.get("/v1/turbodiffusion/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    result: AsyncResult = celery_app.AsyncResult(job_id)
    payload: dict = {"job_id": job_id, "status": result.status}

    try:
        ensure_schema()
        with session_scope() as session:
            row = session.get(TurboDiffusionJob, uuid.UUID(job_id))
            if row is not None:
                payload["db"] = {
                    "status": row.status,
                    "input": {"bucket": row.input_bucket, "key": row.input_key},
                    "output": {"bucket": row.output_bucket, "key": row.output_key},
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                    "error": row.error,
                }
    except Exception as exc:
        payload["db_error"] = str(exc)

    if result.successful():
        try:
            value = result.get(timeout=0)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        payload["result"] = value
        if isinstance(value, dict) and "output_bucket" in value and "output_key" in value:
            client = s3_client()
            url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": value["output_bucket"], "Key": value["output_key"]},
                ExpiresIn=3600,
            )
            payload["output_url"] = url
        return payload

    if result.failed():
        try:
            payload["error"] = str(result.result)
        except Exception:
            payload["error"] = "unknown"
        return payload

    return payload
