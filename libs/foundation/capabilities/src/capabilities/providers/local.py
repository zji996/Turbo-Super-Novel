from __future__ import annotations

from typing import Any
from uuid import uuid4

from capabilities.providers.base import BaseProvider


class LocalProvider(BaseProvider):
    """Route jobs to local Celery worker."""

    provider_type = "local"

    def _get_celery_app(self):
        from celery import Celery
        import os

        broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
        result_backend = os.getenv("CELERY_RESULT_BACKEND", broker_url)
        app = Celery("tsn-capabilities", broker=broker_url, backend=result_backend)
        app.conf.update(
            task_serializer="json",
            result_serializer="json",
            accept_content=["json"],
        )
        return app

    async def submit_tts_job(self, *, text: str, provider: str, **kwargs: Any) -> dict:
        app = self._get_celery_app()
        job_id = str(kwargs.pop("job_id", "") or uuid4())
        app.send_task(
            "cap.tts.synthesize",
            task_id=job_id,
            queue="cap.tts",
            kwargs={"job_id": job_id, "text": text, "provider": provider, **kwargs},
        )
        return {"job_id": job_id, "status": "submitted", "provider_type": "local"}

    async def get_tts_job(self, job_id: str) -> dict:
        from celery.result import AsyncResult

        app = self._get_celery_app()
        result: AsyncResult = app.AsyncResult(job_id)
        payload: dict[str, Any] = {"job_id": job_id, "celery_status": result.status}

        if result.successful():
            try:
                payload["result"] = result.get(timeout=0)
            except Exception as exc:
                payload["result_error"] = str(exc)
        elif result.failed():
            try:
                payload["error"] = str(result.result)
            except Exception:
                payload["error"] = "unknown"

        return payload

    async def submit_imagegen_job(self, *, prompt: str, **kwargs: Any) -> dict:
        raise NotImplementedError("imagegen local provider is not implemented yet")

    async def get_imagegen_job(self, job_id: str) -> dict:
        raise NotImplementedError("imagegen local provider is not implemented yet")

    async def submit_videogen_job(
        self, *, prompt: str, image_path: str, **kwargs: Any
    ) -> dict:
        """Submit video generation job to local Celery worker."""
        app = self._get_celery_app()
        job_id = str(kwargs.pop("job_id", "") or uuid4())

        input_bucket = kwargs.pop("input_bucket", None)
        input_key = kwargs.pop("input_key", None)
        output_bucket = kwargs.pop("output_bucket", None)
        output_key = kwargs.pop("output_key", None)

        if not all([input_bucket, input_key, output_bucket, output_key]):
            raise ValueError(
                "videogen requires input_bucket, input_key, output_bucket, output_key"
            )

        app.send_task(
            "cap.videogen.generate",
            task_id=job_id,
            queue="cap.videogen",
            kwargs={
                "job_id": job_id,
                "prompt": prompt,
                "input_bucket": input_bucket,
                "input_key": input_key,
                "output_bucket": output_bucket,
                "output_key": output_key,
                **kwargs,
            },
        )
        return {"job_id": job_id, "status": "submitted", "provider_type": "local"}

    async def get_videogen_job(self, job_id: str) -> dict:
        """Get video generation job status from Celery."""
        from celery.result import AsyncResult

        app = self._get_celery_app()
        result: AsyncResult = app.AsyncResult(job_id)
        payload: dict[str, Any] = {"job_id": job_id, "celery_status": result.status}

        if result.successful():
            try:
                payload["result"] = result.get(timeout=0)
            except Exception as exc:
                payload["result_error"] = str(exc)
        elif result.failed():
            try:
                payload["error"] = str(result.result)
            except Exception:
                payload["error"] = "unknown"

        return payload
