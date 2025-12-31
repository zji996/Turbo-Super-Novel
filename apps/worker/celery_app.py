from __future__ import annotations

import os

from celery import Celery
from kombu import Queue


def _parse_capabilities() -> list[str]:
    raw = os.getenv("WORKER_CAPABILITIES", "tts,videogen")
    capabilities = [c.strip() for c in str(raw).split(",") if c.strip()]
    return capabilities or ["tts", "videogen"]


def _capability_includes(capabilities: list[str]) -> list[str]:
    mapping = {
        "tts": "capabilities.tts",
        "videogen": "capabilities.videogen",
    }
    return [mapping[c] for c in capabilities if c in mapping]


def make_celery() -> Celery:
    broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    result_backend = os.getenv("CELERY_RESULT_BACKEND", broker_url)
    capabilities = _parse_capabilities()

    app = Celery(
        "tsn-worker",
        broker=broker_url,
        backend=result_backend,
        include=_capability_includes(capabilities),
    )

    task_queues: list[Queue] = [Queue("celery")]
    for cap in capabilities:
        task_queues.append(Queue(f"cap.{cap}"))

    routes: dict[str, dict[str, str]] = {}
    if "tts" in capabilities:
        routes.update(
            {
                "cap.tts.synthesize": {"queue": "cap.tts"},
                "cap.tts.scene": {"queue": "cap.tts"},
            }
        )
    if "videogen" in capabilities:
        routes["cap.videogen.generate"] = {"queue": "cap.videogen"}

    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
        worker_hijack_root_logger=False,
        worker_prefetch_multiplier=1,
        worker_concurrency=1,
        task_queues=task_queues,
        task_routes=routes,
    )
    return app


celery_app = make_celery()
