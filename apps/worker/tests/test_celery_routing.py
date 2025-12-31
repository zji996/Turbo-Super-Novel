from __future__ import annotations

import pytest


def test_celery_task_routes_use_cap_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_CAPABILITIES", "tts,videogen")

    from celery_app import make_celery

    app = make_celery()
    routes = dict(app.conf.task_routes or {})

    assert routes["cap.tts.synthesize"]["queue"] == "cap.tts"
    assert routes["cap.tts.scene"]["queue"] == "cap.tts"
    assert routes["cap.videogen.generate"]["queue"] == "cap.videogen"

