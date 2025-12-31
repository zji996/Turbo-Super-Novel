from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from main import app


def test_create_speaker_profile_requires_wav() -> None:
    client = TestClient(app)
    resp = client.post(
        "/v1/tts/speaker-profiles",
        data={"name": "test", "prompt_text": "hello"},
        files={"prompt_audio": ("ref.mp3", b"not-wav", "audio/mpeg")},
    )
    assert resp.status_code == 422


def test_get_speaker_profile_invalid_id() -> None:
    client = TestClient(app)
    resp = client.get("/v1/tts/speaker-profiles/not-a-uuid")
    assert resp.status_code == 400


def test_create_tts_job_with_profile_missing_fields() -> None:
    client = TestClient(app)
    resp = client.post("/v1/tts/jobs/with-profile", json={})
    assert resp.status_code == 422


def test_create_tts_job_with_profile_invalid_id() -> None:
    client = TestClient(app)
    resp = client.post(
        "/v1/tts/jobs/with-profile",
        json={"text": "hi", "profile_id": "not-a-uuid"},
    )
    assert resp.status_code == 400


def test_create_tts_job_with_profile_requires_text() -> None:
    client = TestClient(app)
    resp = client.post(
        "/v1/tts/jobs/with-profile",
        json={"profile_id": str(uuid.uuid4())},
    )
    assert resp.status_code == 422

