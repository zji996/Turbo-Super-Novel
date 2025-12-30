from fastapi.testclient import TestClient

from main import app


def test_tts_providers() -> None:
    client = TestClient(app)
    resp = client.get("/v1/tts/providers")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["providers"] == ["glm_tts"]


def test_create_tts_job_requires_prompt_for_glm() -> None:
    client = TestClient(app)
    resp = client.post("/v1/tts/jobs", json={"text": "hi", "provider": "glm_tts"})
    assert resp.status_code == 422


def test_upload_prompt_audio_requires_wav() -> None:
    client = TestClient(app)
    resp = client.post(
        "/v1/tts/prompt-audios",
        data={"name": "test", "text": "hello"},
        files={"audio": ("ref.mp3", b"not-wav", "audio/mpeg")},
    )
    assert resp.status_code == 422
