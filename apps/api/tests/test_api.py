from fastapi.testclient import TestClient

from main import app


def test_health() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_paths() -> None:
    client = TestClient(app)
    resp = client.get("/paths")
    assert resp.status_code == 200
    payload = resp.json()
    assert {"models_dir", "data_dir", "logs_dir"} <= set(payload.keys())
