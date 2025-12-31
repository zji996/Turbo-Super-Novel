"""Tests for TTS API routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestTTSRoutes:
    """Tests for /v1/tts endpoints."""

    def test_list_providers(self, client: TestClient):
        """Test listing TTS providers."""

        response = client.get("/v1/tts/providers")
        assert response.status_code == 200
        assert "providers" in response.json()

    def test_create_job_missing_text(self, client: TestClient):
        """Test job creation with missing text."""

        response = client.post("/v1/tts/jobs", json={})
        assert response.status_code == 422

