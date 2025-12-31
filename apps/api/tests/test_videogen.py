"""Tests for videogen API routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestVideoGenRoutes:
    """Tests for /v1/videogen endpoints."""

    def test_get_models(self, client: TestClient):
        """Test listing available models."""
        response = client.get("/v1/videogen/models")
        assert response.status_code == 200

    def test_create_job_missing_image(self, client: TestClient):
        """Test job creation with missing image."""
        response = client.post(
            "/v1/videogen/wan22-i2v/jobs",
            data={"prompt": "test prompt"},
        )
        assert response.status_code == 422

    def test_get_nonexistent_job(self, client: TestClient):
        """Test getting a non-existent job."""
        response = client.get("/v1/videogen/jobs/nonexistent-id")
        assert response.status_code == 200

    def test_legacy_turbodiffusion_404(self, client: TestClient):
        """Ensure legacy /v1/turbodiffusion routes are removed."""
        response = client.get("/v1/turbodiffusion/models")
        assert response.status_code == 404

