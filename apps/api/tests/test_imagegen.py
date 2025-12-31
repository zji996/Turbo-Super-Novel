"""Tests for imagegen API routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestImageGenRoutes:
    """Tests for /v1/imagegen endpoints."""

    def test_create_job_missing_prompt(self, client: TestClient):
        """Test job creation with missing prompt."""

        response = client.post("/v1/imagegen/jobs", json={})
        assert response.status_code == 422

    def test_create_job_prompt_too_long(self, client: TestClient):
        """Test job creation with prompt exceeding max length."""

        response = client.post("/v1/imagegen/jobs", json={"prompt": "x" * 10000})
        assert response.status_code == 422

