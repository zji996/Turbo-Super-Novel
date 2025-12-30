from __future__ import annotations

from typing import Any

import httpx

from capabilities.providers.base import BaseProvider


class RemoteProvider(BaseProvider):
    """Route jobs to remote TSN API (HTTP client)."""

    provider_type = "remote"

    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.remote_url = self.base_url
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            # Support both Bearer token and X-Auth-Key (for Z-Image API)
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-Auth-Key"] = self.api_key
        return headers

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        timeout: float = 30,
    ) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(
                method.upper(),
                url,
                params=params,
                json=json,
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json()

    async def submit_tts_job(self, *, text: str, provider: str, **kwargs: Any) -> dict:
        payload: dict[str, Any] = {"text": text, "provider": provider}

        for key in ("prompt_text", "prompt_audio_id", "sample_rate", "config"):
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]

        data = await self.request_json("POST", "/v1/tts/jobs", json=payload)
        if isinstance(data, dict):
            data["provider_type"] = "remote"
        return data

    async def get_tts_job(self, job_id: str) -> dict:
        data = await self.request_json("GET", f"/v1/tts/jobs/{job_id}")
        if isinstance(data, dict):
            data.setdefault("provider_type", "remote")
        return data

    # -------------------------------------------------------------------------
    # Image Generation (Z-Image API)
    # -------------------------------------------------------------------------

    async def submit_imagegen_job(self, *, prompt: str, **kwargs: Any) -> dict:
        """Submit an image generation job to Z-Image API.

        Z-Image API endpoint: POST /v1/images/generate
        """
        payload: dict[str, Any] = {"prompt": prompt}

        # Optional parameters supported by Z-Image
        optional_keys = (
            "width",
            "height",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "negative_prompt",
            "cfg_normalization",
            "cfg_truncation",
            "max_sequence_length",
            "metadata",
        )
        for key in optional_keys:
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]

        data = await self.request_json(
            "POST", "/v1/images/generate", json=payload, timeout=120
        )

        # Normalize response format for TSN frontend
        return {
            "job_id": data.get("task_id"),
            "status": "PENDING",
            "provider_type": "remote",
            "remote_task_id": data.get("task_id"),
            "remote_status_url": data.get("status_url"),
        }

    async def get_imagegen_job(self, job_id: str) -> dict:
        """Get image generation job status from Z-Image API.

        Z-Image API endpoint: GET /v1/tasks/{task_id}
        """
        data = await self.request_json("GET", f"/v1/tasks/{job_id}", timeout=60)

        # Normalize response format for TSN frontend
        result: dict[str, Any] = {
            "job_id": job_id,
            "status": data.get("status", "PENDING"),
            "progress": data.get("progress", 0),
            "provider_type": "remote",
        }

        status = data.get("status")
        if status == "SUCCESS":
            result["image_url"] = data.get("image_url")
            result["result"] = data.get("result")
        elif status in ("FAILURE", "REVOKED"):
            result["error"] = data.get("error")
            result["error_code"] = data.get("error_code")
            result["error_hint"] = data.get("error_hint")

        return result

    async def cancel_imagegen_job(self, job_id: str) -> dict:
        """Cancel an image generation job.

        Z-Image API endpoint: POST /v1/tasks/{task_id}/cancel
        """
        data = await self.request_json(
            "POST", f"/v1/tasks/{job_id}/cancel", timeout=30
        )
        return {
            "job_id": job_id,
            "status": data.get("status", "CANCELLED"),
            "message": data.get("message", "Cancellation requested"),
            "provider_type": "remote",
        }

    async def get_imagegen_history(
        self, *, limit: int = 20, offset: int = 0
    ) -> list[dict]:
        """Get image generation history from Z-Image API.

        Z-Image API endpoint: GET /v1/history
        """
        data = await self.request_json(
            "GET",
            "/v1/history",
            params={"limit": limit, "offset": offset},
            timeout=30,
        )
        # Z-Image returns a list of TaskSummary objects
        if isinstance(data, list):
            return data
        return []

    async def submit_videogen_job(self, *, prompt: str, image_path: str, **kwargs: Any) -> dict:
        raise NotImplementedError("videogen remote provider is not implemented yet")

    async def get_videogen_job(self, job_id: str) -> dict:
        raise NotImplementedError("videogen remote provider is not implemented yet")

