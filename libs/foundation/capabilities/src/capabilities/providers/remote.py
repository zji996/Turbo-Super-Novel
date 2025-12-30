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
            headers["Authorization"] = f"Bearer {self.api_key}"
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

    async def submit_imagegen_job(self, *, prompt: str, **kwargs: Any) -> dict:
        raise NotImplementedError("imagegen remote provider is not implemented yet")

    async def get_imagegen_job(self, job_id: str) -> dict:
        raise NotImplementedError("imagegen remote provider is not implemented yet")

    async def submit_videogen_job(self, *, prompt: str, image_path: str, **kwargs: Any) -> dict:
        raise NotImplementedError("videogen remote provider is not implemented yet")

    async def get_videogen_job(self, job_id: str) -> dict:
        raise NotImplementedError("videogen remote provider is not implemented yet")

