from __future__ import annotations

import os
from typing import Any

import httpx


class LLMProvider:
    """LLM capability provider (OpenAI-compatible API)."""

    provider_type = "remote"

    def __init__(self, *, base_url: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or (os.getenv("LLM_API_KEY") or "").strip() or None
        self.model = (os.getenv("LLM_MODEL") or "deepseek-chat").strip() or "deepseek-chat"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": model or self.model,
                    "messages": messages,
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                },
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json()

    async def generate_script(self, user_prompt: str) -> str:
        system = (
            "你是专业的视频脚本编剧。根据用户描述，生成详细的视频脚本。"
            "每个场景用 --- 分隔，格式：\n"
            "[场景描述]\n旁白文本\n---"
        )
        result = await self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ]
        )
        return str(result["choices"][0]["message"]["content"])

    async def optimize_image_prompt(self, text: str) -> str:
        system = (
            "将用户描述转换为适合 AI 图像生成的英文 prompt。"
            "输出格式：简洁、描述性强、包含风格关键词。"
        )
        result = await self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ]
        )
        return str(result["choices"][0]["message"]["content"])

