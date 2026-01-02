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

    @staticmethod
    def _first_message_content(payload: dict[str, Any]) -> str | None:
        try:
            choices = payload.get("choices") or []
            if not choices:
                return None
            message = choices[0].get("message") or {}
            content = message.get("content")
            if content is None:
                return None
            text = str(content).strip()
            return text or None
        except Exception:
            return None

    async def enhance_prompt(self, prompt: str, context: str) -> str:
        """Enhance user prompt/text with context-specific guidance."""
        system_prompts = {
            "image_generation": (
                "你是图像提示词优化助手。将用户中文/英文描述改写为更适合图像生成的提示词："
                "更具体、包含主体/场景/光照/镜头/风格等要素；保持简洁；仅输出优化后的提示词。"
            ),
            "video_generation": (
                "你是视频提示词优化助手。将用户描述改写为更适合图生视频(I2V)的提示词："
                "补充动作、镜头运动、节奏、风格；避免冗长；仅输出优化后的提示词。"
            ),
            "tts_text": (
                "你是中文朗读文本优化助手。对用户输入做适合 TTS 的轻量优化："
                "必要时添加合理的标点/分句以提升停顿与韵律；不改变原意；仅输出优化后的文本。"
            ),
        }

        result = await self.chat(
            [
                {"role": "system", "content": system_prompts.get(context, "")},
                {"role": "user", "content": f"请优化以下内容：\n{prompt}"},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        content = self._first_message_content(result)
        return content or prompt

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
