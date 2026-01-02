from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from capabilities import get_capability_router

router = APIRouter(prefix="/v1/llm", tags=["llm"])


class ChatRequest(BaseModel):
    messages: list[dict] = Field(...)
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = Field(default=4096, ge=1, le=32768)


class GenerateScriptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)


class OptimizePromptRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    target: str = Field(default="image")


@router.post("/chat")
async def chat(request: ChatRequest) -> dict:
    try:
        provider = get_capability_router("llm")
        return await provider.chat(  # type: ignore[attr-defined]
            request.messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/generate-script")
async def generate_script(request: GenerateScriptRequest) -> dict:
    try:
        provider = get_capability_router("llm")
        script = await provider.generate_script(request.prompt)  # type: ignore[attr-defined]
        return {"script": script}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/optimize-prompt")
async def optimize_prompt(request: OptimizePromptRequest) -> dict:
    try:
        provider = get_capability_router("llm")
        optimized = await provider.optimize_image_prompt(request.text)  # type: ignore[attr-defined]
        return {"original": request.text, "optimized": optimized}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
