"""Capabilities API routes - Individual module endpoints."""

from __future__ import annotations

import importlib
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter

from capabilities import get_capability_router
from capabilities.config import load_capability_config

router = APIRouter(prefix="/v1/capabilities", tags=["capabilities"])


async def _proxy_request_json(
    cap: object,
    method: str,
    path: str,
    *,
    params: dict | None = None,
) -> dict | None:
    if getattr(cap, "provider_type", "local") == "remote" and hasattr(cap, "request_json"):
        return await cap.request_json(method, path, params=params)  # type: ignore[attr-defined]
    return None


def _import_attr(module: str, attr: str):
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return None
    return getattr(mod, attr, None)


# ============================================================================
# TTS Endpoints
# ============================================================================


@router.get("/tts/voices")
async def list_tts_voices(
    provider: str = "glm_tts",
    language: str | None = None,
) -> dict:
    """List available TTS voices.

    Args:
        provider: TTS provider name (default: "glm_tts").
        language: Filter by language code (e.g., "zh-CN").
    """
    cap = get_capability_router("tts")
    params = {"provider": provider}
    if language is not None:
        params["language"] = language
    proxied = await _proxy_request_json(
        cap,
        "GET",
        "/v1/capabilities/tts/voices",
        params=params,
    )
    if proxied is not None:
        return proxied

    get_tts_provider = _import_attr("tts", "get_tts_provider")
    if get_tts_provider is None:
        return {
            "provider": provider,
            "voices": [],
            "warning": "tts not installed (API is lightweight; set CAP_TTS_PROVIDER=remote to proxy a full TSN API)",
        }

    try:
        tts = get_tts_provider(provider)  # type: ignore[misc]
        voices = await tts.list_voices(language=language)
        return {
            "provider": provider,
            "voices": [v.to_dict() for v in voices],
        }
    except Exception as exc:
        return {"error": str(exc), "voices": []}


@router.get("/tts/providers")
async def list_tts_providers() -> dict:
    """List available TTS providers."""
    cap = get_capability_router("tts")
    proxied = await _proxy_request_json(cap, "GET", "/v1/capabilities/tts/providers")
    if proxied is not None:
        return proxied

    list_available_providers = _import_attr("tts", "list_available_providers")
    if list_available_providers is None:
        return {
            "providers": ["glm_tts"],
            "warning": "tts not installed (API is lightweight; local mode assumes a worker can serve glm_tts)",
        }

    return {"providers": list_available_providers()}  # type: ignore[misc]


# ============================================================================
# Image Generation Endpoints
# ============================================================================


@router.get("/imagegen/models")
async def list_imagegen_models(provider: str | None = None) -> dict:
    """List available image generation models.

    Args:
        provider: Image generation provider name (default from env).
    """
    cap = get_capability_router("imagegen")
    params = {}
    if provider is not None:
        params["provider"] = provider
    proxied = await _proxy_request_json(
        cap,
        "GET",
        "/v1/capabilities/imagegen/models",
        params=params or None,
    )
    if proxied is not None:
        return proxied

    get_imagegen_provider = _import_attr("imagegen", "get_imagegen_provider")
    if get_imagegen_provider is None:
        return {
            "models": [],
            "warning": "imagegen not installed (API is lightweight; set CAP_IMAGEGEN_PROVIDER=remote to proxy a full TSN API)",
        }

    try:
        imagegen = get_imagegen_provider(provider)  # type: ignore[misc]
        models = await imagegen.list_models()
        if hasattr(imagegen, "close"):
            await imagegen.close()
        return {
            "provider": imagegen.provider_name,
            "models": [m.to_dict() for m in models],
        }
    except Exception as exc:
        return {"error": str(exc), "models": []}


@router.get("/imagegen/providers")
async def list_imagegen_providers() -> dict:
    """List available image generation providers."""
    cap = get_capability_router("imagegen")
    proxied = await _proxy_request_json(cap, "GET", "/v1/capabilities/imagegen/providers")
    if proxied is not None:
        return proxied

    list_available_providers = _import_attr("imagegen", "list_available_providers")
    if list_available_providers is None:
        return {
            "providers": [],
            "warning": "imagegen not installed (API is lightweight; set CAP_IMAGEGEN_PROVIDER=remote to proxy a full TSN API)",
        }

    return {"providers": list_available_providers()}  # type: ignore[misc]


@router.get("/imagegen/current-model")
async def get_current_imagegen_model(provider: str | None = None) -> dict:
    """Get the currently loaded model.

    Args:
        provider: Image generation provider name.
    """
    cap = get_capability_router("imagegen")
    params = {}
    if provider is not None:
        params["provider"] = provider
    proxied = await _proxy_request_json(
        cap,
        "GET",
        "/v1/capabilities/imagegen/current-model",
        params=params or None,
    )
    if proxied is not None:
        return proxied

    get_imagegen_provider = _import_attr("imagegen", "get_imagegen_provider")
    if get_imagegen_provider is None:
        return {
            "model": None,
            "warning": "imagegen not installed (API is lightweight; set CAP_IMAGEGEN_PROVIDER=remote to proxy a full TSN API)",
        }

    try:
        imagegen = get_imagegen_provider(provider)  # type: ignore[misc]
        model = await imagegen.get_current_model()
        if hasattr(imagegen, "close"):
            await imagegen.close()
        return {
            "provider": imagegen.provider_name,
            "model": model,
        }
    except Exception as exc:
        return {"error": str(exc), "model": None}


# ============================================================================
# Status Endpoints
# ============================================================================


def _auth_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}", "X-Auth-Key": api_key}


async def _http_health_check(base_url: str, api_key: str | None = None) -> bool:
    """Best-effort health check for remote endpoints."""
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return False

    parsed = urlparse(base)
    base_path = (parsed.path or "").rstrip("/")
    has_v1_suffix = base_path.endswith("/v1")

    candidates = ["health", "models"]
    if not has_v1_suffix:
        candidates.extend(["v1/health", "v1/models"])

    headers = _auth_headers(api_key)

    async with httpx.AsyncClient(timeout=5) as client:
        for candidate in candidates:
            url = f"{base}/{candidate.lstrip('/')}"
            try:
                resp = await client.get(url, headers=headers)
            except Exception:
                continue
            if 200 <= resp.status_code < 300:
                return True

    return False


def _celery_worker_available(queue: str) -> bool:
    """Check whether any worker is listening on the given queue."""
    try:
        from celery_app import celery_app

        inspect = celery_app.control.inspect(timeout=0.5)
        active_queues = inspect.active_queues() or {}
        for _worker, queues in active_queues.items():
            if any(q.get("name") == queue for q in queues):
                return True
    except Exception:
        return False
    return False


async def _probe_capability(name: str) -> dict:
    config = load_capability_config()
    endpoint = getattr(config, name, None)
    if endpoint is None:
        return {"provider": None, "status": "unavailable", "detail": "unknown capability"}

    if name == "llm" and endpoint.provider != "remote":
        return {
            "provider": endpoint.provider,
            "status": "unavailable",
            "detail": "llm only supports provider=remote",
        }

    if endpoint.provider == "remote":
        if not endpoint.remote_url:
            return {
                "provider": "remote",
                "status": "unavailable",
                "detail": "missing remote_url",
            }
        ok = await _http_health_check(endpoint.remote_url, endpoint.remote_api_key)
        return {"provider": "remote", "status": "available" if ok else "unavailable"}

    ok = _celery_worker_available(f"cap.{name}")
    return {"provider": "local", "status": "available" if ok else "unavailable"}


@router.get("/status")
async def get_capabilities_status() -> dict:
    """Return availability status for all capabilities."""
    return {
        "capabilities": {
            "tts": await _probe_capability("tts"),
            "imagegen": await _probe_capability("imagegen"),
            "videogen": await _probe_capability("videogen"),
            "llm": await _probe_capability("llm"),
        }
    }
