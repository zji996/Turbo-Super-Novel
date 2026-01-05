from __future__ import annotations

from functools import wraps
from typing import Awaitable, Callable, TypeVar

import httpx
from fastapi import HTTPException

T = TypeVar("T")


def handle_remote_errors(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Decorator to handle remote API call errors uniformly."""

    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        try:
            return await func(*args, **kwargs)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=int(exc.response.status_code),
                detail=(exc.response.text or str(exc)),
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return wrapper

