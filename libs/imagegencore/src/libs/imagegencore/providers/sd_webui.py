"""Stable Diffusion WebUI API provider.

Connects to a running SD WebUI instance via its REST API.
Supports both AUTOMATIC1111 and Forge WebUI.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import httpx

from ..base import ImageGenProvider, ImageGenError, ModelInfo


class SDWebUIProvider(ImageGenProvider):
    """Image generation provider using SD WebUI API.
    
    Connects to a running Stable Diffusion WebUI instance
    (AUTOMATIC1111 or Forge).
    """
    
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 300.0,
        **kwargs: Any,
    ):
        """Initialize the SD WebUI provider.
        
        Args:
            base_url: Base URL of the SD WebUI instance.
                      Defaults to IMAGEGEN_SD_WEBUI_URL env var or http://localhost:7860.
            timeout: Request timeout in seconds.
        """
        self.base_url = (
            base_url 
            or os.getenv("IMAGEGEN_SD_WEBUI_URL") 
            or "http://localhost:7860"
        ).rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
    
    @property
    def provider_name(self) -> str:
        return "sd_webui"
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def generate(
        self,
        prompt: str,
        output_path: Path,
        *,
        negative_prompt: str | None = None,
        width: int = 1280,
        height: int = 720,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = -1,
        sampler: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Generate an image using SD WebUI API.
        
        Args:
            prompt: The positive prompt.
            output_path: Where to save the generated image.
            negative_prompt: Negative prompt.
            width: Image width.
            height: Image height.
            steps: Sampling steps.
            cfg_scale: CFG scale.
            seed: Random seed (-1 for random).
            sampler: Sampler name (e.g., "DPM++ 2M Karras").
            model: Checkpoint to use (will switch if different).
        
        Returns:
            Path to the saved image.
        """
        if not prompt or not prompt.strip():
            raise ImageGenError("Prompt cannot be empty")
        
        client = await self._get_client()
        
        # Switch model if specified
        if model:
            await self.set_model(model)
        
        # Build request payload
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "sampler_name": sampler or "DPM++ 2M",
            "batch_size": 1,
            "n_iter": 1,
        }
        
        # Merge additional kwargs
        payload.update(kwargs)
        
        try:
            response = await client.post("/sdapi/v1/txt2img", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            raise ImageGenError(f"SD WebUI API request failed: {e}") from e
        
        # Extract and save image
        images = data.get("images", [])
        if not images:
            raise ImageGenError("No images returned from SD WebUI")
        
        # First image is the generated one (others might be grid/extras)
        image_data = base64.b64decode(images[0])
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_data)
        
        if not output_path.is_file():
            raise ImageGenError(f"Failed to save image to {output_path}")
        
        return output_path
    
    async def list_models(self) -> list[ModelInfo]:
        """List available SD models/checkpoints."""
        client = await self._get_client()
        
        try:
            response = await client.get("/sdapi/v1/sd-models")
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            raise ImageGenError(f"Failed to list models: {e}") from e
        
        models: list[ModelInfo] = []
        for m in data:
            models.append(ModelInfo(
                model_id=m.get("title", ""),
                name=m.get("model_name", m.get("title", "")),
                type="checkpoint",
                hash=m.get("hash"),
            ))
        
        return models
    
    async def get_current_model(self) -> str | None:
        """Get the currently loaded model."""
        client = await self._get_client()
        
        try:
            response = await client.get("/sdapi/v1/options")
            response.raise_for_status()
            data = response.json()
            return data.get("sd_model_checkpoint")
        except httpx.HTTPError:
            return None
    
    async def set_model(self, model: str) -> None:
        """Set the model to use.
        
        Args:
            model: Model title or filename.
        """
        current = await self.get_current_model()
        if current and model in current:
            return  # Already loaded
        
        client = await self._get_client()
        
        try:
            response = await client.post(
                "/sdapi/v1/options",
                json={"sd_model_checkpoint": model},
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise ImageGenError(f"Failed to set model: {e}") from e
