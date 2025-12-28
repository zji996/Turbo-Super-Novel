"""Base classes for image generation providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ImageGenError(RuntimeError):
    """Base exception for image generation operations."""
    pass


@dataclass
class ModelInfo:
    """Information about an available model."""
    model_id: str
    name: str
    type: str  # e.g., "checkpoint", "lora", "embedding"
    hash: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "type": self.type,
            "hash": self.hash,
        }


class ImageGenProvider(ABC):
    """Abstract base class for image generation providers.
    
    All image generation implementations must inherit from this class
    and implement the required methods.
    """
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
    
    @abstractmethod
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
        """Generate an image from a text prompt.
        
        Args:
            prompt: The positive prompt describing the image.
            output_path: Path where the image will be saved.
            negative_prompt: Things to avoid in the image.
            width: Image width in pixels.
            height: Image height in pixels.
            steps: Number of generation steps.
            cfg_scale: Classifier-free guidance scale.
            seed: Random seed (-1 for random).
            sampler: Sampler name (provider-specific).
            model: Model to use (provider-specific).
            **kwargs: Provider-specific options.
        
        Returns:
            Path to the generated image file.
        
        Raises:
            ImageGenError: If generation fails.
        """
        pass
    
    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """List available models.
        
        Returns:
            List of available models.
        """
        pass
    
    async def get_current_model(self) -> str | None:
        """Get the currently loaded model.
        
        Returns:
            Model name or None if not applicable.
        """
        return None
    
    async def set_model(self, model: str) -> None:
        """Set the model to use for generation.
        
        Args:
            model: Model name or ID.
        
        Raises:
            ImageGenError: If model cannot be set.
        """
        raise ImageGenError("Model switching not supported by this provider")
