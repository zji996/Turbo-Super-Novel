"""Base classes for TTS providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TTSError(RuntimeError):
    """Base exception for TTS operations."""

    pass


@dataclass
class VoiceInfo:
    """Information about an available voice."""

    voice_id: str
    name: str
    language: str
    gender: str | None = None
    style: str | None = None
    sample_rate: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "voice_id": self.voice_id,
            "name": self.name,
            "language": self.language,
            "gender": self.gender,
            "style": self.style,
            "sample_rate": self.sample_rate,
        }


class TTSProvider(ABC):
    """Abstract base class for TTS providers.

    All TTS implementations must inherit from this class and implement
    the required methods.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str | None = None,
        rate: str | None = None,
        volume: str | None = None,
        pitch: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Synthesize text to audio.

        Args:
            text: The text to synthesize.
            output_path: Path where the audio file will be saved.
            voice: Voice ID to use. If None, uses provider default.
            rate: Speech rate (e.g., "+20%", "-10%").
            volume: Volume level (e.g., "+50%", "-20%").
            pitch: Pitch adjustment (e.g., "+5Hz", "-10Hz").
            **kwargs: Provider-specific options.

        Returns:
            Path to the generated audio file.

        Raises:
            TTSError: If synthesis fails.
        """
        pass

    @abstractmethod
    async def list_voices(
        self,
        *,
        language: str | None = None,
    ) -> list[VoiceInfo]:
        """List available voices.

        Args:
            language: Filter by language code (e.g., "zh-CN", "en-US").

        Returns:
            List of available voices.
        """
        pass

    def get_default_voice(self, language: str = "zh-CN") -> str:
        """Get the default voice for a language.

        Subclasses can override this to provide language-specific defaults.
        """
        defaults = {
            "zh-CN": "zh-CN-XiaoxiaoNeural",
            "zh-TW": "zh-TW-HsiaoChenNeural",
            "en-US": "en-US-JennyNeural",
            "en-GB": "en-GB-SoniaNeural",
            "ja-JP": "ja-JP-NanamiNeural",
        }
        return defaults.get(language, "zh-CN-XiaoxiaoNeural")

    def unload(self) -> None:
        """Release any held resources (GPU weights, caches, temp dirs).

        Default implementation is a no-op for lightweight/remote providers.
        """

        return
