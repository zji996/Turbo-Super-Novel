"""Edge TTS provider implementation.

Uses Microsoft Edge's online TTS service via the edge-tts library.
Free to use, good quality voices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import TTSProvider, TTSError, VoiceInfo


class EdgeTTSProvider(TTSProvider):
    """TTS provider using Microsoft Edge's online TTS service.

    This is a free service that provides high-quality neural voices.
    Requires the `edge-tts` package.
    """

    def __init__(self, **kwargs: Any):
        """Initialize the Edge TTS provider."""
        try:
            import edge_tts

            self._edge_tts = edge_tts
        except ImportError as e:
            raise TTSError(
                "edge-tts package not installed. Install with: pip install edge-tts"
            ) from e

    @property
    def provider_name(self) -> str:
        return "edge"

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
        """Synthesize text to audio using Edge TTS.

        Args:
            text: The text to synthesize.
            output_path: Path where the audio file will be saved.
            voice: Voice ID (e.g., "zh-CN-XiaoxiaoNeural").
            rate: Speech rate (e.g., "+20%", "-10%").
            volume: Volume level (e.g., "+50%", "-20%").
            pitch: Pitch adjustment (e.g., "+5Hz", "-10Hz").

        Returns:
            Path to the generated audio file (MP3 format).
        """
        if not text or not text.strip():
            raise TTSError("Text cannot be empty")

        voice = voice or self.get_default_voice()

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build communicate instance
        communicate = self._edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            volume=volume,
            pitch=pitch,
        )

        try:
            await communicate.save(str(output_path))
        except Exception as e:
            raise TTSError(f"Edge TTS synthesis failed: {e}") from e

        if not output_path.is_file():
            raise TTSError(f"Output file not created: {output_path}")

        return output_path

    async def list_voices(
        self,
        *,
        language: str | None = None,
    ) -> list[VoiceInfo]:
        """List available Edge TTS voices.

        Args:
            language: Filter by language code (e.g., "zh-CN", "en-US").

        Returns:
            List of available voices.
        """
        try:
            voices_data = await self._edge_tts.list_voices()
        except Exception as e:
            raise TTSError(f"Failed to list voices: {e}") from e

        voices: list[VoiceInfo] = []
        for v in voices_data:
            voice_locale = v.get("Locale", "")

            # Filter by language if specified
            if language and not voice_locale.startswith(language):
                continue

            voices.append(
                VoiceInfo(
                    voice_id=v.get("ShortName", ""),
                    name=v.get("FriendlyName", v.get("ShortName", "")),
                    language=voice_locale,
                    gender=v.get("Gender"),
                    style=v.get("VoiceTag", {}).get("VoicePersonalities", [None])[0]
                    if v.get("VoiceTag")
                    else None,
                )
            )

        return voices
