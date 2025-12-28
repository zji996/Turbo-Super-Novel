"""ttscore - TTS abstraction layer with pluggable providers."""

from .base import TTSProvider, TTSError, VoiceInfo
from .service import get_tts_provider, list_available_providers

__all__ = [
    "TTSProvider",
    "TTSError",
    "VoiceInfo",
    "get_tts_provider",
    "list_available_providers",
]
