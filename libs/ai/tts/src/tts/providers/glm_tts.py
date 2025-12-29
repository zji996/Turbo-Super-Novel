"""GLM-TTS local provider (ModelScope weights + vendored GLM-TTS code).

Weights are expected under:
  <MODELS_DIR>/tts/GLM-TTS

GLM-TTS code is vendored in-repo for stability.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any

from core.paths import data_dir, models_dir
from ..base import TTSProvider, TTSError, VoiceInfo


def _default_model_dir() -> Path:
    return (models_dir() / "tts" / "GLM-TTS").resolve()


def _default_code_dir() -> Path:
    return (Path(__file__).resolve().parent.parent / "vendors" / "glm_tts").resolve()


def _write_wav_pcm16(path: Path, audio: Any, sample_rate: int) -> None:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise TTSError(f"numpy is required to write wav: {exc}") from exc

    if "torch" in type(audio).__module__:
        audio = audio.detach().cpu().numpy()

    arr = np.asarray(audio)
    if arr.ndim == 2 and arr.shape[0] in (1, 2) and arr.shape[1] > 8_000:
        arr = arr.T

    if arr.ndim == 1:
        arr = arr[:, None]

    if arr.ndim != 2:
        raise TTSError(f"Unsupported audio shape: {arr.shape}")

    if arr.dtype.kind == "f":
        arr = np.clip(arr, -1.0, 1.0)
        arr = (arr * 32767.0).astype(np.int16)
    elif arr.dtype != np.int16:
        arr = arr.astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(arr.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(arr.tobytes())


class GLMTTSProvider(TTSProvider):
    """GLM-TTS provider (GPU inference)."""

    def __init__(
        self,
        *,
        model_dir: str | Path | None = None,
        code_dir: str | Path | None = None,
        sample_rate: int = 24000,
        use_phoneme: bool = False,
        use_cache: bool = True,
        seed: int = 0,
        **_: Any,
    ) -> None:
        os.environ.setdefault("HF_HOME", str((data_dir() / "hf").resolve()))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        self._model_dir = (
            Path(model_dir).expanduser().resolve()
            if model_dir
            else _default_model_dir()
        )
        self._code_dir = (
            Path(code_dir).expanduser().resolve() if code_dir else _default_code_dir()
        )

        if not self._model_dir.exists() or not any(self._model_dir.iterdir()):
            raise TTSError(
                "GLM-TTS model files not found.\n"
                f"Expected at: {self._model_dir}\n"
                "Download with:\n"
                "  uv run --project apps/api --group tools scripts/download_glm_tts_models.py"
            )

        if not (self._code_dir / "glmtts_inference.py").is_file():
            raise TTSError(
                "GLM-TTS code not found.\n"
                f"Expected vendored snapshot at: {self._code_dir}\n"
                "If you deleted it, restore it from git."
            )

        if sample_rate not in (24000, 32000):
            raise TTSError("GLM-TTS only supports sample_rate=24000 or 32000")

        self._sample_rate = int(sample_rate)
        self._use_phoneme = bool(use_phoneme)
        self._use_cache = bool(use_cache)
        self._seed = int(seed)

        self._load_lock = threading.Lock()
        self._loaded = False
        self._runtime_dir: Path | None = None

        self._m: Any | None = None
        self._frontend: Any | None = None
        self._text_frontend: Any | None = None
        self._llm: Any | None = None
        self._flow: Any | None = None

    @property
    def provider_name(self) -> str:
        return "glm_tts"

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
        if not text or not text.strip():
            raise TTSError("Text cannot be empty")
        if any([voice, rate, volume, pitch]):
            raise TTSError(
                "GLM-TTS provider does not support voice/rate/volume/pitch parameters"
            )

        output_path = Path(output_path)
        return await asyncio.to_thread(
            self._synthesize_sync, text=text, output_path=output_path, **kwargs
        )

    def _synthesize_sync(self, *, text: str, output_path: Path, **kwargs: Any) -> Path:
        self._ensure_loaded()

        prompt_wav = kwargs.pop("prompt_wav", None)
        prompt_text = kwargs.pop("prompt_text", None)
        if prompt_wav is None or prompt_text is None:
            raise TTSError(
                "GLM-TTS requires zero-shot prompt audio: pass prompt_wav and prompt_text"
            )

        if kwargs:
            raise TTSError(f"Unknown GLM-TTS kwargs: {sorted(kwargs.keys())}")

        prompt_wav_path = Path(prompt_wav).expanduser().resolve()
        if not prompt_wav_path.is_file():
            raise TTSError(f"prompt_wav not found: {prompt_wav_path}")

        m = self._m
        assert m is not None
        frontend = self._frontend
        text_frontend = self._text_frontend
        llm = self._llm
        flow = self._flow
        assert (
            frontend is not None
            and text_frontend is not None
            and llm is not None
            and flow is not None
        )

        prompt_text_norm = text_frontend.text_normalize(str(prompt_text))
        syn_text_norm = text_frontend.text_normalize(text)
        if prompt_text_norm is None or syn_text_norm is None:
            raise TTSError("Text normalization failed (got None)")

        prompt_text_token = frontend._extract_text_token(prompt_text_norm + " ")
        prompt_speech_token = frontend._extract_speech_token([str(prompt_wav_path)])
        speech_feat = frontend._extract_speech_feat(
            str(prompt_wav_path), sample_rate=self._sample_rate
        )
        embedding = frontend._extract_spk_embedding(str(prompt_wav_path))

        cache_speech_token = [prompt_speech_token.squeeze().tolist()]
        flow_prompt_token = m.torch.tensor(cache_speech_token, dtype=m.torch.int32).to(
            m.DEVICE
        )
        cache = {
            "cache_text": [prompt_text_norm],
            "cache_text_token": [prompt_text_token],
            "cache_speech_token": cache_speech_token,
            "use_cache": self._use_cache,
        }

        tts_speech, _, _, _ = m.generate_long(
            frontend=frontend,
            text_frontend=text_frontend,
            llm=llm,
            flow=flow,
            text_info=["0", syn_text_norm],
            cache=cache,
            embedding=embedding,
            seed=self._seed,
            flow_prompt_token=flow_prompt_token,
            speech_feat=speech_feat,
            device=m.DEVICE,
            use_phoneme=self._use_phoneme,
        )

        _write_wav_pcm16(output_path, tts_speech.squeeze(0), self._sample_rate)
        if not output_path.is_file():
            raise TTSError(f"Output file not created: {output_path}")
        return output_path

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return

            runtime_dir = Path(tempfile.mkdtemp(prefix="tsn-glmtts-")).resolve()
            self._runtime_dir = runtime_dir

            ckpt_link = runtime_dir / "ckpt"
            frontend_link = runtime_dir / "frontend"
            configs_link = runtime_dir / "configs"

            for link in (ckpt_link, frontend_link, configs_link):
                if link.exists() or link.is_symlink():
                    link.unlink()

            os.symlink(self._model_dir, ckpt_link)

            model_frontend = self._model_dir / "frontend"
            if (model_frontend / "campplus.onnx").is_file():
                os.symlink(model_frontend, frontend_link)
            else:
                os.symlink(self._code_dir / "frontend", frontend_link)

            os.symlink(self._code_dir / "configs", configs_link)

            if str(self._code_dir) not in sys.path:
                sys.path.insert(0, str(self._code_dir))

            spec = importlib.util.spec_from_file_location(
                "tsn_glmtts_inference",
                self._code_dir / "glmtts_inference.py",
            )
            if spec is None or spec.loader is None:
                raise TTSError("Failed to load GLM-TTS inference module spec")
            module = importlib.util.module_from_spec(spec)

            old_cwd = Path.cwd()
            try:
                os.chdir(runtime_dir)
                spec.loader.exec_module(module)  # type: ignore[union-attr]
                frontend, text_frontend, _speech_tokenizer, llm, flow = (
                    module.load_models(
                        use_phoneme=self._use_phoneme,
                        sample_rate=self._sample_rate,
                    )
                )
            finally:
                os.chdir(old_cwd)

            self._m = module
            self._frontend = frontend
            self._text_frontend = text_frontend
            self._llm = llm
            self._flow = flow
            self._loaded = True

    async def list_voices(self, *, language: str | None = None) -> list[VoiceInfo]:
        lang = language or "zh-CN"
        return [
            VoiceInfo(
                voice_id="zero-shot",
                name="GLM-TTS (zero-shot via prompt audio)",
                language=lang,
                gender=None,
                style=None,
                sample_rate=self._sample_rate,
            )
        ]
