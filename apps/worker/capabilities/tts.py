"""TTS capability tasks (GPU)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
import traceback
from typing import Any
import uuid

from celery_app import celery_app
from core.paths import data_dir
from model_manager import ModelManager
from s3_utils import get_s3_client

from .base import CapabilityWorker

logger = logging.getLogger(__name__)


def _s3_client():  # noqa: ANN001
    return get_s3_client()


def _config_cache_key(config: dict[str, Any]) -> str:
    try:
        return json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return str(sorted((k, str(v)) for k, v in config.items()))


def _db_update_tts_job(
    *,
    job_id: str,
    status: str,
    error: str | None = None,
    result: dict | None = None,
    audio_duration_seconds: float | None = None,
) -> None:
    try:
        from db import TTSJob, ensure_schema, session_scope

        ensure_schema()
        with session_scope() as session:
            row = session.get(TTSJob, uuid.UUID(job_id))
            if row is None:
                return
            row.status = status
            row.error = error
            row.result = result
            if audio_duration_seconds is not None:
                row.audio_duration_seconds = float(audio_duration_seconds)
    except Exception:
        logger.exception("DB update crashed for job_id=%s status=%s", job_id, status)


def _audio_duration_seconds(path: Path) -> float | None:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        try:
            import wave

            with wave.open(str(path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
            if rate <= 0:
                return None
            return float(frames) / float(rate)
        except Exception:
            return None

    try:
        import librosa

        return float(librosa.get_duration(path=str(path)))
    except Exception:
        return None


def _audio_duration_seconds_wav(path: Path) -> float | None:
    try:
        import wave

        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
        if rate <= 0:
            return None
        return float(frames) / float(rate)
    except Exception:
        return None


class TTSWorker(CapabilityWorker):
    name = "tts"
    requires_gpu = True

    def __init__(self) -> None:
        self._providers: dict[tuple[str, int, str], Any] = {}

    @property
    def is_loaded(self) -> bool:
        return bool(self._providers)

    def load_models(self) -> None:
        provider = str(os.getenv("CAP_TTS_DEFAULT_PROVIDER", "glm_tts") or "glm_tts")
        sample_rate_raw = os.getenv("CAP_TTS_SAMPLE_RATE", "")
        sample_rate = int(sample_rate_raw) if sample_rate_raw.strip() else 24000
        if provider.strip().lower() not in {"glm_tts", "glm-tts", "glmtts"}:
            return
        self._get_provider(provider=provider, sample_rate=sample_rate, config={})

    def unload_models(self) -> None:
        for instance in self._providers.values():
            try:
                instance.unload()
            except Exception:
                continue
        self._providers = {}

    def execute(self, job: dict) -> dict:
        kind = str(job.get("kind", "") or "")
        if kind == "cap.tts.synthesize":
            return self._execute_synthesize(job)
        if kind == "cap.tts.scene":
            return self._execute_scene_tts(job)
        raise ValueError(f"Unknown TTS job kind: {kind}")

    def _get_provider(
        self, *, provider: str, sample_rate: int, config: dict[str, Any]
    ) -> Any:
        provider_norm = str(provider or "").strip().lower()
        if provider_norm in {"glm_tts", "glm-tts", "glmtts"}:
            provider_norm = "glm_tts"

        if provider_norm != "glm_tts":
            from tts import get_tts_provider

            return get_tts_provider(provider_norm, sample_rate=int(sample_rate), **config)

        key = (provider_norm, int(sample_rate), _config_cache_key(config))
        existing = self._providers.get(key)
        if existing is not None:
            return existing

        from tts import get_tts_provider

        created = get_tts_provider(provider_norm, sample_rate=int(sample_rate), **config)
        self._providers[key] = created
        return created

    def _execute_synthesize(self, job: dict) -> dict:
        job_id: str = str(job["job_id"])
        text: str = str(job["text"])
        provider: str = str(job["provider"])
        output_bucket: str = str(job["output_bucket"])
        output_key: str = str(job["output_key"])
        prompt_text: str | None = job.get("prompt_text")
        prompt_audio_bucket: str | None = job.get("prompt_audio_bucket")
        prompt_audio_key: str | None = job.get("prompt_audio_key")
        sample_rate: int = int(job.get("sample_rate", 24000))
        config: dict[str, Any] = dict(job.get("config", {}))

        job_dir = (data_dir() / "tts" / "jobs" / job_id).resolve()
        job_dir.mkdir(parents=True, exist_ok=True)
        output_path = (job_dir / "output").with_suffix(Path(output_key).suffix or ".wav")
        prompt_wav_path: Path | None = None

        _db_update_tts_job(job_id=job_id, status="STARTED", error=None, result=None)

        try:
            s3 = _s3_client()

            provider_norm = str(provider or "").strip().lower()
            if provider_norm in {"glm_tts", "glm-tts", "glmtts"}:
                if not prompt_text or not prompt_text.strip():
                    raise ValueError("glm_tts requires prompt_text")
                if not prompt_audio_bucket or not prompt_audio_key:
                    raise ValueError("glm_tts requires prompt_audio_bucket/prompt_audio_key")
                prompt_wav_path = (job_dir / "prompt").with_suffix(
                    Path(prompt_audio_key).suffix or ".wav"
                )
                s3.download_file(
                    str(prompt_audio_bucket), str(prompt_audio_key), str(prompt_wav_path)
                )

            from tts import TTSError

            tts_provider = self._get_provider(
                provider=provider_norm, sample_rate=sample_rate, config=config
            )

            synth_kwargs: dict[str, Any] = {}
            if prompt_wav_path is not None:
                synth_kwargs["prompt_wav"] = str(prompt_wav_path)
                synth_kwargs["prompt_text"] = prompt_text

            try:
                asyncio.run(
                    tts_provider.synthesize(
                        text=text, output_path=output_path, **synth_kwargs
                    )
                )
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(
                        tts_provider.synthesize(
                            text=text, output_path=output_path, **synth_kwargs
                        )
                    )
                finally:
                    loop.close()

            if not output_path.is_file():
                raise TTSError(f"Output file not created: {output_path}")

            s3.upload_file(str(output_path), output_bucket, output_key)
            duration = _audio_duration_seconds(output_path)
            result = {
                "job_id": job_id,
                "provider": provider,
                "output_bucket": output_bucket,
                "output_key": output_key,
                "audio_duration_seconds": duration,
            }
            _db_update_tts_job(
                job_id=job_id,
                status="SUCCEEDED",
                error=None,
                result=result,
                audio_duration_seconds=duration,
            )
            return result

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("TTS synthesis failed (job_id=%s): %s\n%s", job_id, exc, tb)
            _db_update_tts_job(job_id=job_id, status="FAILED", error=str(exc), result=None)
            raise

    def _execute_scene_tts(self, job: dict) -> dict[str, Any]:
        scene_id: str = str(job["scene_id"])
        project_id: str = str(job["project_id"])
        pipeline_id: str = str(job["pipeline_id"])
        provider: str = str(job.get("provider", "glm_tts"))
        sample_rate: int = int(job.get("sample_rate", 24000))
        prompt_audio_bucket: str | None = job.get("prompt_audio_bucket")
        prompt_audio_key: str | None = job.get("prompt_audio_key")
        prompt_text: str | None = job.get("prompt_text")
        config: dict[str, Any] = dict(job.get("config", {}))

        try:
            scene_uuid = uuid.UUID(scene_id)
            project_uuid = uuid.UUID(project_id)
            pipeline_uuid = uuid.UUID(pipeline_id)
        except Exception as exc:
            raise ValueError(f"invalid ids: {exc}") from exc

        provider_norm = str(provider or "").strip().lower()
        if provider_norm in {"glm_tts", "glm-tts", "glmtts"}:
            provider_norm = "glm_tts"
        if provider_norm not in {"glm_tts", "edge"}:
            raise ValueError(f"unsupported provider: {provider}")

        s3 = _s3_client()
        bucket = os.getenv("S3_BUCKET_NAME") or ""
        if not bucket:
            raise RuntimeError("Missing required env var: S3_BUCKET_NAME")

        output_key = f"novel/{project_id}/scenes/{scene_id}/audio.wav"
        scene_dir = (data_dir() / "novel" / "scenes" / scene_id).resolve()
        scene_dir.mkdir(parents=True, exist_ok=True)
        output_path = (scene_dir / "audio.wav").resolve()
        provider_output_path = output_path
        if provider_norm == "edge":
            provider_output_path = (scene_dir / "audio.mp3").resolve()

        prompt_wav_path: Path | None = None
        if provider_norm == "glm_tts":
            if not prompt_text or not prompt_text.strip():
                raise ValueError("glm_tts requires prompt_text")
            if not prompt_audio_bucket or not prompt_audio_key:
                raise ValueError("glm_tts requires prompt_audio_bucket/prompt_audio_key")
            prompt_wav_path = (scene_dir / "prompt").with_suffix(
                Path(prompt_audio_key).suffix or ".wav"
            )
            s3.download_file(prompt_audio_bucket, prompt_audio_key, str(prompt_wav_path))

        from db import NovelPipeline, NovelProject, NovelScene, ensure_schema, session_scope

        ensure_schema()
        with session_scope() as session:
            scene = session.get(NovelScene, scene_uuid)
            pipeline = session.get(NovelPipeline, pipeline_uuid)
            project = session.get(NovelProject, project_uuid)
            if scene is None:
                raise ValueError("scene not found")
            if project is None:
                raise ValueError("project not found")
            if pipeline is None:
                raise ValueError("pipeline not found")
            if scene.project_id != project_uuid:
                raise ValueError("scene.project_id mismatch")
            if pipeline.project_id != project_uuid:
                raise ValueError("pipeline.project_id mismatch")

            scene_text = scene.text
            scene.status = "TTS_RUNNING"
            scene.error = None
            scene.audio_bucket = bucket
            scene.audio_key = output_key
            if pipeline.status in {"PENDING"}:
                pipeline.status = "RUNNING"

        try:
            from tts import TTSError

            tts_provider = self._get_provider(
                provider=provider_norm, sample_rate=sample_rate, config=config
            )
            synth_kwargs: dict[str, Any] = {}
            if prompt_wav_path is not None:
                synth_kwargs["prompt_wav"] = str(prompt_wav_path)
                synth_kwargs["prompt_text"] = prompt_text

            try:
                asyncio.run(
                    tts_provider.synthesize(
                        text=scene_text,
                        output_path=provider_output_path,
                        **synth_kwargs,
                    )
                )
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(
                        tts_provider.synthesize(
                            text=scene_text,
                            output_path=provider_output_path,
                            **synth_kwargs,
                        )
                    )
                finally:
                    loop.close()

            if not provider_output_path.is_file():
                raise TTSError(f"Output file not created: {provider_output_path}")

            if provider_norm == "edge":
                try:
                    import librosa
                    import soundfile as sf

                    audio, _sr = librosa.load(
                        str(provider_output_path), sr=int(sample_rate), mono=True
                    )
                    sf.write(str(output_path), audio, int(sample_rate), subtype="PCM_16")
                except Exception as exc:
                    raise TTSError(f"Failed to convert Edge output to wav: {exc}") from exc

            if not output_path.is_file():
                raise TTSError(f"Output wav not created: {output_path}")

            s3.upload_file(
                str(output_path),
                bucket,
                output_key,
                ExtraArgs={"ContentType": "audio/wav"},
            )
            duration = _audio_duration_seconds_wav(output_path)

            ensure_schema()
            with session_scope() as session:
                scene = session.get(NovelScene, scene_uuid)
                pipeline = session.get(NovelPipeline, pipeline_uuid)
                if scene is not None:
                    scene.status = "TTS_SUCCEEDED"
                    scene.audio_bucket = bucket
                    scene.audio_key = output_key
                    scene.audio_duration_seconds = duration
                    scene.error = None
                if pipeline is not None:
                    pipeline.completed_tasks = int(pipeline.completed_tasks or 0) + 1
                    if int(pipeline.completed_tasks or 0) >= int(
                        pipeline.total_tasks or 0
                    ):
                        pipeline.status = "COMPLETED"
                        pipeline.completed_at = datetime.now(timezone.utc)

            return {
                "scene_id": scene_id,
                "project_id": project_id,
                "pipeline_id": pipeline_id,
                "output_bucket": bucket,
                "output_key": output_key,
                "audio_duration_seconds": duration,
            }

        except Exception as exc:
            tb = traceback.format_exc()
            (scene_dir / "tts_error.txt").write_text(tb, encoding="utf-8", errors="replace")

            ensure_schema()
            with session_scope() as session:
                scene = session.get(NovelScene, scene_uuid)
                pipeline = session.get(NovelPipeline, pipeline_uuid)
                if scene is not None:
                    scene.status = "TTS_FAILED"
                    scene.error = str(exc)
                if pipeline is not None:
                    pipeline.status = "FAILED"
                    pipeline.error = str(exc)
                    pipeline.completed_at = datetime.now(timezone.utc)

            raise


_tts_worker = TTSWorker()
ModelManager.register(_tts_worker)


@celery_app.task(name="cap.tts.synthesize", bind=True)
def synthesize_tts(  # type: ignore[misc]
    self,
    *,
    job_id: str,
    text: str,
    provider: str,
    output_bucket: str,
    output_key: str,
    prompt_text: str | None = None,
    prompt_audio_bucket: str | None = None,
    prompt_audio_key: str | None = None,
    sample_rate: int = 24000,
    **config: Any,
) -> dict:
    worker = ModelManager.acquire("tts")
    try:
        return worker.execute(
            {
                "kind": "cap.tts.synthesize",
                "job_id": job_id,
                "text": text,
                "provider": provider,
                "output_bucket": output_bucket,
                "output_key": output_key,
                "prompt_text": prompt_text,
                "prompt_audio_bucket": prompt_audio_bucket,
                "prompt_audio_key": prompt_audio_key,
                "sample_rate": int(sample_rate),
                "config": dict(config),
            }
        )
    finally:
        ModelManager.release("tts")


@celery_app.task(name="cap.tts.scene", bind=True)
def process_scene_tts(  # type: ignore[misc]
    self,
    *,
    scene_id: str,
    project_id: str,
    pipeline_id: str,
    provider: str = "glm_tts",
    sample_rate: int = 24000,
    prompt_audio_bucket: str | None = None,
    prompt_audio_key: str | None = None,
    prompt_text: str | None = None,
    **config: Any,
) -> dict[str, Any]:
    worker = ModelManager.acquire("tts")
    try:
        return worker.execute(
            {
                "kind": "cap.tts.scene",
                "scene_id": scene_id,
                "project_id": project_id,
                "pipeline_id": pipeline_id,
                "provider": provider,
                "sample_rate": int(sample_rate),
                "prompt_audio_bucket": prompt_audio_bucket,
                "prompt_audio_key": prompt_audio_key,
                "prompt_text": prompt_text,
                "config": dict(config),
            }
        )
    finally:
        ModelManager.release("tts")
