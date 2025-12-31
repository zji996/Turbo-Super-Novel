from __future__ import annotations

import argparse
import io
import sys
import time
import wave
from pathlib import Path
from typing import Any

import httpx


def _make_silence_wav_bytes(*, seconds: float = 0.5, sample_rate: int = 24000) -> bytes:
    frames = max(1, int(seconds * sample_rate))
    pcm16 = b"\x00\x00" * frames
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    return buf.getvalue()


def _poll_json(
    client: httpx.Client,
    url: str,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    terminal_statuses: set[str],
    status_key: str = "status",
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_payload: dict[str, Any] | None = None

    while time.time() < deadline:
        resp = client.get(url)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            last_payload = payload
            status = str(payload.get(status_key, "") or "")
            if status in terminal_statuses:
                return payload
        time.sleep(poll_interval_seconds)

    if last_payload is None:
        raise RuntimeError(f"Polling timed out (no payload): {url}")
    raise RuntimeError(f"Polling timed out (last={last_payload}): {url}")


def _smoke_tts(
    client: httpx.Client,
    api_base: str,
    *,
    text: str,
    provider: str,
    prompt_text: str,
    sample_rate: int,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    wav_bytes = _make_silence_wav_bytes(seconds=0.5, sample_rate=sample_rate)
    files = {
        "prompt_audio": ("prompt.wav", wav_bytes, "audio/wav"),
    }
    data = {
        "name": "smoke-profile",
        "prompt_text": prompt_text,
        "provider": provider,
        "sample_rate": str(sample_rate),
        "is_default": "false",
    }
    resp = client.post(f"{api_base}/v1/tts/speaker-profiles", data=data, files=files)
    resp.raise_for_status()
    profile = resp.json()
    profile_id = str(profile.get("id") or profile.get("profile_id") or "")
    if not profile_id:
        raise RuntimeError(f"Unexpected speaker profile response: {profile}")

    resp = client.post(
        f"{api_base}/v1/tts/jobs/with-profile",
        json={"text": text, "profile_id": profile_id},
    )
    resp.raise_for_status()
    job = resp.json()
    job_id = str(job.get("job_id") or "")
    if not job_id:
        raise RuntimeError(f"Unexpected job response: {job}")

    return _poll_json(
        client,
        f"{api_base}/v1/tts/jobs/{job_id}",
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        terminal_statuses={"SUCCEEDED", "FAILED", "SUBMIT_FAILED"},
    )


def _smoke_videogen(
    client: httpx.Client,
    api_base: str,
    *,
    image_path: Path,
    prompt: str,
    seed: int,
    num_steps: int,
    quantized: bool,
    duration_seconds: int,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    with image_path.open("rb") as f:
        files = {"image": (image_path.name, f, "image/jpeg")}
        data = {
            "prompt": prompt,
            "seed": str(seed),
            "num_steps": str(num_steps),
            "quantized": str(quantized).lower(),
            "duration_seconds": str(duration_seconds),
        }
        resp = client.post(f"{api_base}/v1/videogen/wan22-i2v/jobs", data=data, files=files)
        resp.raise_for_status()
        job = resp.json()

    job_id = str(job.get("job_id") or "")
    if not job_id:
        raise RuntimeError(f"Unexpected videogen response: {job}")

    return _poll_json(
        client,
        f"{api_base}/v1/videogen/jobs/{job_id}",
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        terminal_statuses={"SUCCESS", "FAILURE", "REVOKED"},
        status_key="status",
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="E2E smoke test: Web/API -> Celery -> Worker (TTS + VideoGen)."
    )
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--require-success", action="store_true")

    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--skip-videogen", action="store_true")

    parser.add_argument("--tts-text", default="Hello from TSN smoke test.")
    parser.add_argument("--tts-provider", default="glm_tts")
    parser.add_argument("--tts-prompt-text", default="Hello.")
    parser.add_argument("--tts-sample-rate", type=int, default=24000)

    parser.add_argument(
        "--videogen-image",
        type=Path,
        default=Path("assets/turbodiffusion/i2v_inputs/i2v_input_0.jpg"),
    )
    parser.add_argument("--videogen-prompt", default="A cute cat, cinematic, 4k")
    parser.add_argument("--videogen-seed", type=int, default=0)
    parser.add_argument("--videogen-num-steps", type=int, default=4)
    parser.add_argument("--videogen-quantized", action="store_true", default=True)
    parser.add_argument("--videogen-duration-seconds", type=int, default=2)
    args = parser.parse_args(argv)

    api_base = str(args.api_base).rstrip("/")
    failures: list[str] = []

    with httpx.Client(timeout=60) as client:
        try:
            resp = client.get(f"{api_base}/health")
            resp.raise_for_status()
        except Exception as exc:
            print(f"[health] FAIL: {exc}", file=sys.stderr)
            return 2
        print("[health] OK")

        if not args.skip_tts:
            try:
                payload = _smoke_tts(
                    client,
                    api_base,
                    text=str(args.tts_text),
                    provider=str(args.tts_provider),
                    prompt_text=str(args.tts_prompt_text),
                    sample_rate=int(args.tts_sample_rate),
                    timeout_seconds=float(args.timeout_seconds),
                    poll_interval_seconds=float(args.poll_interval_seconds),
                )
                print(f"[tts] done: {payload}")
                if args.require_success and str(payload.get("status")) != "SUCCEEDED":
                    failures.append(f"tts status={payload.get('status')}")
            except Exception as exc:
                failures.append(f"tts error={exc}")
                print(f"[tts] FAIL: {exc}", file=sys.stderr)

        if not args.skip_videogen:
            try:
                payload = _smoke_videogen(
                    client,
                    api_base,
                    image_path=Path(args.videogen_image),
                    prompt=str(args.videogen_prompt),
                    seed=int(args.videogen_seed),
                    num_steps=int(args.videogen_num_steps),
                    quantized=bool(args.videogen_quantized),
                    duration_seconds=int(args.videogen_duration_seconds),
                    timeout_seconds=float(args.timeout_seconds),
                    poll_interval_seconds=float(args.poll_interval_seconds),
                )
                print(f"[videogen] done: {payload}")
                if args.require_success and str(payload.get("status")) != "SUCCESS":
                    failures.append(f"videogen status={payload.get('status')}")
            except Exception as exc:
                failures.append(f"videogen error={exc}")
                print(f"[videogen] FAIL: {exc}", file=sys.stderr)

    if failures:
        print("[summary] failures:", ", ".join(failures), file=sys.stderr)
        return 1 if args.require_success else 0

    print("[summary] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

