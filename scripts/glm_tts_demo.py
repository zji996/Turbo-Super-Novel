from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from core.paths import repo_root
from tts import TTSError, get_tts_provider

DEFAULT_PROMPT_TEXT = "他当时还跟线下其他的站姐吵架，然后，打架进局子了。"
DEFAULT_THIRD_PARTY_PROMPT_WAV = (
    repo_root() / "third_party" / "GLM-TTS" / "examples" / "prompt" / "jiayan_zh.wav"
)


def _resolve_default_prompt(
    prompt_wav: Path | None, prompt_text: str | None, *, allow_third_party: bool
) -> tuple[Path | None, str | None]:
    if prompt_wav is not None:
        return prompt_wav, prompt_text
    if not allow_third_party:
        return None, prompt_text
    if DEFAULT_THIRD_PARTY_PROMPT_WAV.is_file():
        return DEFAULT_THIRD_PARTY_PROMPT_WAV, prompt_text or DEFAULT_PROMPT_TEXT
    return None, prompt_text


async def _run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Run GLM-TTS inference via tts (ModelScope weights)."
    )
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out.wav"),
        help="Output wav path (default: ./out.wav)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Model directory (default: <MODELS_DIR>/tts/GLM-TTS)",
    )
    parser.add_argument(
        "--prompt-wav",
        type=Path,
        default=None,
        help="Prompt/reference wav for zero-shot voice cloning",
    )
    parser.add_argument(
        "--prompt-text", default=None, help="Transcript of the prompt wav"
    )
    parser.add_argument(
        "--use-third-party-prompt",
        action="store_true",
        help="Use third_party/GLM-TTS example prompt wav (if present) for convenience",
    )
    args = parser.parse_args(argv)

    prompt_wav, prompt_text = _resolve_default_prompt(
        args.prompt_wav,
        args.prompt_text,
        allow_third_party=args.use_third_party_prompt,
    )

    if prompt_wav is None:
        print(
            "No prompt wav provided. For GLM-TTS zero-shot, pass --prompt-wav and --prompt-text."
        )
        print(
            f"For a quick demo (if submodule exists): {DEFAULT_THIRD_PARTY_PROMPT_WAV}"
        )

    tts = get_tts_provider("glm_tts", model_dir=args.model_dir)
    try:
        out = await tts.synthesize(
            text=args.text,
            output_path=args.output,
            prompt_wav=str(prompt_wav) if prompt_wav is not None else None,
            prompt_text=prompt_text,
        )
    except TTSError:
        raise
    print(f"Saved: {out}")
    return 0


def main(argv: list[str]) -> int:
    return asyncio.run(_run(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
