from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from libs.pycore.paths import data_dir


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _write_concat_list(paths: list[Path]) -> Path:
    tmp_dir = (data_dir() / "tmp").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    list_path = (tmp_dir / "ffmpeg_concat.txt").resolve()

    lines: list[str] = []
    for p in paths:
        resolved = p.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(str(resolved))
        # ffmpeg concat demuxer syntax: file '<path>'
        escaped = str(resolved).replace("'", "\\'")
        lines.append(f"file '{escaped}'")

    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_path


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg  # noqa: PLC0415

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Concatenate multiple MP4 clips into a single MP4 (ffmpeg concat).")
    parser.add_argument("--output", required=True, help="Output .mp4 path")
    parser.add_argument("inputs", nargs="+", help="Input .mp4 clips, in order")
    args = parser.parse_args(argv)

    output_path = Path(args.output).resolve()
    input_paths = [Path(p) for p in args.inputs]
    list_path = _write_concat_list(input_paths)

    ffmpeg = _ffmpeg_exe()

    # Fast path: stream copy (requires same codec/params across inputs).
    try:
        _run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(output_path)])
        return 0
    except subprocess.CalledProcessError:
        pass

    # Fallback: re-encode for compatibility.
    _run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
