from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from libs.pycore.paths import models_dir
from libs.turbodiffusion.registry import iter_artifacts


def _format_bytes(value: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024:
            return f"{value:.1f}{unit}" if unit != "B" else f"{value}{unit}"
        value /= 1024
    return f"{value:.1f}PiB"


def _download(url: str, dst: Path, *, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        print(f"skip: {dst}")
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    req = urllib.request.Request(url, headers={"User-Agent": "Turbo-Super-Novel/td-downloader"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        total = int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        with open(tmp, "wb") as f:  # noqa: PTH123
            while True:
                chunk = resp.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\rdownloading: {dst.name} {pct:6.2f}% ({_format_bytes(downloaded)}/{_format_bytes(total)})", end="")
                else:
                    print(f"\rdownloading: {dst.name} ({_format_bytes(downloaded)})", end="")
    print()

    tmp.replace(dst)
    print(f"done: {dst}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Download TurboDiffusion model artifacts into MODELS_DIR.")
    parser.add_argument("--model", default="TurboWan2.2-I2V-A14B-720P", help="Model name (default: TurboWan2.2-I2V-A14B-720P)")
    parser.add_argument("--quantized", action=argparse.BooleanOptionalAction, default=True, help="Use quantized checkpoints (default: true)")
    parser.add_argument(
        "--groups",
        default="base,dit",
        help="Comma-separated groups to download: base,dit (default: base,dit)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without downloading")
    args = parser.parse_args(argv)

    root = models_dir().resolve()
    groups = tuple(part.strip() for part in str(args.groups).split(",") if part.strip())
    planned = list(iter_artifacts(args.model, quantized=bool(args.quantized), groups=groups))  # type: ignore[arg-type]

    print(f"MODELS_DIR={root}")
    if os.getenv("MODELS_DIR"):
        print("MODELS_DIR is set via env var.")
    print(f"planned={len(planned)} files")

    for artifact in planned:
        dst = (root / artifact.relative_path).resolve()
        print(f"- {artifact.group}: {dst}")
        if args.dry_run:
            continue
        try:
            _download(artifact.url, dst, force=args.force)
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"download failed ({exc.code}) for {artifact.url}") from exc

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

