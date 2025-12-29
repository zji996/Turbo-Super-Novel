from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from core.paths import models_dir

_DEFAULT_ITEMS = (
    "vae",
    "text-encoder",
    "text-encoder-df11",
    "wan2.2-i2v",
    "wan2.2-i2v-quant",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move TurboDiffusion (to-video) model folders from <MODELS_DIR>/ to <MODELS_DIR>/2v/."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=models_dir(),
        help="Root MODELS_DIR (default: resolved from env/repo).",
    )
    parser.add_argument(
        "--items",
        default=",".join(_DEFAULT_ITEMS),
        help=f"Comma-separated folders to move (default: {','.join(_DEFAULT_ITEMS)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without modifying files",
    )
    return parser.parse_args(argv)


def _split_items(raw: str) -> tuple[str, ...]:
    parts = [p.strip().strip("/").strip() for p in str(raw).split(",")]
    return tuple(p for p in parts if p)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.models_dir).expanduser().resolve()
    src_root = root
    dst_root = (root / "2v").resolve()
    items = _split_items(args.items)

    print(f"MODELS_DIR={root}")
    print(f"src={src_root}")
    print(f"dst={dst_root}")

    dst_root.mkdir(parents=True, exist_ok=True) if not args.dry_run else None

    planned: list[tuple[Path, Path]] = []
    for name in items:
        src = (src_root / name).resolve()
        dst = (dst_root / name).resolve()
        if not src.exists():
            continue
        if src_root == dst_root or src == dst:
            continue
        if dst.exists():
            raise SystemExit(f"Refusing to overwrite existing path: {dst}")
        planned.append((src, dst))

    if not planned:
        print("Nothing to move.")
        return 0

    print(f"planned={len(planned)}")
    for src, dst in planned:
        print(f"- {src} -> {dst}")

    if args.dry_run:
        return 0

    for src, dst in planned:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
