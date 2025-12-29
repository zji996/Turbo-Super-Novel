from __future__ import annotations

from pathlib import Path

_vendored = (
    Path(__file__).resolve().parent.parent / "videogen" / "_vendor" / "rcm"
).resolve()
if _vendored.is_dir():
    __path__.append(str(_vendored))  # type: ignore[name-defined]
