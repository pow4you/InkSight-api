"""JSON reader/writer: the simplest format to hand paths to another program.

Layout::

    {"text": "hello" | null, "strokes": [[[x, y], [x, y], ...], ...]}
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.ink import Ink, Stroke
from .base import InkReader, InkWriter, ink_readers, ink_writers


def ink_to_dict(ink: Ink, *, decimals: int = 3) -> dict:
    """Plain-Python form of an ink (lists of ``[x, y]`` per stroke)."""
    return {
        "text": ink.text,
        "strokes": [[[round(x, decimals), round(y, decimals)] for x, y in stroke] for stroke in ink],
    }


@ink_writers.register(".json")
class JSONWriter(InkWriter):
    """Write an ink as JSON.

    Args:
        decimals: Digits kept after the decimal point.
        indent: Passed to ``json.dumps`` (``None`` for one line).
    """

    def __init__(self, decimals: int = 3, indent: int | None = None) -> None:
        self.decimals = decimals
        self.indent = indent

    def to_string(self, ink: Ink) -> str:
        return json.dumps(ink_to_dict(ink, decimals=self.decimals), indent=self.indent,
                          ensure_ascii=False)

    def write(self, ink: Ink, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_string(ink) + "\n", encoding="utf-8")
        return path


@ink_readers.register(".json")
class JSONReader(InkReader):
    """Read the layout written by :class:`JSONWriter`."""

    def read(self, path: str | Path) -> Ink:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return Ink([Stroke(points) for points in data["strokes"]], text=data.get("text"))
