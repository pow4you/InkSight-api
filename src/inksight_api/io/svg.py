"""SVG export: each stroke becomes a polyline path, ready for plotters and editors."""

from __future__ import annotations

from pathlib import Path

from ..core.ink import Ink
from .base import InkWriter, ink_writers
from .inkml import format_number as _fmt

_SVG_NS = "http://www.w3.org/2000/svg"


@ink_writers.register(".svg")
class SVGWriter(InkWriter):
    """Write ink as an SVG with one ``<path>`` per stroke.

    Args:
        stroke_width: Line width in user units.
        stroke: Line colour.
        size: Fixed ``(width, height)`` canvas starting at the origin, e.g.
            ``(224, 224)`` for raw model output. ``None`` fits the ink's bounds.
        padding: Margin around the ink when fitting to its bounds.
    """

    def __init__(self, stroke_width: float = 1.0, stroke: str = "black",
                 size: tuple[float, float] | None = None, padding: float = 4.0) -> None:
        self.stroke_width = stroke_width
        self.stroke = stroke
        self.size = size
        self.padding = padding

    def to_string(self, ink: Ink) -> str:
        """The SVG document as text."""
        if self.size is not None:
            x0, y0, w, h = 0.0, 0.0, self.size[0], self.size[1]
        else:
            b = ink.bounds() or (0.0, 0.0, 1.0, 1.0)
            x0, y0 = b[0] - self.padding, b[1] - self.padding
            w, h = b[2] - b[0] + 2 * self.padding, b[3] - b[1] + 2 * self.padding

        paths = []
        for stroke in ink:
            if len(stroke) == 0:
                continue
            pts = [f"{_fmt(x)} {_fmt(y)}" for x, y in stroke]
            if len(pts) == 1:  # a lone point still shows up as a dot with round caps
                pts.append(pts[0])
            paths.append(f'    <path d="M {" L ".join(pts)}"/>')

        body = "\n".join(paths)
        return (
            f'<svg xmlns="{_SVG_NS}" viewBox="{_fmt(x0)} {_fmt(y0)} {_fmt(w)} {_fmt(h)}" '
            f'width="{_fmt(w)}" height="{_fmt(h)}">\n'
            f'  <g fill="none" stroke="{self.stroke}" stroke-width="{_fmt(self.stroke_width)}" '
            f'stroke-linecap="round" stroke-linejoin="round">\n{body}\n  </g>\n</svg>\n'
        )

    def write(self, ink: Ink, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_string(ink), encoding="utf-8")
        return path
