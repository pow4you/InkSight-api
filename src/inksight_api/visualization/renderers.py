"""The notebook's rainbow stroke style as a configurable renderer."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.ink import Ink
from ..utils.lazy import require
from .base import InkRenderer, renderers


@dataclass(frozen=True)
class StrokeStyle:
    """Look of :class:`RainbowInkRenderer` (defaults reproduce the notebook).

    Attributes:
        linewidth: Stroke width in points.
        outline_color: Colour of the halo around each stroke; ``None`` for no halo.
        outline_scale: Halo width as a multiple of ``linewidth``.
        alpha_start: Opacity at the start of a stroke.
        alpha_end: Opacity at the end of a stroke.
        colormap: Matplotlib colormap sampled once per stroke.
        value_scale: Multiplier on the colour's HSV value (darkens the colours).
    """

    linewidth: float = 1.8
    outline_color: str | None = "white"
    outline_scale: float = 1.25
    alpha_start: float = 1.0
    alpha_end: float = 0.5
    colormap: str = "rainbow"
    value_scale: float = 0.65


@renderers.register("rainbow")
class RainbowInkRenderer(InkRenderer):
    """One colour per stroke along a colormap, fading along each stroke.

    Args:
        style: Visual parameters; defaults to :class:`StrokeStyle`.
    """

    def __init__(self, style: StrokeStyle | None = None) -> None:
        self.style = style or StrokeStyle()

    def stroke_color(self, index: int, total: int) -> tuple[float, float, float]:
        """RGB colour of stroke ``index`` out of ``total`` (reversed colormap, darkened)."""
        mpl = require("matplotlib", extra="viz", purpose=" to draw ink")
        total = max(total, 1)
        cmap = mpl.colormaps[self.style.colormap].resampled(total)
        r, g, b, _ = cmap(total - 1 - index)
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return colorsys.hsv_to_rgb(h, s, max(0.0, v * self.style.value_scale))

    def draw(self, ax: Any, ink: Ink, *, segment_budget: int | None = None) -> list[Any]:
        require("matplotlib", extra="viz", purpose=" to draw ink")
        from matplotlib.collections import LineCollection
        from matplotlib.patheffects import withStroke

        st = self.style
        artists: list[Any] = []
        remaining = segment_budget
        for index, stroke in enumerate(ink):
            if len(stroke) < 2:
                continue
            segments = stroke.segments
            if remaining is not None:
                if remaining <= 0:
                    break
                segments = segments[:remaining]
                remaining -= len(segments)

            rgb = self.stroke_color(index, len(ink))
            alphas = np.linspace(st.alpha_start, st.alpha_end, len(segments))
            collection = LineCollection(segments, colors=[(*rgb, a) for a in alphas],
                                        linewidth=st.linewidth)
            if st.outline_color is not None:
                collection.set_path_effects([withStroke(
                    linewidth=st.linewidth * st.outline_scale, foreground=st.outline_color)])
            ax.add_collection(collection)
            artists.append(collection)
        return artists
