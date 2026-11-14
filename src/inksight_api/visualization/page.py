"""The notebook's three-panel full-page result figure."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..imaging.ops import darken, resize_to_max_side
from ..pipeline.results import PipelineResult
from ..utils.lazy import require
from .renderers import RainbowInkRenderer, StrokeStyle


def adaptive_line_width(result: PipelineResult, default: float = 1.8) -> float:
    """Line width that scales with the average word size (``mean(min side) / 150``)."""
    if not result.words:
        return default
    sides = [min(w.box.width, w.box.height) for w in result.words]
    return float(np.mean(sides)) / 150


def _figure_size(width: int, height: int, max_side: float = 80) -> tuple[float, float]:
    """Figure size from the image aspect ratio, capped like in the notebook."""
    aspect = height / width
    if aspect < 0.5:
        fig_h = 5.0
        return min(fig_h / aspect, max_side), fig_h
    if aspect > 2:
        fig_w = min(max_side, 15.0)
        return fig_w, min(fig_w * aspect, max_side)
    fig_w = min(width / 100, max_side)
    return fig_w, fig_w * aspect


def plot_page_result(result: PipelineResult, *, title: str = "InkSight Result (Public Small-p)",
                     max_display_side: int = 2000, line_width: float | None = None,
                     save_to: str | None = None, dpi: int = 300) -> Any:
    """Draw input, word boxes and the derendered ink as three stacked panels.

    Args:
        result: A page result from :class:`~inksight_api.pipeline.PageInkPipeline`.
        title: Title of the third panel.
        max_display_side: Longer image side is downscaled to this for display.
        line_width: Stroke width; derived from word sizes if omitted.
        save_to: If given, also save the figure there.
        dpi: Resolution used when saving.

    Returns:
        The matplotlib ``Figure``.
    """
    plt = require("matplotlib.pyplot", extra="viz", purpose=" to draw the page figure")
    lw = line_width if line_width is not None else adaptive_line_width(result)

    src_w, src_h = result.image.size
    display = resize_to_max_side(result.image, max_display_side)
    display_boxes = resize_to_max_side(result.annotated_image(), max_display_side)
    sx, sy = display.size[0] / src_w, display.size[1] / src_h

    fig, ax = plt.subplots(3, 1, figsize=_figure_size(src_w, src_h))
    ax[0].imshow(display)
    ax[0].set_title("Input Full page image")
    ax[1].imshow(display_boxes)
    ax[1].set_title("Bounding boxes")
    ax[2].imshow(darken(display, 0.45))
    ax[2].set_title(title)
    for panel in ax:  # strokes must not change the framing of the page
        panel.set_autoscale_on(False)

    on_dark = RainbowInkRenderer(StrokeStyle(linewidth=lw, alpha_start=0.5, alpha_end=1.0))
    on_boxes = RainbowInkRenderer(StrokeStyle(linewidth=lw, alpha_start=0.5, alpha_end=1.0,
                                              outline_color="black", outline_scale=1.8))
    for word in result.words:
        scaled = word.ink.scaled(sx, sy)
        on_dark.draw(ax[2], scaled)
        on_boxes.draw(ax[1], scaled)

    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=dpi, bbox_inches="tight")
    return fig
