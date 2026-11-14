"""One-call plotting of ink (port of the notebook's ``plot_ink``)."""

from __future__ import annotations

from typing import Any

from PIL import Image

from ..core.ink import Ink
from ..imaging.ops import darken
from ..utils.lazy import require
from .base import InkRenderer
from .renderers import RainbowInkRenderer


def plot_ink(ink: Ink, ax: Any | None = None, *, image: Image.Image | None = None,
             renderer: InkRenderer | None = None,
             canvas_size: tuple[float, float] | None = (224, 224),
             background_brightness: float = 0.45) -> Any:
    """Draw ``ink`` on ``ax`` (a new axes if omitted) and return the axes.

    Args:
        ink: The ink to draw.
        ax: Target axes.
        image: Optional backdrop, darkened so the strokes stand out.
        renderer: Stroke look; default is :class:`RainbowInkRenderer`.
        canvas_size: ``(width, height)`` of the coordinate frame. The default
            ``(224, 224)`` matches raw model output. Pass ``None`` when the
            ink is in source-image pixels (the image then sets the extent).
        background_brightness: Brightness factor applied to ``image``.

    Returns:
        The matplotlib axes.
    """
    plt = require("matplotlib.pyplot", extra="viz", purpose=" to draw ink")
    if ax is None:
        _, ax = plt.subplots()
    renderer = renderer or RainbowInkRenderer()

    if image is not None:
        ax.imshow(darken(image, background_brightness))
        ax.set_autoscale_on(False)  # strokes must not change the framing of the backdrop
    renderer.draw(ax, ink)

    if canvas_size is not None:
        ax.set_xlim(0, canvas_size[0])
        ax.set_ylim(0, canvas_size[1])
        ax.invert_yaxis()
    elif image is None and (bounds := ink.bounds()) is not None:
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[3], bounds[1])  # inverted: y grows downwards
    return ax
