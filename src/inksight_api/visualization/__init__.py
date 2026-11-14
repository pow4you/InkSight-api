"""Drawing and animating ink (needs the ``viz`` extra)."""

from .animation import InkAnimator
from .base import InkRenderer, renderers
from .page import adaptive_line_width, plot_page_result
from .plotting import plot_ink
from .renderers import RainbowInkRenderer, StrokeStyle

__all__ = [
    "InkAnimator", "InkRenderer", "RainbowInkRenderer", "StrokeStyle",
    "adaptive_line_width", "plot_ink", "plot_page_result", "renderers",
]
