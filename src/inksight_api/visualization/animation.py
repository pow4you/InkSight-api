"""Stroke-by-stroke animations (ports of ``plot_ink_to_video`` and ``plot_ink_to_gif``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from ..core.ink import Ink
from ..imaging.ops import darken
from ..utils.lazy import require
from .base import InkRenderer
from .renderers import RainbowInkRenderer

_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


class InkAnimator:
    """Replay an ink being written and save it as GIF or video.

    Args:
        renderer: Stroke look; default is :class:`RainbowInkRenderer`.
        fps: Frames per second.
        figsize: Matplotlib figure size in inches.
        dpi: Figure resolution.
        canvas_size: Coordinate frame ``(width, height)``; ``(224, 224)`` for raw model output.
        background_brightness: Brightness factor for the optional backdrop image.
    """

    def __init__(self, renderer: InkRenderer | None = None, *, fps: int = 30,
                 figsize: tuple[float, float] = (4, 4), dpi: int = 150,
                 canvas_size: tuple[float, float] = (224, 224),
                 background_brightness: float = 0.45) -> None:
        self.renderer = renderer or RainbowInkRenderer()
        self.fps = fps
        self.figsize = figsize
        self.dpi = dpi
        self.canvas_size = canvas_size
        self.background_brightness = background_brightness

    def save(self, ink: Ink, path: str | Path, *, image: Image.Image | None = None) -> Path:
        """Render the animation to ``path``.

        The format follows the suffix: ``.gif`` is written with Pillow (no
        ImageMagick needed); ``.mp4``/``.webm``/... need ``ffmpeg`` on the PATH.

        Raises:
            ValueError: For an unsupported suffix.
        """
        require("matplotlib", extra="viz", purpose=" to animate ink")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".gif":
            writer: Any = PillowWriter(fps=self.fps)
        elif suffix in _VIDEO_SUFFIXES:
            writer = FFMpegWriter(fps=self.fps)
        else:
            raise ValueError(f"unsupported animation format {suffix!r}; use .gif or one of "
                             f"{sorted(_VIDEO_SUFFIXES)}")

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        try:
            if image is not None:
                ax.imshow(darken(image, self.background_brightness))
            ax.set_xlim(0, self.canvas_size[0])
            ax.set_ylim(0, self.canvas_size[1])
            ax.invert_yaxis()
            ax.axis("off")
            fig.tight_layout()

            drawn: list[Any] = []

            def update(frame: int) -> None:
                for artist in drawn:
                    artist.remove()
                drawn[:] = self.renderer.draw(ax, ink, segment_budget=frame)

            total_segments = sum(max(len(s) - 1, 0) for s in ink)
            animation = FuncAnimation(fig, update, frames=total_segments + 1, blit=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            animation.save(str(path), writer=writer)
        finally:
            plt.close(fig)
        return path
