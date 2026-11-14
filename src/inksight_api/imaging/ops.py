"""Small pure image operations used by visualisation."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageEnhance

from ..core.geometry import BoundingBox


def darken(image: Image.Image, factor: float = 0.45) -> Image.Image:
    """Return a copy scaled in brightness (``0.45`` is the notebook's backdrop)."""
    return ImageEnhance.Brightness(image).enhance(factor)


def resize_to_max_side(image: Image.Image, max_size: int = 2000) -> Image.Image:
    """Downscale so the longer side is at most ``max_size`` (never upscales)."""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def draw_boxes(image: Image.Image, boxes: list[BoundingBox], *,
               outline: str = "red", width: int = 2) -> Image.Image:
    """Return a copy of ``image`` with rectangles drawn around ``boxes``."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for b in boxes:
        draw.rectangle([b.left, b.top, b.right, b.bottom], outline=outline, width=width)
    return annotated
