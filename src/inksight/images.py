"""Turning "an image" - a file, a NumPy array or a PIL image - into the forms the
vectorizers work on, and conditioning word crops for the model.

The vectorizers work on ``(H, W, 3)`` ``uint8`` RGB arrays (the model sees
colour).  :func:`load_grayscale` is provided for callers that want gray.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
from PIL import Image

from .errors import InvalidImageError

#: Anything :func:`load_rgb` accepts.
ImageLike = Union[str, "os.PathLike[str]", np.ndarray, Image.Image]

#: Side of the square model input of the released checkpoint.
MODEL_INPUT_SIZE = 224


def load_rgb(image: ImageLike) -> np.ndarray:
    """Convert any supported input to an ``(H, W, 3)`` ``uint8`` array.

    Accepted inputs:

    * a path to an image file (anything Pillow can open),
    * a :class:`PIL.Image.Image`,
    * a NumPy array shaped ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)`` (RGB) or
      ``(H, W, 4)`` (RGBA).  ``uint8`` is used as is; ``bool`` maps False/True
      to 0/255; other integers are scaled from their full range; floats are
      taken as 0..1 (or 0..255 if any value exceeds 1).

    Transparency is flattened onto white.
    """
    return np.asarray(_open(image).convert("RGB"), dtype=np.uint8)


def load_grayscale(image: ImageLike) -> np.ndarray:
    """Like :func:`load_rgb`, reduced to an ``(H, W)`` ``uint8`` array (ITU-R 601 luma)."""
    return np.asarray(_open(image).convert("L"), dtype=np.uint8)


def save_png(pixels: np.ndarray, path: Union[str, "os.PathLike[str]"]) -> Path:
    """Write an ``(H, W)`` or ``(H, W, 3)`` ``uint8`` array as a PNG and return the path."""
    path = Path(path)
    Image.fromarray(np.ascontiguousarray(pixels, dtype=np.uint8)).save(path, format="PNG")
    return path


# --------------------------------------------------------------------------- model input
@dataclass(frozen=True)
class CropTransform:
    """Mapping between *source image* pixels and *model input* pixels.

    Forward: crop the box whose top-left corner is ``(origin_x, origin_y)``,
    multiply by ``ratio``, add the centring padding ``(pad_x, pad_y)``.
    """

    ratio: float = 1.0
    pad_x: float = 0.0
    pad_y: float = 0.0
    origin_x: float = 0.0
    origin_y: float = 0.0

    def model_to_source(self, points: np.ndarray) -> np.ndarray:
        """Map ``(N, 2)`` model-frame points (pixel-edge convention) to source pixels."""
        pts = np.asarray(points, dtype=np.float64)
        return (pts - [self.pad_x, self.pad_y]) / self.ratio + [self.origin_x, self.origin_y]

    def source_to_model(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        return (pts - [self.origin_x, self.origin_y]) * self.ratio + [self.pad_x, self.pad_y]


def prepare_word(rgb: np.ndarray, box: Sequence[float] | None = None,
                 size: int = MODEL_INPUT_SIZE) -> Tuple[np.ndarray, CropTransform]:
    """Crop ``box`` and fit it, aspect ratio kept, centred on a black ``size`` square.

    This is the official notebook's ``load_and_pad_img`` / ``scale_and_pad``.

    Args:
        rgb: ``(H, W, 3)`` source image.
        box: ``(x0, y0, x1, y1)`` in pixels (truncated to ints, clipped to the
            image); None for the whole image.
        size: Side of the square model input.

    Returns:
        The ``(size, size, 3)`` model input and the transform back to ``rgb``.
    """
    height, width = rgb.shape[:2]
    if box is None:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
        x1, y1 = min(width, int(box[2])), min(height, int(box[3]))
    if x1 <= x0 or y1 <= y0:
        raise InvalidImageError(f"word box {tuple(box)} has no pixels inside the image")

    crop = Image.fromarray(rgb[y0:y1, x0:x1])
    ratio = min(size / crop.width, size / crop.height)
    new_w, new_h = max(1, int(crop.width * ratio)), max(1, int(crop.height * ratio))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    dx, dy = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(crop.resize((new_w, new_h)), (dx, dy))
    return np.asarray(canvas), CropTransform(ratio, dx, dy, x0, y0)


# --------------------------------------------------------------------------- helpers
def _open(image: ImageLike) -> Image.Image:
    """Any supported input as a PIL image with transparency flattened onto white."""
    if isinstance(image, (str, os.PathLike)):
        path = Path(image)
        if not path.is_file():
            raise InvalidImageError(f"image file not found: {path}")
        try:
            with Image.open(path) as opened:
                opened.load()
                return _flatten(opened)
        except OSError as exc:
            raise InvalidImageError(f"cannot read image file {path}: {exc}") from exc
    if isinstance(image, Image.Image):
        return _flatten(image)
    if isinstance(image, np.ndarray):
        return _flatten(_array_to_pil(image))
    raise InvalidImageError(
        f"unsupported image type {type(image).__name__}; "
        "expected a file path, a NumPy array or a PIL image"
    )


def _flatten(image: Image.Image) -> Image.Image:
    if image.width == 0 or image.height == 0:
        raise InvalidImageError(f"image has no pixels (size {image.size})")
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        paper = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(paper, rgba).convert("RGB")
    if image.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        return _array_to_pil(np.asarray(image))
    return image


def _array_to_pil(array: np.ndarray) -> Image.Image:
    """Validate an array's shape and bring its values to 8 bits."""
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if not (array.ndim == 2 or (array.ndim == 3 and array.shape[2] in (3, 4))):
        raise InvalidImageError(
            f"expected an array shaped (H, W), (H, W, 1), (H, W, 3) or (H, W, 4); got {array.shape}"
        )
    if array.size == 0:
        raise InvalidImageError(f"image has no pixels (shape {array.shape})")
    return Image.fromarray(np.ascontiguousarray(_to_uint8(array)))


def _to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scaled = (array.astype(np.float64) - info.min) / float(info.max - info.min)
        return np.rint(scaled * 255.0).astype(np.uint8)
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise InvalidImageError("image contains NaN or infinite values")
        scale = 255.0 if float(array.max(initial=0.0)) <= 1.0 else 1.0
        return np.rint(np.clip(array.astype(np.float64) * scale, 0.0, 255.0)).astype(np.uint8)
    raise InvalidImageError(f"unsupported array dtype {array.dtype}")
