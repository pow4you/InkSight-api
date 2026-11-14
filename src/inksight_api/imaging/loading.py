"""Accept images as paths, bytes or PIL objects."""

from __future__ import annotations

import io
import os
from typing import Union

from PIL import Image

#: Everything :func:`load_image` understands.
ImageLike = Union[str, "os.PathLike[str]", bytes, bytearray, Image.Image]


def load_image(source: ImageLike, *, mode: str = "RGB") -> Image.Image:
    """Open ``source`` as a PIL image converted to ``mode``.

    Args:
        source: A file path, raw encoded bytes, or an existing PIL image.
        mode: Target colour mode. The model only sees RGB, so that is the default.

    Raises:
        TypeError: For any other type.
    """
    if isinstance(source, (bytes, bytearray)):
        image = Image.open(io.BytesIO(source))
    elif isinstance(source, (str, os.PathLike)):
        image = Image.open(source)
    elif isinstance(source, Image.Image):
        image = source
    else:
        raise TypeError("image source must be a path, bytes or PIL.Image")
    return image.convert(mode) if image.mode != mode else image
