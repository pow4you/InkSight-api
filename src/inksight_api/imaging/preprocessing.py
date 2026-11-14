"""Turn an image (or a box inside one) into the model's square input.

This module replaces the notebook's four near-identical helpers
(``load_and_pad_img_dir``, ``load_and_pad_img``, ``scale_and_pad`` and
``rotate_crop_scale_and_pad``) with a single strategy class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image

from ..core.geometry import BoundingBox, CropTransform


class PadColor(Enum):
    """Colour of the padding that squares the image up."""

    BLACK = "black"                    # what the model was fed in the notebook
    WHITE = "white"                    # used for the visualisation backdrop
    CORNER_AVERAGE = "corner_average"  # mean of the four corner pixels


@dataclass(frozen=True)
class PreparedImage:
    """A model-ready image together with the way back to the source.

    Attributes:
        image: The square RGB image fed to the model.
        transform: Maps model-frame coordinates back to source pixels.
        source_box: The region of the source that was cropped.
    """

    image: Image.Image
    transform: CropTransform
    source_box: BoundingBox


class ImagePreprocessor(ABC):
    """Strategy interface: ``(image, optional box) -> PreparedImage``.

    Subclass it to experiment with other input conditioning (binarisation,
    deskewing, different padding ...) and hand the instance to a pipeline.
    """

    @abstractmethod
    def prepare(self, image: Image.Image, box: BoundingBox | None = None) -> PreparedImage:
        """Crop ``box`` (whole image if ``None``) and condition it for the model."""


class SquarePadPreprocessor(ImagePreprocessor):
    """Fit the crop inside ``size`` x ``size`` keeping aspect ratio, centre it on a padded canvas.

    Args:
        size: Side of the square model input (``224`` for InkSight).
        pad_color: Padding colour, see :class:`PadColor`.
    """

    def __init__(self, size: int = 224, pad_color: PadColor = PadColor.BLACK) -> None:
        self.size = size
        self.pad_color = pad_color

    def prepare(self, image: Image.Image, box: BoundingBox | None = None) -> PreparedImage:
        image = image.convert("RGB")
        if box is None:
            box = BoundingBox(0, 0, image.width, image.height)
            crop = image
        else:
            crop = image.crop(box.as_crop_box())

        ratio = min(self.size / crop.width, self.size / crop.height)
        new_w = max(1, int(crop.width * ratio))
        new_h = max(1, int(crop.height * ratio))
        resized = crop.resize((new_w, new_h))

        canvas = Image.new("RGB", (self.size, self.size), self._pad_rgb(resized))
        dx, dy = (self.size - new_w) // 2, (self.size - new_h) // 2
        canvas.paste(resized, (dx, dy))

        transform = CropTransform(ratio=ratio, pad_x=dx, pad_y=dy,
                                  origin_x=box.left, origin_y=box.top)
        return PreparedImage(canvas, transform, box)

    def _pad_rgb(self, resized: Image.Image) -> tuple[int, int, int]:
        if self.pad_color is PadColor.BLACK:
            return (0, 0, 0)
        if self.pad_color is PadColor.WHITE:
            return (255, 255, 255)
        arr = np.asarray(resized)
        h, w = arr.shape[:2]
        corners = [arr[min(1, h - 1), min(1, w - 1)], arr[min(1, h - 1), w - 1],
                   arr[h - 1, min(1, w - 1)], arr[h - 1, w - 1]]
        mean = np.rint(np.mean(corners, axis=0)).astype(np.uint8)
        return tuple(int(c) for c in mean)  # type: ignore[return-value]
