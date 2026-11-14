"""Boxes and the invertible crop -> scale -> pad transform.

The notebook threads a bare tuple ``(ratio, dx, dy, min_x, min_y, angle)``
from the pre-processing step to the post-processing step. :class:`CropTransform`
names those numbers and owns both directions of the mapping, so pre- and
post-processing can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ink import Ink, PointArray


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in pixel coordinates (``top`` grows downwards)."""

    left: float
    top: float
    width: float
    height: float

    @classmethod
    def from_xyxy(cls, x0: float, y0: float, x1: float, y1: float) -> "BoundingBox":
        """Build from two opposite corners."""
        return cls(x0, y0, x1 - x0, y1 - y0)

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    def expanded(self, fraction: float, bounds: tuple[int, int] | None = None) -> "BoundingBox":
        """Grow every side by ``fraction`` of the box width / height.

        Args:
            fraction: ``0.1`` adds 10 % of the width on the left *and* right
                and 10 % of the height on top and bottom.
            bounds: Optional ``(image_width, image_height)`` to clip to.
        """
        dw, dh = self.width * fraction, self.height * fraction
        x0, y0 = self.left - dw, self.top - dh
        x1, y1 = self.right + dw, self.bottom + dh
        if bounds is not None:
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(bounds[0], x1), min(bounds[1], y1)
        return BoundingBox.from_xyxy(x0, y0, x1, y1)

    def to_int(self) -> "BoundingBox":
        """Truncate all four numbers to ``int`` (the notebook's ``map(int, ...)``)."""
        return BoundingBox(int(self.left), int(self.top), int(self.width), int(self.height))

    def as_crop_box(self) -> tuple[int, int, int, int]:
        """``(left, top, right, bottom)`` as ints, the argument of ``PIL.Image.crop``."""
        return int(self.left), int(self.top), int(self.right), int(self.bottom)


@dataclass(frozen=True)
class CropTransform:
    """Mapping between *source image* pixels and *model input* pixels.

    The forward direction is: crop the box whose top-left corner is
    ``(origin_x, origin_y)``, multiply by ``ratio``, then add the centring
    padding ``(pad_x, pad_y)``. The model emits ink in the padded frame;
    :meth:`model_to_source` undoes the three steps.
    """

    ratio: float = 1.0
    pad_x: float = 0.0
    pad_y: float = 0.0
    origin_x: float = 0.0
    origin_y: float = 0.0

    def source_to_model(self, points: PointArray) -> PointArray:
        """Map ``(N, 2)`` source-image points into the model input frame."""
        pts = np.asarray(points, float)
        return (pts - [self.origin_x, self.origin_y]) * self.ratio + [self.pad_x, self.pad_y]

    def model_to_source(self, points: PointArray) -> PointArray:
        """Map ``(N, 2)`` model-frame points back to source-image pixels.

        This is the notebook's ``unpad_unscale_unrotate_uncrop`` (its rotation
        angle was hard-coded to 0, so it is not modelled here).
        """
        pts = np.asarray(points, float)
        return (pts - [self.pad_x, self.pad_y]) / self.ratio + [self.origin_x, self.origin_y]

    def ink_to_source(self, ink: Ink) -> Ink:
        """Map a whole :class:`Ink` from the model frame to source pixels."""
        return ink.transformed(self.model_to_source)
