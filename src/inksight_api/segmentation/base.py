"""Word segmentor interface.

A *segmentor* answers one question: "where are the words on this page?".
InkSight itself works on single words, so full-page conversion is
``segment -> derender every word -> stitch back``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

from ..core.geometry import BoundingBox
from ..registry import Registry


@dataclass(frozen=True)
class WordBox:
    """A word location, optionally with the text an OCR engine read there.

    Attributes:
        box: Pixel box in the source image.
        text: OCR text (used by the *derender with text* mode).
        confidence: OCR confidence if the engine provides one.
    """

    box: BoundingBox
    text: str | None = None
    confidence: float | None = None


class WordSegmenter(ABC):
    """Find word boxes in a page image.

    To add your own engine subclass this, implement :meth:`segment` and
    register it::

        @segmenters.register("my-engine")
        class MyEngine(WordSegmenter): ...
    """

    @abstractmethod
    def segment(self, image: Image.Image) -> list[WordBox]:
        """Return the word boxes of an RGB ``image`` in reading order."""

    def __call__(self, image: Image.Image) -> list[WordBox]:
        return self.segment(image)


#: Registry of available segmentors (``"tesseract"``, ``"doctr"``, ``"whole"``).
segmenters: Registry[WordSegmenter] = Registry("segmenter")
