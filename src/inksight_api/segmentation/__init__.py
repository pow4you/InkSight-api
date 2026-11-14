"""Word segmentation: interface, registry and the built-in engines."""

from .base import WordBox, WordSegmenter, segmenters
from .builtin import DoctrSegmenter, TesseractSegmenter, WholeImageSegmenter

__all__ = [
    "DoctrSegmenter", "TesseractSegmenter", "WholeImageSegmenter",
    "WordBox", "WordSegmenter", "segmenters",
]
