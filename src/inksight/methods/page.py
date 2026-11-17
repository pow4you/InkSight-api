"""Derender a full page: find the words, derender each, put them back together."""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Union

import numpy as np

from ..drawing import VectorDrawing
from ..images import ImageLike, load_rgb
from ..model import ModelJob, ModelVectorizer
from ..words import WordDetector, Words, get_detector
from .word import REFERENCE

#: Forms in which word boxes may be supplied to :meth:`InkSightPage.vectorize`.
WordsLike = Union[Words, str, "os.PathLike[str]", np.ndarray, Sequence[Any]]


class InkSightPage(ModelVectorizer):
    """Full-page handwriting to strokes, as in the official notebook.

    Works in two stages:

    1. a :class:`~inksight.words.WordDetector` finds the words - docTR by
       default, as in the notebook; PaddleOCR (``"paddleocr"``) is a newer
       alternative, Tesseract (``"tesseract"``) a lightweight one;
    2. the InkSight model derenders every word, and the strokes are mapped
       back into the page.

    Both stages happen inside :meth:`vectorize`; the boxes used are returned
    in ``drawing.metadata["words"]``.

    Example::

        from inksight import InkSightPage

        page = InkSightPage(word_detector="paddleocr")
        drawing = page.vectorize("page.jpg")             # finds the words itself
        drawing = page.vectorize("page.jpg", words="page.words.json")   # or reuse some

    Args:
        word_detector: A :class:`~inksight.words.WordDetector` or a registered
            name (``"doctr"``, ``"paddleocr"``, ``"tesseract"``, ``"whole"``).
            Engines load lazily, on the first detection.
        **options: See :class:`~inksight.model.ModelVectorizer`
            (``runner``, ``mode``, ``fallback_mode``, ``batch_size``,
            ``timeout``, ``workdir``, ``invert``).
    """

    name = "page"
    aliases = ("inksight", "inksight-page")
    title = "InkSight (full page)"
    reference = REFERENCE

    def __init__(self, *, word_detector: Union[str, WordDetector] = "doctr", **options: Any) -> None:
        super().__init__(**options)
        self.word_detector = get_detector(word_detector)

    def vectorize(self, image: ImageLike, words: Optional[WordsLike] = None) -> VectorDrawing:
        """Derender every word of one page.

        Args:
            image: Path to an image file, a NumPy array or a PIL image.
            words: Word boxes to use instead of running the detector: a
                :class:`~inksight.words.Words`, the path of a JSON file written
                by :meth:`Words.save <inksight.words.Words.save>`, or an
                ``(N, 4)`` array of ``x0, y0, x1, y1`` rows.

        Returns:
            The strokes; ``metadata["words"]`` holds per-word details and
            ``metadata["text"]`` the recognised text.
        """
        return super().vectorize(image, words=words)

    def detect_words(self, image: ImageLike) -> Words:
        """Run only stage 1 - e.g. to inspect or edit the boxes before passing
        them back to :meth:`vectorize`.

        The image goes through the same preprocessing as in :meth:`vectorize`.
        """
        return self.word_detector.detect(self.preprocess(load_rgb(image)))

    def _stage(self, rgb: np.ndarray, words: Optional[WordsLike] = None, **inputs: Any) -> ModelJob:
        if inputs:
            raise TypeError(f"{type(self).__name__}.vectorize() got unexpected "
                            f"argument(s): {', '.join(sorted(inputs))}")
        boxes = self.word_detector.detect(rgb) if words is None else Words.coerce(words)
        return ModelJob(boxes, metadata={"detector": getattr(self.word_detector, "name", "")
                                         if words is None else "given"})

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(runner={self._runner_spec!r}, mode={self.mode.alias!r}, "
                f"word_detector={self.word_detector!r})")
