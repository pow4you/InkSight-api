"""Derender an image that shows a single word (InkSight's native task)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..drawing import VectorDrawing
from ..images import ImageLike
from ..model import ModelJob, ModelVectorizer
from ..words import Words

REFERENCE = ("B. Mitrevski, A. Rak, J. Schnitzler, C. Li, A. Maksai, J. Berent and C. Musat. "
             "InkSight: Offline-to-Online Handwriting Conversion by Teaching Vision-Language "
             "Models to Read and Write. Transactions on Machine Learning Research, 2025.")


class InkSightWord(ModelVectorizer):
    """The whole image is one word; it goes to the model as it is.

    Wraps the authors' released model (InkSight-Small-p, bundled with the
    package) and the official notebook's word-level flow: pad the image to the
    model's square input, derender, map the strokes back.

    Example::

        from inksight import InkSightWord

        drawing = InkSightWord(mode="vanilla").vectorize("word.png")
        drawing.save("word.svg")

    Args:
        **options: See :class:`~inksight.model.ModelVectorizer`
            (``runner``, ``mode``, ``fallback_mode``, ``batch_size``,
            ``timeout``, ``workdir``, ``invert``).
    """

    name = "word"
    aliases = ("inksight-word",)
    title = "InkSight (single word)"
    reference = REFERENCE

    def vectorize(self, image: ImageLike, text: Optional[str] = None) -> VectorDrawing:
        """Derender one word image.

        Args:
            image: Path to an image file, a NumPy array or a PIL image.
            text: The word's text; needed (only) for the ``"d+t"`` mode.
        """
        return super().vectorize(image, text=text)

    def _stage(self, rgb: np.ndarray, text: Optional[str] = None, **inputs: Any) -> ModelJob:
        if inputs:
            raise TypeError(f"{type(self).__name__}.vectorize() got unexpected "
                            f"argument(s): {', '.join(sorted(inputs))}")
        height, width = rgb.shape[:2]
        return ModelJob(Words.whole(width, height, text))
