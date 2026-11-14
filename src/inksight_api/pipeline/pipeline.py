"""The conversion pipeline: segment -> prepare -> derender -> map back.

:class:`InkPipeline` is a *template method*: :meth:`~InkPipeline.run` fixes the
order of the stages and each stage is a small overridable hook. The word-level
and full-page flows from the notebook are just two configurations of it
(:class:`WordInkPipeline` and :class:`PageInkPipeline`).
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Sequence

from PIL import Image

from ..core.ink import Ink
from ..core.prompts import InferenceMode
from ..imaging.loading import ImageLike, load_image
from ..imaging.preprocessing import ImagePreprocessor, PreparedImage, SquarePadPreprocessor
from ..models.inksight_model import InkSightModel, ModelOutput
from ..segmentation.base import WordBox, WordSegmenter, segmenters
from ..segmentation.builtin import WholeImageSegmenter
from .results import PipelineResult, WordResult

logger = logging.getLogger(__name__)


class InkPipeline:
    """Convert an image into digital ink with pluggable stages.

    Args:
        model: The :class:`InkSightModel` to run.
        segmenter: Finds the words (a :class:`WordSegmenter` or a registered name).
        preprocessor: Conditions each word for the model
            (default: :class:`SquarePadPreprocessor`, 224 px, black padding).
        mode: Inference mode; defaults to the model's ``default_mode``.

    Override any of :meth:`_segment`, :meth:`_prepare`, :meth:`_infer` or
    :meth:`_to_source` to change one stage without touching the others.
    """

    def __init__(self, model: InkSightModel, segmenter: WordSegmenter | str,
                 preprocessor: ImagePreprocessor | None = None,
                 mode: InferenceMode | str | None = None) -> None:
        self.model = model
        self.segmenter = segmenters.create(segmenter) if isinstance(segmenter, str) else segmenter
        self.preprocessor = preprocessor or SquarePadPreprocessor()
        self.mode = InferenceMode.from_alias(mode) if mode is not None else model.default_mode

    # -- public API -------------------------------------------------------
    def run(self, image: ImageLike, *, texts: Sequence[str | None] | None = None) -> PipelineResult:
        """Run the full pipeline on one image.

        Args:
            image: Path, bytes or PIL image.
            texts: Optional text per segmented word that overrides the
                segmentor's own OCR text (needed for *derender with text* when
                the segmentor reads no text).

        Returns:
            A :class:`PipelineResult` (empty ``words`` if nothing was found).
        """
        started = time.perf_counter()
        source = load_image(image)

        word_boxes = self._segment(source)
        if texts is not None:
            if len(texts) != len(word_boxes):
                raise ValueError(f"got {len(texts)} texts for {len(word_boxes)} segmented words")
            word_boxes = [dataclasses.replace(wb, text=t) for wb, t in zip(word_boxes, texts)]
        logger.info("segmented %d word(s)", len(word_boxes))

        prepared = [self._prepare(source, wb) for wb in word_boxes]
        outputs = self._infer(prepared, word_boxes)

        words = [
            WordResult(i, wb.box, wb.text, prep.image, prep.transform, out,
                       self._to_source(out.ink, prep))
            for i, (wb, prep, out) in enumerate(zip(word_boxes, prepared, outputs))
        ]
        return PipelineResult(source, words, time.perf_counter() - started)

    # -- overridable stages -----------------------------------------------
    def _segment(self, image: Image.Image) -> list[WordBox]:
        return self.segmenter.segment(image)

    def _prepare(self, image: Image.Image, word_box: WordBox) -> PreparedImage:
        return self.preprocessor.prepare(image, word_box.box)

    def _infer(self, prepared: Sequence[PreparedImage], word_boxes: Sequence[WordBox]) -> list[ModelOutput]:
        return self.model.derender(
            [p.image for p in prepared],
            mode=self.mode,
            texts=[wb.text for wb in word_boxes] if self.mode.needs_text else None,
        )

    def _to_source(self, ink: Ink, prepared: PreparedImage) -> Ink:
        return prepared.transform.ink_to_source(ink)


class WordInkPipeline(InkPipeline):
    """Word-level flow: the whole image *is* one word (no segmentation)."""

    def __init__(self, model: InkSightModel, preprocessor: ImagePreprocessor | None = None,
                 mode: InferenceMode | str | None = None) -> None:
        super().__init__(model, WholeImageSegmenter(), preprocessor, mode)

    def run(self, image: ImageLike, *, text: str | None = None) -> PipelineResult:  # type: ignore[override]
        """Convert one word image. ``text`` is only needed for *derender with text*."""
        return super().run(image, texts=[text] if text is not None else None)


class PageInkPipeline(InkPipeline):
    """Full-page flow: segment into words, derender each, stitch the result.

    Args:
        model: The model to run.
        segmenter: A :class:`WordSegmenter` or a registered name
            (``"tesseract"`` or ``"doctr"``).
        preprocessor: See :class:`InkPipeline`.
        mode: See :class:`InkPipeline`.
    """

    def __init__(self, model: InkSightModel, segmenter: WordSegmenter | str = "tesseract",
                 preprocessor: ImagePreprocessor | None = None,
                 mode: InferenceMode | str | None = None) -> None:
        super().__init__(model, segmenter, preprocessor, mode)
