"""Built-in segmentors: Tesseract, docTR and the trivial whole-image one."""

from __future__ import annotations

from typing import Any

from PIL import Image

from ..core.exceptions import SegmentationError
from ..core.geometry import BoundingBox
from ..utils.lazy import require
from .base import WordBox, WordSegmenter, segmenters


@segmenters.register("whole", "word")
class WholeImageSegmenter(WordSegmenter):
    """Treat the entire image as one word (the notebook's *word-level* flow)."""

    def segment(self, image: Image.Image) -> list[WordBox]:
        return [WordBox(BoundingBox(0, 0, image.width, image.height))]


@segmenters.register("tesseract")
class TesseractSegmenter(WordSegmenter):
    """Word boxes from Tesseract OCR (needs ``pytesseract`` and the system binary).

    Args:
        box_expansion: Grow each box by this fraction (the notebook uses none).
        min_confidence: Drop words below this Tesseract confidence (``None`` keeps all).
        lang: Tesseract language(s), e.g. ``"eng+deu"``.
        config: Extra Tesseract command line options.
    """

    def __init__(self, box_expansion: float = 0.0, min_confidence: float | None = None,
                 lang: str | None = None, config: str = "") -> None:
        self.box_expansion = box_expansion
        self.min_confidence = min_confidence
        self.lang = lang
        self.config = config

    def segment(self, image: Image.Image) -> list[WordBox]:
        pytesseract = require("pytesseract", extra="tesseract", purpose=" for Tesseract segmentation")
        try:
            data = pytesseract.image_to_data(
                image, lang=self.lang, config=self.config,
                output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractNotFoundError as exc:  # pragma: no cover - env specific
            raise SegmentationError(
                "The tesseract binary was not found (apt install tesseract-ocr)"
            ) from exc

        boxes: list[WordBox] = []
        for i, text in enumerate(data["text"]):
            if not text.strip():
                continue
            conf = float(data["conf"][i])
            if self.min_confidence is not None and conf < self.min_confidence:
                continue
            box = BoundingBox(data["left"][i], data["top"][i], data["width"][i], data["height"][i])
            if self.box_expansion:
                box = box.expanded(self.box_expansion, bounds=image.size).to_int()
            boxes.append(WordBox(box, text=text, confidence=conf))
        return boxes


@segmenters.register("doctr")
class DoctrSegmenter(WordSegmenter):
    """Word boxes from docTR (``python-doctr[tf]``).

    Args:
        box_expansion: Grow each box by this fraction on every side (``0.1`` as in the notebook).
        predictor: A ready docTR ``ocr_predictor``. If omitted one is built
            lazily (and cached) with pretrained weights.
    """

    def __init__(self, box_expansion: float = 0.1, predictor: Any | None = None) -> None:
        self.box_expansion = box_expansion
        self._predictor = predictor

    @property
    def predictor(self) -> Any:
        """The docTR predictor, created on first use."""
        if self._predictor is None:
            # docTR runs on TensorFlow too, so the notebook's XLA flags apply.
            from ..models.tf_runtime import configure_tensorflow
            configure_tensorflow()
            models = require("doctr.models", extra="doctr", purpose=" for docTR segmentation")
            self._predictor = models.ocr_predictor(pretrained=True)
        return self._predictor

    def segment(self, image: Image.Image) -> list[WordBox]:
        import numpy as np

        width, height = image.size
        result = self.predictor([np.asarray(image.convert("RGB"))])

        boxes: list[WordBox] = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        if not word.value.strip():
                            continue
                        (rx0, ry0), (rx1, ry1) = word.geometry
                        box = BoundingBox.from_xyxy(
                            int(rx0 * width), int(ry0 * height),
                            int(rx1 * width), int(ry1 * height),
                        )
                        box = box.expanded(self.box_expansion, bounds=(width, height)).to_int()
                        boxes.append(WordBox(box, text=word.value,
                                             confidence=getattr(word, "confidence", None)))
        return boxes
