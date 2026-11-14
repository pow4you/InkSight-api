"""Result objects returned by the pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from ..core.geometry import BoundingBox, CropTransform
from ..core.ink import Ink
from ..imaging.ops import draw_boxes
from ..models.inksight_model import ModelOutput


@dataclass(frozen=True)
class WordResult:
    """The outcome for a single word.

    Attributes:
        index: Position of the word in reading order.
        box: Where the word sits in the source image.
        ocr_text: Text the segmentor read there (``None`` if it reads nothing).
        model_input: The square image the model actually saw.
        transform: Source <-> model-frame mapping used for this word.
        output: Raw model output; its ``ink`` is in the model's 224 x 224 frame.
        ink: The same ink mapped back to *source image* pixels.
    """

    index: int
    box: BoundingBox
    ocr_text: str | None
    model_input: Image.Image
    transform: CropTransform
    output: ModelOutput
    ink: Ink

    @property
    def text(self) -> str | None:
        """Best available text: what the model recognised, else the OCR text."""
        return self.output.recognized_text or self.ocr_text


@dataclass
class PipelineResult:
    """The outcome for a whole image.

    Attributes:
        image: The source RGB image.
        words: One :class:`WordResult` per segmented word.
        elapsed_seconds: Wall-clock time of the run.
    """

    image: Image.Image
    words: list[WordResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def ink(self) -> Ink:
        """All words' ink merged, in source-image pixel coordinates."""
        return Ink.merge((w.ink for w in self.words), text=self.text or None)

    @property
    def text(self) -> str:
        """Words joined by spaces (empty string if there is no text)."""
        return " ".join(w.text for w in self.words if w.text)

    @property
    def boxes(self) -> list[BoundingBox]:
        return [w.box for w in self.words]

    def annotated_image(self, *, outline: str = "red", width: int = 2) -> Image.Image:
        """The source image with every word box drawn on it."""
        return draw_boxes(self.image, self.boxes, outline=outline, width=width)

    def plot(self, ax: Any | None = None, **kwargs: Any) -> Any:
        """Quick matplotlib overlay of the merged ink on the source image.

        Needs the ``viz`` extra. Keyword arguments go to
        :func:`inksight_api.visualization.plot_ink`.
        """
        from ..visualization.plotting import plot_ink
        return plot_ink(self.ink, ax, image=self.image, canvas_size=None, **kwargs)
