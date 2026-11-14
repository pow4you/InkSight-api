"""The one-stop entry point for typical use."""

from __future__ import annotations

from .core.prompts import InferenceMode
from .imaging.loading import ImageLike
from .imaging.preprocessing import ImagePreprocessor
from .models.inksight_model import InkSightModel
from .pipeline.pipeline import PageInkPipeline, WordInkPipeline
from .pipeline.results import PipelineResult
from .segmentation.base import WordSegmenter, segmenters


class InkSight:
    """Facade over model + pipelines for the two common jobs.

    Example:
        >>> kit = InkSight.from_pretrained(segmenter="doctr")   # doctest: +SKIP
        >>> page = kit.convert_page("page.jpg")                 # doctest: +SKIP
        >>> page.ink, page.text                                 # doctest: +SKIP

    Args:
        model: A ready :class:`InkSightModel`.
        segmenter: Default segmentor for :meth:`convert_page` (name or instance).
            A name is only instantiated on first use, so OCR dependencies stay
            optional for word-level work.
        preprocessor: Optional replacement for the default square-pad preprocessing.
    """

    def __init__(self, model: InkSightModel, *, segmenter: WordSegmenter | str = "tesseract",
                 preprocessor: ImagePreprocessor | None = None) -> None:
        self.model = model
        self.segmenter = segmenter
        self.preprocessor = preprocessor
        self._segmenter_cache: dict[str, WordSegmenter] = {}

    @classmethod
    def from_pretrained(cls, source: str | None = None, *, segmenter: WordSegmenter | str = "tesseract",
                        backend: str = "tf-saved-model",
                        preprocessor: ImagePreprocessor | None = None, **backend_kwargs) -> "InkSight":
        """Load the model and wrap it.

        Args:
            source: A ``.zip`` path or URL, a SavedModel directory, or a Hugging
                Face repository id. ``None`` downloads the authors' ``small-p-cpu.zip``.
            segmenter: Default segmentor for :meth:`convert_page`.
            backend: Registered backend name.
            preprocessor: Optional replacement for the default preprocessing.
            **backend_kwargs: ``batch_size``, ``cache_dir`` ... for the backend.
        """
        model = InkSightModel.from_pretrained(source, backend=backend, **backend_kwargs)
        return cls(model, segmenter=segmenter, preprocessor=preprocessor)

    def convert_word(self, image: ImageLike, *, text: str | None = None,
                     mode: InferenceMode | str | None = None) -> PipelineResult:
        """Convert a single-word image. ``text`` is only needed for *derender with text*."""
        return WordInkPipeline(self.model, self.preprocessor, mode).run(image, text=text)

    def convert_page(self, image: ImageLike, *, segmenter: WordSegmenter | str | None = None,
                     mode: InferenceMode | str | None = None) -> PipelineResult:
        """Convert a full-page image: segment into words, derender each, stitch together."""
        chosen = self._resolve_segmenter(segmenter if segmenter is not None else self.segmenter)
        return PageInkPipeline(self.model, chosen, self.preprocessor, mode).run(image)

    def _resolve_segmenter(self, spec: WordSegmenter | str) -> WordSegmenter:
        """Instances pass through; names are built once and cached (docTR loads weights)."""
        if isinstance(spec, WordSegmenter):
            return spec
        key = spec.strip().lower()
        if key not in self._segmenter_cache:
            self._segmenter_cache[key] = segmenters.create(key)
        return self._segmenter_cache[key]
