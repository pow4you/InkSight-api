"""inksight_api: an object-oriented, extensible API around Google Research's InkSight.

Layers (each depends only on the ones above it)::

    core            Ink / Stroke, geometry, prompts, tokenizer      (NumPy only)
    imaging         loading + model-input preprocessing             (Pillow)
    segmentation    word finders: tesseract, doctr, whole-image     (optional OCR deps)
    models          backend interface, TF backend, InkSightModel    (optional TF)
    pipeline        segment -> prepare -> derender -> map back
    visualization   renderers, plots, animation                     (optional matplotlib)
    io, datasets    InkML / SVG formats, supplementary dataset
    InkSight        facade for the common cases

Heavy third-party packages are imported lazily, so ``import inksight_api`` is cheap
and the data model works without TensorFlow or matplotlib installed.

The original notebook code is Copyright 2024 Google LLC (Apache-2.0); see NOTICE.
"""

from __future__ import annotations

from .core import (
    BoundingBox, CropTransform, InferenceMode, Ink, InkSightError, InkTokenError, InkTokenizer,
    MissingDependencyError, ModelLoadError, PromptError, RegistryError, SegmentationError, Stroke,
)
from .facade import InkSight
from .imaging import PadColor, SquarePadPreprocessor
from .io import read_ink, write_ink
from .models import InkSightModel, ModelBackend, ModelOutput
from .pipeline import InkPipeline, PageInkPipeline, PipelineResult, WordInkPipeline, WordResult
from .segmentation import WordBox, WordSegmenter

__version__ = "0.1.0"

__all__ = [
    "BoundingBox", "CropTransform", "InferenceMode", "Ink", "InkPipeline", "InkSight",
    "InkSightError", "InkSightModel", "InkTokenError", "InkTokenizer", "MissingDependencyError",
    "ModelBackend", "ModelLoadError", "ModelOutput", "PadColor", "PageInkPipeline",
    "PipelineResult", "PromptError", "RegistryError", "SegmentationError", "SquarePadPreprocessor",
    "Stroke", "WordBox", "WordInkPipeline", "WordResult", "WordSegmenter", "read_ink", "write_ink",
    "__version__",
]
