"""inksight - handwriting images to pen strokes with InkSight, behind one small API.

The released InkSight model (Mitrevski et al., *InkSight: Offline-to-Online
Handwriting Conversion by Teaching Vision-Language Models to Read and Write*,
TMLR 2025) is bundled with the package and wrapped - not re-implemented - in
two methods:

* :class:`InkSightWord` - the image shows one word
* :class:`InkSightPage` - find the words (docTR, PaddleOCR or Tesseract), derender each

Quick start::

    import inksight

    drawing = inksight.vectorize("page.png", method="page")   # or a NumPy array
    for stroke in drawing:          # (N, 2) arrays of x, y pixel coordinates
        ...
    drawing.save("page.svg")

Class hierarchy::

    Vectorizer                 image in -> VectorDrawing out
    └── ModelVectorizer        ... via the InkSight model
        ├── InkSightWord
        └── InkSightPage       + WordDetector (Doctr / PaddleOCR / Tesseract)

    Runner                     where the model executes
    ├── LocalRunner            TensorFlow in this process
    └── DockerRunner           a container (image built automatically)
"""

from __future__ import annotations

import sys
from typing import Any

from . import worker as _worker

# The checkpoint needs XLA flags that only work if set before TensorFlow's first
# import - and docTR imports TensorFlow too.  Set them now if it is not too late.
if "tensorflow" not in sys.modules:
    _worker.configure_tensorflow()

from .base import Vectorizer, available_methods, get_vectorizer, get_vectorizer_class  # noqa: E402
from .drawing import VectorDrawing  # noqa: E402
from .errors import (  # noqa: E402
    BackendBuildError,
    BackendNotAvailableError,
    InksightError,
    InvalidImageError,
    MissingDependencyError,
    VectorizationError,
)
from .images import ImageLike, load_grayscale, load_rgb  # noqa: E402
from .methods import InkSightPage, InkSightWord  # noqa: E402
from .model import ModelJob, ModelVectorizer  # noqa: E402
from .prompts import InferenceMode  # noqa: E402
from .runners import DockerRunner, LocalRunner, Runner, RunResult, get_runner  # noqa: E402
from .words import (  # noqa: E402
    DoctrWordDetector,
    PaddleOCRWordDetector,
    TesseractWordDetector,
    WholeImageDetector,
    WordDetector,
    Words,
    available_detectors,
    get_detector,
)

__version__ = "0.2.0"

__all__ = [
    "vectorize",
    # vectorizers
    "Vectorizer", "ModelVectorizer", "ModelJob", "InkSightWord", "InkSightPage", "InferenceMode",
    "available_methods", "get_vectorizer", "get_vectorizer_class",
    # results and inputs
    "VectorDrawing", "ImageLike", "load_rgb", "load_grayscale",
    # word detection
    "WordDetector", "Words", "DoctrWordDetector", "PaddleOCRWordDetector",
    "TesseractWordDetector", "WholeImageDetector", "available_detectors", "get_detector",
    # runners
    "Runner", "RunResult", "LocalRunner", "DockerRunner", "get_runner",
    # errors
    "InksightError", "InvalidImageError", "MissingDependencyError",
    "BackendNotAvailableError", "BackendBuildError", "VectorizationError",
]


def vectorize(image: ImageLike, method: str = "page", **options: Any) -> VectorDrawing:
    """Turn an image of handwriting into strokes in one call.

    Args:
        image: Path to an image file, a NumPy array or a PIL image.
        method: ``"page"`` (find the words, derender each) or ``"word"`` (the
            image is one word); see :func:`available_methods`.
        **options: Constructor options of the chosen vectorizer, e.g.
            ``runner="docker"``, ``mode="vanilla"`` or ``word_detector="paddleocr"``.

    Returns:
        The strokes as a :class:`VectorDrawing`.

    For repeated use, create the vectorizer once (``InkSightPage(...)``) and
    call its ``vectorize`` method; that keeps the model and the word detector loaded.
    """
    return get_vectorizer(method, **options).vectorize(image)
