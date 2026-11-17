"""The root of the class hierarchy: :class:`Vectorizer` and the method registry.

::

    Vectorizer                      image in, VectorDrawing out          (this module)
    └── ModelVectorizer             ... by running the InkSight model    (inksight.model)
        ├── InkSightWord            the image is one word                (inksight.methods.word)
        └── InkSightPage            find the words, derender each        (inksight.methods.page)

To add a method, subclass whichever level fits, give the class a ``name`` and
implement the one abstract method of that level; the class registers itself
and becomes available through :func:`get_vectorizer` and the command line.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Tuple, Type

import numpy as np

from .drawing import VectorDrawing
from .images import ImageLike, load_rgb


class Vectorizer(ABC):
    """Turns a raster image of handwriting into vector strokes.

    The public entry point, :meth:`vectorize`, is a template method::

        image (file / array / PIL)
          -> load      8-bit RGB, dark ink on light paper       (fixed)
          -> preprocess                                         (hook, overridable)
          -> _vectorize                                         (abstract)
          -> postprocess                                        (hook, overridable)
          -> VectorDrawing

    Subclasses implement :meth:`_vectorize`; they may refine the two hooks.

    Class attributes for subclasses:
        name: Registry key.  A subclass that sets it is registered automatically.
        aliases: Additional registry keys.
        title: Human-readable method name.
        reference: The paper to cite.

    Args:
        invert: Set for light-on-dark images (chalkboard style); the image is
            inverted before processing so that ink is dark.
    """

    name: ClassVar[str] = ""
    aliases: ClassVar[Tuple[str, ...]] = ()
    title: ClassVar[str] = ""
    reference: ClassVar[str] = ""

    _registry: ClassVar[Dict[str, Type["Vectorizer"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only classes that declare their own name are concrete, selectable methods.
        if cls.__dict__.get("name"):
            for key in (cls.name, *cls.__dict__.get("aliases", ())):
                Vectorizer._registry[key.lower()] = cls

    def __init__(self, *, invert: bool = False) -> None:
        self.invert = invert

    # ------------------------------------------------------------- public API
    def vectorize(self, image: ImageLike, **inputs: Any) -> VectorDrawing:
        """Vectorize one image.

        Args:
            image: Path to an image file, a NumPy array (``(H, W)`` gray,
                ``(H, W, 3)`` RGB or ``(H, W, 4)`` RGBA) or a PIL image.
            **inputs: Additional per-image inputs understood by the concrete
                method (for example precomputed word boxes).

        Returns:
            The strokes, in pixel coordinates of the input image.
        """
        rgb = self.preprocess(load_rgb(image))
        drawing = self._vectorize(rgb, **inputs)
        drawing.method = drawing.method or self.name
        return self.postprocess(drawing)

    def __call__(self, image: ImageLike, **inputs: Any) -> VectorDrawing:
        """Alias for :meth:`vectorize`."""
        return self.vectorize(image, **inputs)

    # ------------------------------------------------------------------ hooks
    def preprocess(self, rgb: np.ndarray) -> np.ndarray:
        """Adjust the RGB image before vectorization (default: optional inversion)."""
        return 255 - rgb if self.invert else rgb

    def postprocess(self, drawing: VectorDrawing) -> VectorDrawing:
        """Adjust the result before it is returned (default: nothing)."""
        return drawing

    @abstractmethod
    def _vectorize(self, rgb: np.ndarray, **inputs: Any) -> VectorDrawing:
        """Vectorize a preprocessed ``(H, W, 3)`` ``uint8`` image (dark ink on light paper)."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# --------------------------------------------------------------------------- registry
def available_methods() -> List[str]:
    """Primary names of all registered vectorization methods."""
    return sorted({cls.name for cls in Vectorizer._registry.values()})


def get_vectorizer_class(name: str) -> Type[Vectorizer]:
    """Look up a vectorizer class by registry name or alias (case-insensitive)."""
    try:
        return Vectorizer._registry[name.lower()]
    except KeyError:
        known = ", ".join(sorted(Vectorizer._registry))
        raise ValueError(f"unknown vectorization method {name!r}; known: {known}") from None


def get_vectorizer(name: str, **options: Any) -> Vectorizer:
    """Instantiate a vectorizer by name, e.g. ``get_vectorizer("page", word_detector="paddleocr")``."""
    return get_vectorizer_class(name)(**options)
