"""Model backend interface.

A *backend* is the thinnest possible adapter around "something that turns
``(image, prompt)`` into decoder text". Everything model-agnostic (prompts,
tokenizing, fallback policy) lives one level up in
:class:`~inksight_api.models.InkSightModel`, so supporting a new runtime
(ONNX, a remote endpoint, a stub for tests ...) means implementing one method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from PIL import Image

from ..registry import Registry


class ModelBackend(ABC):
    """Run the network. No pre- or post-processing happens here."""

    @abstractmethod
    def generate(self, images: Sequence[Image.Image], prompts: Sequence[str]) -> list[str]:
        """Return the raw decoder text for every ``(image, prompt)`` pair.

        Args:
            images: Square RGB model inputs (already padded to the model size).
            prompts: One prompt per image, same length as ``images``.

        Returns:
            One string per input, in order, containing ``<ink_token_N>`` markers.
        """


#: Registry of backends (``"tf-saved-model"`` is built in).
backends: Registry[ModelBackend] = Registry("backend")
