"""Core, dependency-light building blocks: data model, geometry, prompts, tokens."""

from .exceptions import (
    InkSightError,
    InkTokenError,
    MissingDependencyError,
    ModelLoadError,
    PromptError,
    RegistryError,
    SegmentationError,
)
from .geometry import BoundingBox, CropTransform
from .ink import Ink, Stroke
from .prompts import InferenceMode
from .tokens import InkTokenizer

__all__ = [
    "BoundingBox", "CropTransform", "InferenceMode", "Ink", "InkSightError",
    "InkTokenError", "InkTokenizer", "MissingDependencyError", "ModelLoadError",
    "PromptError", "RegistryError", "SegmentationError", "Stroke",
]
