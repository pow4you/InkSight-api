"""Exception hierarchy.  Everything inksight raises derives from :class:`InksightError`."""

from __future__ import annotations


class InksightError(Exception):
    """Base class of all inksight errors."""


class InvalidImageError(InksightError, ValueError):
    """The input could not be interpreted as an image."""


class MissingDependencyError(InksightError, ImportError):
    """An optional Python dependency (e.g. TensorFlow or an OCR engine) is not installed."""


class BackendNotAvailableError(InksightError):
    """No runner is able to execute the model."""


class BackendBuildError(InksightError):
    """Preparing a backend (bundling the model, building the Docker image) failed."""


class VectorizationError(InksightError):
    """The model ran but did not produce a usable result.

    Attributes:
        log: Everything the model program printed, for diagnosis.
    """

    def __init__(self, message: str, log: str = "") -> None:
        super().__init__(message)
        self.log = log
