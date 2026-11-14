"""Exception hierarchy.

Everything this package raises on purpose derives from :class:`InkSightError`,
so callers can catch one type. Where it makes sense the exceptions *also*
derive from the matching builtin (``ValueError``, ``ImportError`` ...), so
existing ``except ValueError`` code keeps working.
"""

from __future__ import annotations


class InkSightError(Exception):
    """Base class for all errors raised by ``inksight_api``."""


class InkTokenError(InkSightError, ValueError):
    """An ink token id is outside the model vocabulary."""


class PromptError(InkSightError, ValueError):
    """A prompt cannot be built (e.g. *derender with text* but no text given)."""


class MissingDependencyError(InkSightError, ImportError):
    """An optional third-party dependency is not installed."""


class ModelLoadError(InkSightError, RuntimeError):
    """The model checkpoint could not be loaded."""


class SegmentationError(InkSightError, RuntimeError):
    """A word segmentor failed to produce boxes."""


class RegistryError(InkSightError, LookupError):
    """A component name is unknown or already taken in a :class:`Registry`."""
