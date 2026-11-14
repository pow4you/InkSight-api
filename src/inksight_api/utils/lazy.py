"""Lazy imports of optional dependencies with actionable error messages."""

from __future__ import annotations

import importlib
from types import ModuleType

from ..core.exceptions import MissingDependencyError


def require(module: str, *, extra: str | None = None, purpose: str = "") -> ModuleType:
    """Import ``module`` or raise :class:`MissingDependencyError`.

    Importing heavy optional packages (TensorFlow, docTR, matplotlib ...) only
    when they are actually needed keeps ``import inksight_api`` instant and
    lets people use the data model without any of them installed.

    Args:
        module: Dotted module path, e.g. ``"pytesseract"``.
        extra: Name of the ``inksight-api`` extra that provides it.
        purpose: Short phrase appended to the error (``" to draw ink"``).

    Raises:
        MissingDependencyError: If the import fails. The original
            ``ImportError`` is chained so real import problems stay visible.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        hint = f' Install it with: pip install "inksight-api[{extra}]"' if extra else ""
        raise MissingDependencyError(
            f"{module!r} is required{purpose}.{hint}"
        ) from exc
