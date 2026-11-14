"""Renderer interface for drawing ink onto matplotlib axes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.ink import Ink
from ..registry import Registry


class InkRenderer(ABC):
    """Draws an :class:`Ink` onto a matplotlib ``Axes``.

    Plotting, animation and the page figure all talk to this interface, so a
    new look (single colour, pressure-style width, ...) is one subclass away
    and works everywhere. Use ``@renderers.register("name")`` to make it
    selectable by name.
    """

    @abstractmethod
    def draw(self, ax: Any, ink: Ink, *, segment_budget: int | None = None) -> list[Any]:
        """Draw ``ink`` and return the matplotlib artists that were added.

        Args:
            ax: Target ``matplotlib.axes.Axes``.
            ink: What to draw.
            segment_budget: If given, draw only this many line segments in
                total, in writing order. This is what makes animation possible.

        Returns:
            The created artists, so callers can remove them again.
        """


#: Registry of renderers (``"rainbow"`` is built in).
renderers: Registry[InkRenderer] = Registry("renderer")
