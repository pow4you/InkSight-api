"""Digital-ink data model: :class:`Stroke` and :class:`Ink`.

Both are *immutable value objects* backed by NumPy. Every geometric operation
returns a new object, so an ink can be passed around (and drawn, animated,
exported ...) without anyone mutating it behind your back. This replaces the
notebook's list-of-floats ``Stroke`` / ``Ink`` classes; ``stroke.x``,
``stroke.y``, ``len()``, indexing and iteration keep working.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

Point = tuple[float, float]
PointArray = np.ndarray  # shape (N, 2), dtype float


class Stroke:
    """One pen-down trajectory: an ordered sequence of ``(x, y)`` points.

    Args:
        points: Anything convertible to an ``(N, 2)`` float array
            (list of tuples, array, generator ...). ``None`` gives an empty
            stroke. The data is copied and made read-only.

    Raises:
        ValueError: If ``points`` is not shaped ``(N, 2)``.
    """

    __slots__ = ("_points",)

    def __init__(self, points: Iterable[Point] | PointArray | None = None) -> None:
        if points is None:
            arr = np.empty((0, 2), dtype=float)
        else:
            source = points if isinstance(points, np.ndarray) else list(points)
            arr = np.array(source, dtype=float)  # always copies
            if arr.size == 0:
                arr = arr.reshape(0, 2)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"A stroke needs an (N, 2) array of points, got shape {arr.shape}")
        arr.setflags(write=False)
        self._points = arr

    # -- construction -----------------------------------------------------
    @classmethod
    def from_xy(cls, x: Sequence[float], y: Sequence[float]) -> "Stroke":
        """Build a stroke from separate x and y sequences."""
        return cls(np.column_stack([np.asarray(x, float), np.asarray(y, float)]))

    # -- views ------------------------------------------------------------
    @property
    def points(self) -> PointArray:
        """Read-only ``(N, 2)`` array of points."""
        return self._points

    @property
    def x(self) -> np.ndarray:
        """All x coordinates (compat with the notebook's ``stroke.x``)."""
        return self._points[:, 0]

    @property
    def y(self) -> np.ndarray:
        """All y coordinates (compat with the notebook's ``stroke.y``)."""
        return self._points[:, 1]

    @property
    def segments(self) -> np.ndarray:
        """Line segments as an ``(N-1, 2, 2)`` array (empty for < 2 points).

        This is the shape :class:`matplotlib.collections.LineCollection` expects.
        """
        p = self._points
        if len(p) < 2:
            return np.empty((0, 2, 2), dtype=float)
        return np.stack([p[:-1], p[1:]], axis=1)

    # -- measurements -----------------------------------------------------
    def bounds(self) -> tuple[float, float, float, float] | None:
        """``(xmin, ymin, xmax, ymax)`` or ``None`` for an empty stroke."""
        if len(self) == 0:
            return None
        lo, hi = self._points.min(axis=0), self._points.max(axis=0)
        return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])

    def length(self) -> float:
        """Polyline length of the stroke."""
        if len(self) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self._points, axis=0), axis=1).sum())

    # -- transformations (all return new strokes) -------------------------
    def transformed(self, fn: Callable[[PointArray], PointArray]) -> "Stroke":
        """Apply ``fn`` to the ``(N, 2)`` point array and wrap the result."""
        return Stroke(fn(self._points))

    def affine(self, scale: tuple[float, float] = (1.0, 1.0),
               offset: tuple[float, float] = (0.0, 0.0)) -> "Stroke":
        """Return ``points * scale + offset`` as a new stroke."""
        return Stroke(self._points * np.asarray(scale, float) + np.asarray(offset, float))

    # -- sequence protocol ------------------------------------------------
    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self) -> Iterator[Point]:
        for px, py in self._points:
            yield float(px), float(py)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Stroke(self._points[index])
        px, py = self._points[index]
        return float(px), float(py)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Stroke) and np.array_equal(self._points, other._points)

    __hash__ = None  # type: ignore[assignment]  # equality is by value, so not hashable

    def __repr__(self) -> str:
        return f"Stroke(n_points={len(self)})"


class Ink:
    """An ordered collection of :class:`Stroke` objects (one handwriting sample).

    Args:
        strokes: The strokes, in writing order.
        text: Recognised or reference text, if known (``None`` otherwise).
        metadata: Free-form annotations (e.g. the InkML ``annotation`` fields).
    """

    __slots__ = ("_strokes", "text", "metadata")

    def __init__(self, strokes: Iterable[Stroke | Iterable[Point]] | None = None, *,
                 text: str | None = None, metadata: Mapping[str, str] | None = None) -> None:
        self._strokes: tuple[Stroke, ...] = tuple(
            s if isinstance(s, Stroke) else Stroke(s) for s in (strokes or ())
        )
        self.text = text
        self.metadata: dict[str, str] = dict(metadata or {})

    # -- construction -----------------------------------------------------
    @classmethod
    def merge(cls, inks: Iterable["Ink"], *, text: str | None = None) -> "Ink":
        """Concatenate the strokes of several inks into one."""
        return cls([s for ink in inks for s in ink], text=text)

    # -- views ------------------------------------------------------------
    @property
    def strokes(self) -> tuple[Stroke, ...]:
        """The strokes as a tuple (compat with the notebook's ``ink.strokes``)."""
        return self._strokes

    @property
    def is_empty(self) -> bool:
        """``True`` if there are no strokes."""
        return len(self._strokes) == 0

    @property
    def point_count(self) -> int:
        """Total number of points over all strokes."""
        return sum(len(s) for s in self._strokes)

    def bounds(self) -> tuple[float, float, float, float] | None:
        """``(xmin, ymin, xmax, ymax)`` over all strokes, or ``None`` if empty."""
        boxes = [b for b in (s.bounds() for s in self._strokes) if b is not None]
        if not boxes:
            return None
        xs0, ys0, xs1, ys1 = zip(*boxes)
        return min(xs0), min(ys0), max(xs1), max(ys1)

    def to_polylines(self) -> list[np.ndarray]:
        """Plain ``(N, 2)`` arrays, one per stroke: handy for plotters, SVG, G-code."""
        return [s.points.copy() for s in self._strokes]

    # -- transformations (all return new inks, text/metadata preserved) ----
    def _derive(self, strokes: Iterable[Stroke]) -> "Ink":
        return Ink(strokes, text=self.text, metadata=self.metadata)

    def transformed(self, fn: Callable[[PointArray], PointArray]) -> "Ink":
        """Apply ``fn`` to every stroke's ``(N, 2)`` point array."""
        return self._derive(s.transformed(fn) for s in self._strokes)

    def affine(self, scale: tuple[float, float] = (1.0, 1.0),
               offset: tuple[float, float] = (0.0, 0.0)) -> "Ink":
        """Return ``points * scale + offset`` for every stroke."""
        return self._derive(s.affine(scale, offset) for s in self._strokes)

    def scaled(self, sx: float, sy: float | None = None) -> "Ink":
        """Scale about the origin. ``scaled(2)`` is isotropic."""
        return self.affine((sx, sx if sy is None else sy))

    def translated(self, dx: float, dy: float) -> "Ink":
        """Shift every point by ``(dx, dy)``."""
        return self.affine(offset=(dx, dy))

    def with_text(self, text: str | None) -> "Ink":
        """Copy of this ink with different ``text``."""
        return Ink(self._strokes, text=text, metadata=self.metadata)

    # -- sequence protocol ------------------------------------------------
    def __len__(self) -> int:
        return len(self._strokes)

    def __iter__(self) -> Iterator[Stroke]:
        return iter(self._strokes)

    def __getitem__(self, index):
        return self._strokes[index]

    def __repr__(self) -> str:
        return f"Ink(n_strokes={len(self)}, n_points={self.point_count}, text={self.text!r})"
