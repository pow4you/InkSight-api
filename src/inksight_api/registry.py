"""Name -> class registries for pluggable components.

Every extension point (segmentors, model backends, ink file formats, ...)
owns one :class:`Registry`. Registering a class makes it constructible by
name, which is how :class:`~inksight_api.InkSight` and the pipelines accept
plain strings such as ``segmenter="doctr"``.

Example:
    >>> from inksight_api.segmentation import WordSegmenter, segmenters
    >>> @segmenters.register("my-ocr")
    ... class MySegmenter(WordSegmenter):
    ...     def segment(self, image):
    ...         return []
    >>> segmenters.create("my-ocr")        # doctest: +ELLIPSIS
    <...MySegmenter object at ...>
"""

from __future__ import annotations

from typing import Callable, Generic, Iterator, TypeVar

from .core.exceptions import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """A case-insensitive mapping from names to classes.

    Args:
        kind: Human readable component kind, used in error messages
            (``"segmenter"``, ``"backend"`` ...).
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    @staticmethod
    def _normalise(name: str) -> str:
        return name.strip().lower()

    def register(self, *names: str, replace: bool = False) -> Callable[[type[T]], type[T]]:
        """Class decorator that registers the class under one or more names.

        Args:
            *names: One or more names (aliases) for the class.
            replace: Allow overriding an existing registration. Without it,
                re-using a name for a *different* class raises
                :class:`RegistryError`.
        """
        if not names:
            raise ValueError("register() needs at least one name")

        def decorator(cls: type[T]) -> type[T]:
            for name in names:
                key = self._normalise(name)
                existing = self._items.get(key)
                if existing is not None and existing is not cls and not replace:
                    raise RegistryError(
                        f"{self.kind} {name!r} is already registered "
                        f"({existing.__name__}); pass replace=True to override"
                    )
                self._items[key] = cls
            return cls

        return decorator

    def unregister(self, name: str) -> None:
        """Remove a registration (no error if the name is unknown)."""
        self._items.pop(self._normalise(name), None)

    def get(self, name: str) -> type[T]:
        """Return the class registered under ``name``."""
        try:
            return self._items[self._normalise(name)]
        except KeyError:
            known = ", ".join(self.names()) or "none"
            raise RegistryError(
                f"unknown {self.kind} {name!r}; registered: {known}"
            ) from None

    def create(self, name: str, /, **kwargs) -> T:
        """Instantiate the class registered under ``name`` with ``kwargs``."""
        return self.get(name)(**kwargs)

    def names(self) -> list[str]:
        """All registered names, sorted."""
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._normalise(name) in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)
