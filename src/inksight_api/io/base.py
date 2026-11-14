"""Reader / writer interfaces for ink file formats.

Reading and writing are separate interfaces: some formats are write-only
(SVG here), and a class should never be forced to implement a method it
cannot honour.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..core.exceptions import RegistryError
from ..core.ink import Ink
from ..registry import Registry


class InkReader(ABC):
    """Parse a file into an :class:`Ink`."""

    @abstractmethod
    def read(self, path: str | Path) -> Ink: ...


class InkWriter(ABC):
    """Serialise an :class:`Ink` to a file."""

    @abstractmethod
    def write(self, ink: Ink, path: str | Path) -> Path: ...


#: Readers keyed by file suffix, e.g. ``".inkml"``.
ink_readers: Registry[InkReader] = Registry("ink reader")
#: Writers keyed by file suffix, e.g. ``".svg"``.
ink_writers: Registry[InkWriter] = Registry("ink writer")


def read_ink(path: str | Path, **reader_kwargs) -> Ink:
    """Read ``path`` with the reader registered for its suffix."""
    suffix = Path(path).suffix.lower()
    try:
        reader = ink_readers.create(suffix, **reader_kwargs)
    except RegistryError as exc:
        raise RegistryError(f"cannot read {suffix or 'files without a suffix'}: {exc}") from exc
    return reader.read(path)


def write_ink(ink: Ink, path: str | Path, **writer_kwargs) -> Path:
    """Write ``ink`` to ``path`` with the writer registered for its suffix."""
    suffix = Path(path).suffix.lower()
    try:
        writer = ink_writers.create(suffix, **writer_kwargs)
    except RegistryError as exc:
        raise RegistryError(f"cannot write {suffix or 'files without a suffix'}: {exc}") from exc
    return writer.write(ink, path)
