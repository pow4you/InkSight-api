"""Safe loader for ``human_tracing_hash_to_ink.npy``.

That file is a NumPy object array holding a pickled ``dict`` of the
*notebook's* ``Ink`` / ``Stroke`` instances. Plain ``np.load(allow_pickle=True)``
has two problems here: it only works when classes with those names exist in
the module the pickle points at (``__main__`` in Colab), and unpickling can
run arbitrary code. :func:`load_legacy_ink_dict` uses a restricted unpickler
instead: it accepts only NumPy's array reconstruction and classes *named*
``Ink`` / ``Stroke`` (from any module), and converts them to this package's types.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..core.ink import Ink, Stroke


class _LegacyStroke:
    x: list
    y: list


class _LegacyInk:
    strokes: list


_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("copyreg", "_reconstructor"), ("builtins", "object"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if name == "Ink":
            return _LegacyInk
        if name == "Stroke":
            return _LegacyStroke
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to unpickle {module}.{name}")


def _convert(obj) -> Ink:
    if not isinstance(obj, _LegacyInk):
        raise pickle.UnpicklingError(f"expected an Ink, found {type(obj).__name__}")
    return Ink([Stroke.from_xy(s.x, s.y) for s in getattr(obj, "strokes", [])])


def _skip_npy_header(fh) -> None:
    """Advance past the ``.npy`` magic and header (format versions 1 to 3)."""
    magic = fh.read(8)
    if magic[:6] != b"\x93NUMPY":
        raise pickle.UnpicklingError("not a .npy file")
    size_bytes = 2 if magic[6] == 1 else 4
    fh.seek(int.from_bytes(fh.read(size_bytes), "little"), 1)


def load_legacy_ink_dict(path: str | Path) -> dict[str, Ink]:
    """Read a ``.npy`` file holding ``{sample_id: notebook Ink}`` into ``{sample_id: Ink}``."""
    with open(path, "rb") as fh:
        _skip_npy_header(fh)
        payload = _RestrictedUnpickler(fh).load()
    data = payload.item() if isinstance(payload, np.ndarray) else payload
    if not isinstance(data, dict):
        raise pickle.UnpicklingError("expected a dict of inks")
    return {str(key): _convert(value) for key, value in data.items()}
