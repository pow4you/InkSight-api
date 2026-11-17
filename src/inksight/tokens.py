"""Decoding the model's output text into strokes.

The decoder emits ``<ink_token_N>`` markers.  With ``L`` the coordinate range
(224, the model's input size): ids ``0..L`` are an x coordinate, ``L+1..2L+1``
a y coordinate (offset by ``L+1``), and ``2L+2`` starts a new stroke.  This is
the official notebook's ``detokenize``.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import numpy as np

_INK_TOKEN = re.compile(r"<ink_token_(\d+)>")
_SPECIAL_TOKEN = re.compile(r"<extra_id_\d+>|</?s>|<pad>")


def extract_tokens(text: str) -> List[int]:
    """The integer ids of all ``<ink_token_N>`` markers in ``text``."""
    return [int(m) for m in _INK_TOKEN.findall(text)]


def recognized_text(text: str) -> str:
    """The decoder text without ink and special markers.

    After *recognize and derender* this is the word the model read.  The
    official notebook never uses it, so treat it as best effort.
    """
    plain = _SPECIAL_TOKEN.sub(" ", _INK_TOKEN.sub(" ", text))
    return " ".join(plain.split())


def detokenize(tokens: Sequence[int], coordinate_length: int = 224) -> List[np.ndarray]:
    """Token ids -> strokes as ``(N, 2)`` float arrays in the model's input frame.

    An x token must be followed by a y token; out-of-range pairs are dropped
    and stray tokens skipped, exactly as upstream does.

    Raises:
        ValueError: An id is outside the ink vocabulary.
    """
    per_dim = coordinate_length + 1
    start = 2 * per_dim
    if any(t < 0 or t > start for t in tokens):
        raise ValueError(f"ink token ids must be between 0 and {start}")

    strokes: List[List[Tuple[int, int]]] = []
    current: List[Tuple[int, int]] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == start:
            if current:
                strokes.append(current)
            current = []
            i += 1
        elif i + 1 < len(tokens) and tokens[i + 1] != start:
            x, y = tokens[i], tokens[i + 1] - per_dim
            if 0 <= x <= coordinate_length and 0 <= y <= coordinate_length:
                current.append((x, y))
            i += 2
        else:                                   # dangling x without a y
            i += 1
    if current:
        strokes.append(current)
    return [np.asarray(points, dtype=np.float64) for points in strokes]


def decode(text: str, coordinate_length: int = 224) -> Tuple[List[np.ndarray], str]:
    """Model output text -> (strokes in the model frame, recognised text)."""
    return detokenize(extract_tokens(text), coordinate_length), recognized_text(text)
