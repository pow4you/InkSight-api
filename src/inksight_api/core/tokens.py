"""Ink tokenizer: model output text <-> :class:`Ink`.

The decoder emits ``<ink_token_N>`` markers. Token ids ``0..L`` encode an x
coordinate, ``L+1 .. 2L+1`` encode a y coordinate (offset by ``L+1``) and the
last id starts a new stroke, where ``L`` is the coordinate length (224 for the
released model, i.e. the model's input resolution).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .exceptions import InkTokenError
from .ink import Ink, Stroke

_INK_TOKEN = re.compile(r"<ink_token_(\d+)>")


@dataclass(frozen=True)
class InkTokenizer:
    """Codec between decoder text and ink.

    Args:
        coordinate_length: Largest valid coordinate value (``224`` for InkSight).
    """

    coordinate_length: int = 224

    # -- vocabulary layout ------------------------------------------------
    @property
    def tokens_per_dimension(self) -> int:
        return self.coordinate_length + 1

    @property
    def vocabulary_size(self) -> int:
        return self.tokens_per_dimension * 2 + 1

    @property
    def start_token(self) -> int:
        """Id that marks the start of a new stroke."""
        return self.tokens_per_dimension * 2

    # -- text <-> tokens --------------------------------------------------
    def extract_tokens(self, text: str) -> list[int]:
        """Pull the integer ids out of ``<ink_token_N>`` markers (notebook: ``text_to_tokens``)."""
        return [int(m) for m in _INK_TOKEN.findall(text)]

    def strip_tokens(self, text: str) -> str:
        """The decoder text with all ink markers removed.

        For *recognize and derender* this is the recognised text. That is a
        best-effort reading of the output format: the official notebook only
        uses the ink tokens.
        """
        return " ".join(_INK_TOKEN.sub(" ", text).split())

    # -- tokens -> ink ----------------------------------------------------
    def detokenize(self, tokens: Sequence[int]) -> Ink:
        """Decode token ids to an :class:`Ink`.

        Mirrors the notebook's ``detokenize``: an x token must be followed by
        a y token, out-of-range pairs are dropped and stray tokens skipped.

        Raises:
            InkTokenError: If any id is outside ``[0, vocabulary_size)``.
        """
        length = self.coordinate_length
        per_dim = self.tokens_per_dimension
        if any(t < 0 or t >= self.vocabulary_size for t in tokens):
            raise InkTokenError(
                f"Ink token indices should be between 0 and {self.vocabulary_size}"
            )

        strokes: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token == self.start_token:
                if current:
                    strokes.append(current)
                current = []
                idx += 1
            elif idx + 1 < len(tokens) and tokens[idx + 1] != self.start_token:
                x, y = tokens[idx], tokens[idx + 1] - per_dim
                if 0 <= x <= length and 0 <= y <= length:
                    current.append((x, y))
                idx += 2
            else:  # dangling x without a y
                idx += 1
        if current:
            strokes.append(current)
        return Ink([Stroke(points) for points in strokes])

    def decode(self, text: str) -> Ink:
        """Full decode: model output text -> :class:`Ink` (with ``text`` set if recognised)."""
        ink = self.detokenize(self.extract_tokens(text))
        recognised = self.strip_tokens(text)
        return ink.with_text(recognised) if recognised else ink
