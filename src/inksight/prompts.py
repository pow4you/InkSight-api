"""The three ways the InkSight model can be prompted."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union


class InferenceMode(Enum):
    """Inference modes of the released model, with their exact prompts.

    Each member carries a short alias (``vanilla`` / ``r+d`` / ``d+t``, the
    folder names of the authors' dataset) and the prompt template.
    """

    #: Derender without any text guidance.
    VANILLA = ("vanilla", "Derender the ink.")
    #: Read the text *and* derender (the official notebook's default).
    RECOGNIZE_AND_DERENDER = ("r+d", "Recognize and derender.")
    #: Derender, given the text (e.g. from an OCR engine).
    DERENDER_WITH_TEXT = ("d+t", "Derender the ink: {text}")

    def __init__(self, alias: str, template: str) -> None:
        self.alias = alias
        self.template = template

    @property
    def needs_text(self) -> bool:
        """Whether :meth:`prompt` requires a ``text`` argument."""
        return "{text}" in self.template

    def prompt(self, text: Optional[str] = None) -> str:
        """The prompt string for this mode.

        Raises:
            ValueError: The mode needs text and none (or only blank text) was given.
        """
        if not self.needs_text:
            return self.template
        if text is None or not text.strip():
            raise ValueError(f"inference mode {self.alias!r} needs the word's text, "
                             "but none was provided")
        return self.template.format(text=text.strip())

    @classmethod
    def coerce(cls, value: Union[str, "InferenceMode"]) -> "InferenceMode":
        """Resolve ``"r+d"``, ``"vanilla"``, ``"RECOGNIZE_AND_DERENDER"`` ... to a member."""
        if isinstance(value, cls):
            return value
        key = str(value).strip().lower()
        for member in cls:
            if key in (member.alias, member.name.lower()):
                return member
        valid = ", ".join(m.alias for m in cls)
        raise ValueError(f"unknown inference mode {value!r}; expected one of: {valid}")
