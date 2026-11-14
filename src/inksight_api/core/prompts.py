"""Inference modes and their text prompts."""

from __future__ import annotations

from enum import Enum

from .exceptions import PromptError


class InferenceMode(Enum):
    """The three ways the model can be prompted.

    Each member carries the short alias used for the dataset's folder names
    (``vanilla`` / ``r+d`` / ``d+t``) and the exact prompt template.
    """

    #: Derender without any text guidance.
    VANILLA = ("vanilla", "Derender the ink.")
    #: Read the text *and* derender (the notebook's default).
    RECOGNIZE_AND_DERENDER = ("r+d", "Recognize and derender.")
    #: Derender, given the text (e.g. from an OCR engine).
    DERENDER_WITH_TEXT = ("d+t", "Derender the ink: {text}")

    def __init__(self, alias: str, template: str) -> None:
        self.alias = alias
        self.template = template

    @property
    def needs_text(self) -> bool:
        """Whether :meth:`build_prompt` requires a ``text`` argument."""
        return "{text}" in self.template

    def build_prompt(self, text: str | None = None) -> str:
        """Return the prompt string for this mode.

        Raises:
            PromptError: If the mode needs text and none (or only blank text)
                was given.
        """
        if not self.needs_text:
            return self.template
        if text is None or not text.strip():
            raise PromptError(f"{self.name} needs the word's text, but none was provided")
        return self.template.format(text=text.strip())

    @classmethod
    def from_alias(cls, value: "str | InferenceMode") -> "InferenceMode":
        """Resolve ``"r+d"``, ``"vanilla"``, ``"RECOGNIZE_AND_DERENDER"`` ... to a member."""
        if isinstance(value, cls):
            return value
        key = str(value).strip().lower()
        for member in cls:
            if key in (member.alias, member.name.lower()):
                return member
        valid = ", ".join(m.alias for m in cls)
        raise PromptError(f"unknown inference mode {value!r}; expected one of: {valid}")
