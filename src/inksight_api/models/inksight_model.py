"""The model-level API: prompts + backend + tokenizer + fallback policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from PIL import Image

from ..core.ink import Ink
from ..core.prompts import InferenceMode
from ..core.tokens import InkTokenizer
from .base import ModelBackend, backends
from . import tf_backend  # noqa: F401  (registers the built-in backend)


@dataclass(frozen=True)
class ModelOutput:
    """Everything the model produced for one image.

    Attributes:
        raw_text: The decoder text, including ``<ink_token_N>`` markers.
        ink: Decoded ink in the *model's* 224 x 224 frame.
        mode: The mode that produced this output (the fallback mode if it was used).
        prompt: The exact prompt that was sent.
        used_fallback: ``True`` if the first attempt yielded no strokes.
    """

    raw_text: str
    ink: Ink
    mode: InferenceMode
    prompt: str
    used_fallback: bool = False

    @property
    def recognized_text(self) -> str | None:
        """Text the model read (best effort, mainly for *recognize and derender*)."""
        return self.ink.text


class InkSightModel:
    """Derender word images into ink.

    Args:
        backend: Runs the network (see :class:`ModelBackend`).
        tokenizer: Decoder-text codec; the default matches the released checkpoint.
        default_mode: Mode used when :meth:`derender` is not told otherwise.
        fallback_mode: Mode retried for images whose first attempt returned no
            strokes (the notebook falls back from *recognize and derender* to
            *vanilla*). ``None`` disables the retry.
    """

    def __init__(self, backend: ModelBackend, tokenizer: InkTokenizer | None = None, *,
                 default_mode: InferenceMode = InferenceMode.RECOGNIZE_AND_DERENDER,
                 fallback_mode: InferenceMode | None = InferenceMode.VANILLA) -> None:
        self.backend = backend
        self.tokenizer = tokenizer or InkTokenizer()
        self.default_mode = default_mode
        self.fallback_mode = fallback_mode

    @classmethod
    def from_pretrained(cls, source: str | None = None, *, backend: str = "tf-saved-model",
                        default_mode: InferenceMode = InferenceMode.RECOGNIZE_AND_DERENDER,
                        fallback_mode: InferenceMode | None = InferenceMode.VANILLA,
                        **backend_kwargs) -> "InkSightModel":
        """Build a model from a registered backend (the TensorFlow one by default).

        Args:
            source: Where the checkpoint is: a ``.zip`` path or URL, a SavedModel
                directory, or a Hugging Face repository id. ``None`` uses the
                backend's default.
            backend: Registered backend name.
            **backend_kwargs: Passed to the backend's ``from_pretrained``
                (``batch_size``, ``cache_dir`` ...).
        """
        backend_cls = backends.get(backend)
        if not hasattr(backend_cls, "from_pretrained"):
            raise TypeError(f"backend {backend!r} does not support from_pretrained()")
        return cls(backend_cls.from_pretrained(source, **backend_kwargs),
                   default_mode=default_mode, fallback_mode=fallback_mode)

    def derender(self, images: Sequence[Image.Image], *, mode: InferenceMode | str | None = None,
                 texts: Sequence[str | None] | None = None,
                 use_fallback: bool = True) -> list[ModelOutput]:
        """Derender a batch of model-ready images.

        Args:
            images: Square RGB images, e.g. from a
                :class:`~inksight_api.imaging.SquarePadPreprocessor`.
            mode: Inference mode (defaults to :attr:`default_mode`).
            texts: Per-image text, required for *derender with text*.
            use_fallback: Retry empty results with :attr:`fallback_mode`
                (all retries go out as one batch).

        Returns:
            One :class:`ModelOutput` per image, in order.
        """
        n = len(images)
        if n == 0:
            return []
        mode = InferenceMode.from_alias(mode) if mode is not None else self.default_mode
        if texts is not None and len(texts) != n:
            raise ValueError("texts must have one entry per image")
        text_for = (lambda i: texts[i]) if texts is not None else (lambda i: None)

        prompts = [mode.build_prompt(text_for(i)) for i in range(n)]
        outputs = [self._decode(raw, mode, prompt)
                   for raw, prompt in zip(self.backend.generate(images, prompts), prompts)]

        fb = self.fallback_mode
        if use_fallback and fb is not None and fb is not mode:
            empty = [i for i, o in enumerate(outputs) if o.ink.is_empty]
            if empty:
                fb_prompts = [fb.build_prompt(text_for(i)) for i in empty]
                fb_raw = self.backend.generate([images[i] for i in empty], fb_prompts)
                for i, raw, prompt in zip(empty, fb_raw, fb_prompts):
                    outputs[i] = self._decode(raw, fb, prompt, used_fallback=True)
        return outputs

    def derender_one(self, image: Image.Image, *, mode: InferenceMode | str | None = None,
                     text: str | None = None, use_fallback: bool = True) -> ModelOutput:
        """Convenience wrapper for a single model-ready image."""
        return self.derender([image], mode=mode, texts=[text], use_fallback=use_fallback)[0]

    def _decode(self, raw: str, mode: InferenceMode, prompt: str, used_fallback: bool = False) -> ModelOutput:
        return ModelOutput(raw, self.tokenizer.decode(raw), mode, prompt, used_fallback)
