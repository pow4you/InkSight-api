"""Middle of the hierarchy: vectorizers implemented by the InkSight model."""

from __future__ import annotations

import json
import os
import tempfile
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Union

import numpy as np

from . import worker
from .base import Vectorizer
from .drawing import VectorDrawing
from .errors import VectorizationError
from .images import CropTransform, prepare_word, save_png
from .prompts import InferenceMode
from .runners import Runner, RunResult, get_runner
from .tokens import decode
from .words import Words


@dataclass
class ModelJob:
    """What a :class:`ModelVectorizer` subclass asks to be derendered.

    Attributes:
        words: The regions of the image to derender, one model input each.
            Their ``texts`` are used by the *derender with text* mode.
        metadata: Extra entries to attach to the resulting drawing.
    """

    words: Words
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelVectorizer(Vectorizer):
    """A vectorizer whose algorithm is the InkSight model.

    This class owns all the plumbing around the model: cropping each region
    and padding it to the model's square input, the prompts and the fallback
    policy, a scratch directory with the inputs as files, running the model
    program through a :class:`~inksight.runners.Runner` (in-process or in a
    container), decoding the ink tokens, and mapping the strokes back to the
    input image.

    A concrete method only declares *which regions* to derender, by
    implementing :meth:`_stage`.

    Args:
        runner: Where the model runs - ``"auto"`` (TensorFlow here if
            installed, else Docker), ``"local"``, ``"docker"`` or a
            :class:`~inksight.runners.Runner` instance.
        mode: Inference mode: ``"r+d"`` (recognize and derender, the
            notebook's default), ``"vanilla"`` (no text involved) or ``"d+t"``
            (derender the given text), or an :class:`InferenceMode`.
        fallback_mode: Mode retried for words whose first answer has no
            strokes (notebook: ``"vanilla"``); None disables the retry.
        batch_size: Words per forward pass (notebook: 32).
        timeout: Seconds after which the model program is aborted (Docker
            runner; None: no limit).
        workdir: Keep the staged word images, the job, the raw output and the
            program's log in this directory instead of a temporary one.
        invert: See :class:`~inksight.base.Vectorizer`.

    Attributes:
        last_run: :class:`~inksight.runners.RunResult` of the most recent
            model invocation (command, exit status, full log), or None.
    """

    #: Name of the model program.
    tool: ClassVar[str] = worker.TOOL

    #: File names used inside the scratch directory.
    JOB_NAME: ClassVar[str] = "job.json"
    OUTPUT_NAME: ClassVar[str] = "output.json"
    IMAGE_PATTERN: ClassVar[str] = "word_{:03d}.png"

    #: Side of the square model input; ink coordinates range over 0..MODEL_SIZE.
    MODEL_SIZE: ClassVar[int] = 224

    def __init__(self, *, runner: Union[str, Runner, None] = "auto",
                 mode: Union[str, InferenceMode] = InferenceMode.RECOGNIZE_AND_DERENDER,
                 fallback_mode: Union[str, InferenceMode, None] = InferenceMode.VANILLA,
                 batch_size: int = 32, timeout: Optional[float] = None,
                 workdir: Union[str, "os.PathLike[str]", None] = None,
                 invert: bool = False) -> None:
        super().__init__(invert=invert)
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self._runner_spec = runner
        self._runner: Optional[Runner] = runner if isinstance(runner, Runner) else None
        self.mode = InferenceMode.coerce(mode)
        self.fallback_mode = InferenceMode.coerce(fallback_mode) if fallback_mode is not None else None
        self.batch_size = int(batch_size)
        self.timeout = timeout
        self.workdir = Path(workdir) if workdir is not None else None
        self.last_run: Optional[RunResult] = None

    @property
    def runner(self) -> Runner:
        """The runner in use; resolved on first access so construction stays cheap."""
        if self._runner is None:
            self._runner = get_runner(self._runner_spec, self.tool)
        return self._runner

    # ------------------------------------------------------------------ hooks
    @abstractmethod
    def _stage(self, rgb: np.ndarray, **inputs: Any) -> ModelJob:
        """Say which regions of ``rgb`` to derender.

        Args:
            rgb: The preprocessed ``(H, W, 3)`` ``uint8`` image.
            **inputs: Extra keyword arguments given to :meth:`vectorize`.
        """

    def prompts(self, words: Words) -> "tuple[List[str], Optional[List[str]]]":
        """The prompt and the fallback prompt (or None) for every word."""
        prompts = [self.mode.prompt(text) for text in words.texts]
        fallback = self.fallback_mode
        if fallback is None or fallback is self.mode:
            return prompts, None
        # A fallback that needs text cannot help words without any.
        return prompts, [fallback.prompt(t) if (t or not fallback.needs_text) else ""
                         for t in words.texts]

    # --------------------------------------------------------------- pipeline
    def _vectorize(self, rgb: np.ndarray, **inputs: Any) -> VectorDrawing:
        height, width = rgb.shape[:2]
        job = self._stage(rgb, **inputs)
        words = job.words
        if len(words) == 0:
            return VectorDrawing([], width, height, method=self.name,
                                 metadata={"words": [], "text": "", **job.metadata})
        prompts, fallback_prompts = self.prompts(words)

        with self._scratch_dir() as workdir:
            transforms: List[CropTransform] = []
            names: List[str] = []
            for index, box in enumerate(words.boxes):
                pixels, transform = prepare_word(rgb, box, self.MODEL_SIZE)
                names.append(save_png(pixels, workdir / self.IMAGE_PATTERN.format(index)).name)
                transforms.append(transform)
            (workdir / self.JOB_NAME).write_text(json.dumps({
                "images": names, "prompts": prompts, "fallback_prompts": fallback_prompts,
                "batch_size": self.batch_size}, indent=1), encoding="utf-8")

            result = self.runner.run(self.tool, [self.JOB_NAME], workdir, self.timeout)
            self.last_run = result
            if self.workdir is not None:
                (workdir / "log.txt").write_text(result.output, encoding="utf-8")

            output = workdir / self.OUTPUT_NAME
            if result.returncode != 0 or not output.is_file():
                raise VectorizationError(self._failure_message(result), result.output)
            answer = json.loads(output.read_text(encoding="utf-8"))

        drawing = self._assemble(answer, words, transforms, width, height)
        drawing.metadata.update(elapsed=result.elapsed, runner=self.runner.name,
                                tokenizer=answer.get("tokenizer"), **job.metadata)
        return drawing

    def _assemble(self, answer: Dict[str, Any], words: Words, transforms: List[CropTransform],
                  width: int, height: int) -> VectorDrawing:
        """Decode every word's answer and put its strokes back into the image."""
        outputs = answer["outputs"]
        if len(outputs) != len(words):
            raise VectorizationError(f"the model answered {len(outputs)} of {len(words)} words")
        strokes: List[np.ndarray] = []
        details = []
        for (box, ocr_text), transform, out in zip(words, transforms, outputs):
            word_strokes, recognized = decode(out["raw"], self.MODEL_SIZE)
            start = len(strokes)
            # Model coordinates are pixel indices; +0.5 moves them to pixel centres.
            strokes += [transform.model_to_source(points + 0.5) for points in word_strokes]
            details.append({
                "box": [float(v) for v in box], "text": recognized or ocr_text or "",
                "ocr_text": ocr_text, "raw": out["raw"], "prompt": out["prompt"],
                "fallback": bool(out["fallback"]), "strokes": (start, len(strokes)),
            })
        text = " ".join(d["text"] for d in details if d["text"])
        return VectorDrawing(strokes, width, height, method=self.name,
                             metadata={"words": details, "text": text, "mode": self.mode.alias})

    @contextmanager
    def _scratch_dir(self) -> Iterator[Path]:
        """The user's ``workdir`` if one was given, else a self-deleting temporary one."""
        if self.workdir is not None:
            self.workdir.mkdir(parents=True, exist_ok=True)
            yield self.workdir
        else:
            with tempfile.TemporaryDirectory(prefix="inksight-") as temporary:
                yield Path(temporary)

    def _failure_message(self, result: RunResult) -> str:
        tail = "\n".join(result.output.strip().splitlines()[-15:])
        reason = (f"exited with status {result.returncode}" if result.returncode != 0
                  else f"finished without writing {self.OUTPUT_NAME}")
        return (f"{self.tool} {reason}.\nLast output:\n{tail}\n"
                "(the full log is in the exception's .log attribute)")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(runner={self._runner_spec!r}, mode={self.mode.alias!r})"
