"""Middle of the hierarchy: vectorizers implemented by the InkSight model."""

from __future__ import annotations

import json
import os
import tempfile
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np

from . import worker
from .base import Vectorizer
from .drawing import VectorDrawing
from .errors import VectorizationError
from .images import CropTransform, ImageLike, load_rgb, prepare_word, save_png
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
    def vectorize_many(self, images: Iterable[ImageLike],
                       inputs: Optional[Sequence[Dict[str, Any]]] = None) -> List[VectorDrawing]:
        """Vectorize several images with **one** model invocation.

        The words of all images go into a single job, so the batches of
        ``batch_size`` are filled across images (a page of 7 words and one of
        30 cost two forward passes, not three), and the runner's start-up -
        a container plus the model's loading for the Docker runner - is paid
        once for all of them.

        Args:
            images: Paths, NumPy arrays or PIL images.
            inputs: Optional per-image keyword arguments of :meth:`vectorize`
                (e.g. ``[{"words": "a.words.json"}, {"words": None}]``).

        Returns:
            One :class:`~inksight.drawing.VectorDrawing` per image, in order;
            ``metadata["elapsed"]`` is the shared run time.
        """
        images = list(images)
        per_image = list(inputs) if inputs is not None else [{} for _ in images]
        if len(per_image) != len(images):
            raise ValueError(f"got {len(images)} images but {len(per_image)} input dicts")
        rgbs = [self.preprocess(load_rgb(image)) for image in images]
        jobs = [self._stage(rgb, **extra) for rgb, extra in zip(rgbs, per_image)]
        drawings = self._run_jobs(rgbs, jobs)
        for drawing in drawings:
            drawing.method = drawing.method or self.name
        return [self.postprocess(drawing) for drawing in drawings]

    def _vectorize(self, rgb: np.ndarray, **inputs: Any) -> VectorDrawing:
        return self._run_jobs([rgb], [self._stage(rgb, **inputs)])[0]

    def _run_jobs(self, rgbs: Sequence[np.ndarray], jobs: Sequence[ModelJob]) -> List[VectorDrawing]:
        """Derender the words of every ``(image, job)`` pair in one runner call."""
        total = sum(len(job.words) for job in jobs)
        if total == 0:
            return [VectorDrawing([], rgb.shape[1], rgb.shape[0], method=self.name,
                                  metadata={"words": [], "text": "", **job.metadata})
                    for rgb, job in zip(rgbs, jobs)]

        prompts: List[str] = []
        fallback_prompts: Optional[List[str]] = []
        for job in jobs:
            first, fallback = self.prompts(job.words)
            prompts += first
            fallback_prompts = None if fallback is None else (fallback_prompts or []) + fallback

        with self._scratch_dir() as workdir:
            transforms: List[List[CropTransform]] = []
            names: List[str] = []
            for rgb, job in zip(rgbs, jobs):
                transforms.append([])
                for box in job.words.boxes:
                    pixels, transform = prepare_word(rgb, box, self.MODEL_SIZE)
                    names.append(save_png(pixels, workdir / self.IMAGE_PATTERN.format(len(names))).name)
                    transforms[-1].append(transform)
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

        outputs = answer["outputs"]
        if len(outputs) != total:
            raise VectorizationError(f"the model answered {len(outputs)} of {total} words")
        drawings, start = [], 0
        for rgb, job, crops in zip(rgbs, jobs, transforms):
            stop = start + len(job.words)
            drawing = self._assemble({"outputs": outputs[start:stop]}, job.words, crops,
                                     rgb.shape[1], rgb.shape[0])
            drawing.metadata.update(elapsed=result.elapsed, runner=self.runner.name,
                                    tokenizer=answer.get("tokenizer"), **job.metadata)
            if len(jobs) > 1:
                drawing.metadata["batched_images"] = len(jobs)
            drawings.append(drawing)
            start = stop
        return drawings

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
