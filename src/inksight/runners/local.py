"""Run the model in this Python process, with TensorFlow installed here."""

from __future__ import annotations

import importlib.util
import io
import logging
import os
import time
from pathlib import Path
from typing import Optional, Sequence, Union

from .. import worker
from ..bundle import bundle_info, extract_model
from ..errors import BackendNotAvailableError
from ..paths import MODEL_MANIFEST
from .base import Runner, RunResult

logger = logging.getLogger("inksight")


class LocalRunner(Runner):
    """Executes ``inksight-model`` in-process; the model stays loaded between calls.

    Needs the ``tf`` extra (TensorFlow, sentencepiece; tensorflow-text where it
    exists).  The model is looked up, in order, at

    1. ``model_dir`` passed to the constructor,
    2. the folder named by ``$INKSIGHT_MODEL_DIR``,
    3. the model bundled with the package, unpacked on first use into the
       per-user data directory (:func:`inksight.bundle.extract_model`).

    Without ``tensorflow-text`` (always the case on Windows) the graph's
    SentencePiece ops are swapped for the ``sentencepiece`` package once
    (:mod:`inksight.tokenizer_patch`).

    Args:
        model_dir: SavedModel folder to use instead of the bundled model.
        tokenizer: ``"auto"`` (tensorflow-text if installed, else
            sentencepiece), ``"tensorflow-text"`` or ``"sentencepiece"``.

    Running in-process means a call cannot be aborted: ``timeout`` is honoured
    by the Docker runner only.
    """

    name = "local"

    def __init__(self, model_dir: Union[str, "os.PathLike[str]", None] = None,
                 tokenizer: str = "auto") -> None:
        if tokenizer not in worker.TOKENIZERS:
            raise ValueError(f"tokenizer must be one of {', '.join(worker.TOKENIZERS)}")
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.tokenizer = tokenizer

    # ----------------------------------------------------------------- lookup
    def model_source(self) -> Optional[Path]:
        """The SavedModel folder, or the bundle's manifest if it is not unpacked yet."""
        chosen = self.model_dir or (Path(os.environ["INKSIGHT_MODEL_DIR"])
                                    if os.environ.get("INKSIGHT_MODEL_DIR") else None)
        if chosen is not None:
            return chosen if (chosen / "saved_model.pb").is_file() else None
        return MODEL_MANIFEST if bundle_info() is not None else None

    def find(self, tool: str) -> Optional[Path]:
        """Where the model for ``tool`` comes from, or None if it cannot run here."""
        if tool != worker.TOOL or importlib.util.find_spec("tensorflow") is None:
            return None
        return self.model_source()

    def is_available(self, tool: str) -> bool:
        return self.find(tool) is not None

    def prepare(self) -> Path:
        """Unpack the bundled model now (if it is used) and return the SavedModel folder."""
        source = self.model_source()
        if source is None:
            raise BackendNotAvailableError("no model found: the bundle is missing and no "
                                           "model_dir / $INKSIGHT_MODEL_DIR is set")
        return extract_model() if source == MODEL_MANIFEST else source

    # -------------------------------------------------------------------- run
    def run(self, tool: str, args: Sequence[str], workdir: Path,
            timeout: Optional[float] = None) -> RunResult:
        if tool != worker.TOOL:
            raise BackendNotAvailableError(f"the local runner only runs '{worker.TOOL}', not '{tool}'")
        if importlib.util.find_spec("tensorflow") is None:
            raise BackendNotAvailableError(
                "TensorFlow is not installed here. Install the 'tf' extra "
                "(uv sync --extra tf), or use the Docker runner.")
        if timeout is not None:
            logger.warning("inksight: the local runner cannot abort the model; timeout ignored")

        argv = [*map(str, args), "--tokenizer", self.tokenizer]
        if self.model_dir is not None:
            argv += ["--model-dir", str(self.model_dir)]

        # Collect what the program logs, as a subprocess's output would be.
        log, buffer = logging.getLogger(worker.logger.name), io.StringIO()
        handler = logging.StreamHandler(buffer)
        handler.setFormatter(logging.Formatter("%(message)s"))
        previous_level = log.level
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        started = time.perf_counter()
        try:
            returncode = worker.main(argv, workdir=Path(workdir))
        finally:
            log.removeHandler(handler)
            log.setLevel(previous_level)
        return RunResult([tool, *argv], returncode, buffer.getvalue(), time.perf_counter() - started)

    def describe(self) -> str:
        source = self.model_dir or os.environ.get("INKSIGHT_MODEL_DIR") or "bundled model"
        return f"in-process TensorFlow ({source})"
