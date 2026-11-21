"""The model program, ``inksight-model``: word images + prompts in, decoder text out.

This is what a :class:`~inksight.runners.Runner` executes.  It runs inside a
scratch directory prepared by :class:`~inksight.model.ModelVectorizer`::

    inksight-model job.json          # reads job.json + word PNGs, writes output.json

``job.json``::

    {"images": ["word_000.png", ...],         # square RGB model inputs
     "prompts": ["Recognize and derender.", ...],
     "fallback_prompts": ["Derender the ink.", ...] or null,
     "batch_size": 32}

Words whose first answer contains no strokes are asked again with their
fallback prompt (the official notebook falls back from *recognize and
derender* to *vanilla*), all in one more batch.  ``output.json``::

    {"outputs": [{"raw": "<ink_token_450> ...", "prompt": "...", "fallback": false}, ...],
     "model": "/path/to/saved_model", "tokenizer": "tensorflow-text",
     "load_seconds": 5.6, "run_seconds": 14.2}

:class:`~inksight.runners.LocalRunner` calls :func:`main` in-process and keeps
the loaded model between calls; :class:`~inksight.runners.DockerRunner` runs
the same function as the container's ``inksight-model`` command.

``inksight-model --serve SPOOL`` keeps the model loaded and executes jobs as
they appear: one sub-directory of ``SPOOL`` per job, picked up once it holds a
``.ready`` marker; the program then writes ``output.json``, ``log.txt``,
``exitcode`` and finally a ``.done`` marker.  A ``.stop`` file in ``SPOOL``
(or ``--idle-timeout`` seconds without work) ends the program.  The persistent
Docker runner uses this so that a container - and the model's loading and XLA
compilation - is started once, not once per call.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .errors import BackendNotAvailableError, InksightError, MissingDependencyError
from .tokens import detokenize, extract_tokens

logger = logging.getLogger("inksight.model")

#: Name of the program, as runners know it.
TOOL = "inksight-model"

#: XLA flags from the official notebook (including its spelling).  The
#: checkpoint was exported with this GEMM lowering and autotuning setting;
#: XLA reads them once, when TensorFlow is first imported.
XLA_FLAGS = (
    "--xla_gpu_autotune_level=0",
    "--xla_disable_hlo_passes=custom-kernel-fusion-rewriter,custom_kernel-fusion-autotuner",
)

#: On AMD GPUs (ROCm) the notebook's ``--xla_gpu_autotune_level=0`` leaves
#: MIOpen's convolutions without scratch memory, and the image encoder fails
#: with "Not enough workspace".  With a ROCm build of TensorFlow, autotuning
#: stays on and only the pass list is kept.
XLA_FLAGS_ROCM = XLA_FLAGS[1:]

TOKENIZERS = ("auto", "tensorflow-text", "sentencepiece")


# --------------------------------------------------------------------------- TensorFlow
def tensorflow_is_rocm() -> bool:
    """Whether the installed TensorFlow is AMD's ROCm build (checked without importing it).

    ``$INKSIGHT_ROCM`` (``1``/``0``) overrides the check.
    """
    forced = os.environ.get("INKSIGHT_ROCM")
    if forced is not None:
        return forced not in ("", "0", "false")
    from importlib import metadata
    try:
        metadata.distribution("tensorflow-rocm")
    except metadata.PackageNotFoundError:
        return False
    return True


def configure_tensorflow() -> None:
    """Set the notebook's ``XLA_FLAGS`` (and quieter TF logs) before TF is imported."""
    flags = " ".join(XLA_FLAGS_ROCM if tensorflow_is_rocm() else XLA_FLAGS)
    current = os.environ.get("XLA_FLAGS", "")
    if flags not in current:
        if "tensorflow" in sys.modules:
            warnings.warn("TensorFlow was imported before inksight could set XLA_FLAGS, so the "
                          "released checkpoint's numerics are not guaranteed. Import inksight "
                          "(or call inksight.worker.configure_tensorflow()) first.",
                          RuntimeWarning, stacklevel=3)
        os.environ["XLA_FLAGS"] = f"{current} {flags}".strip()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def import_tensorflow() -> Any:
    configure_tensorflow()
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise MissingDependencyError(
            "running the model locally needs TensorFlow: install the 'tf' extra "
            "(uv sync --extra tf, or pip install -e '.[tf]'), or use the Docker runner") from exc
    return tf


def has_tensorflow_text() -> bool:
    try:
        import tensorflow_text  # noqa: F401  (registers the SentencePiece ops)
    except ImportError:
        return False
    return True


# --------------------------------------------------------------------------- model
class ModelSession:
    """A loaded checkpoint and its ``serving_default`` signature."""

    _cache: Dict[Tuple[str, str], "ModelSession"] = {}

    def __init__(self, model_dir: Path, tokenizer: str = "auto") -> None:
        if tokenizer not in TOKENIZERS:
            raise ValueError(f"tokenizer must be one of {', '.join(TOKENIZERS)}")
        tf = import_tensorflow()
        started = time.perf_counter()
        use_text = tokenizer != "sentencepiece" and has_tensorflow_text()
        if tokenizer == "tensorflow-text" and not use_text:
            raise MissingDependencyError("tokenizer='tensorflow-text' was requested, "
                                         "but tensorflow-text is not installed")
        try:
            if use_text:
                model, self.loaded_from, self.tokenizer = tf.saved_model.load(str(model_dir)), \
                    model_dir, "tensorflow-text"
            else:
                from .tokenizer_patch import load_patched
                model, self.loaded_from = load_patched(tf, model_dir)
                self.tokenizer = "sentencepiece"
        except InksightError:
            raise
        except Exception as exc:
            raise BackendNotAvailableError(f"could not load the model at {model_dir}: {exc}") from exc
        self._tf, self._model = tf, model
        accelerators = [d.name for d in tf.config.list_logical_devices() if d.device_type != "CPU"]
        self.devices = accelerators or ["CPU"]
        self.on_accelerator = bool(accelerators)
        self._signature = model.signatures["serving_default"]
        self.model_dir = model_dir
        self.load_seconds = time.perf_counter() - started
        logger.info("model loaded from %s in %.1f s (tokenizer: %s, devices: %s)",
                    self.loaded_from, self.load_seconds, self.tokenizer, ", ".join(self.devices))

    @classmethod
    def get(cls, model_dir: Path, tokenizer: str = "auto") -> "ModelSession":
        """A cached session, so repeated in-process calls load the model once."""
        key = (str(Path(model_dir).resolve()), tokenizer)
        if key not in cls._cache:
            cls._cache[key] = cls(Path(model_dir), tokenizer)
        return cls._cache[key]

    def generate(self, images: Sequence[np.ndarray], prompts: Sequence[str],
                 batch_size: int = 32, *, sort: bool = True,
                 pad: Optional[bool] = None) -> List[str]:
        """Raw decoder text for every ``(image, prompt)`` pair, as in the notebook.

        Two scheduling tweaks.  Rows of a batch are decoded independently, so
        neither changes what a word's answer is (beyond float rounding that
        can differ between batch shapes):

        * ``sort``: the graph decodes a batch until its *longest* answer ends,
          so when there are more pairs than one batch holds, they are grouped
          by :func:`ink_estimate` (similar-length words share a batch).
        * ``pad``: the decoder is XLA-compiled per batch shape, so every new
          number of words costs a fresh compilation.  Padding each batch up
          to a power of two (by repeating its last pair, whose duplicate ends
          at the same step) caps that at log2(batch_size) shapes.  Default:
          on with a GPU (padding is nearly free there), off on a CPU (where
          the extra rows cost real time).
        """
        tf = self._tf
        count = len(images)
        if sort and count > batch_size:
            order = sorted(range(count), key=lambda i: ink_estimate(images[i]))
        else:
            order = list(range(count))
        pad = self.on_accelerator if pad is None else pad
        outputs: List[str] = [""] * count
        for start in range(0, count, batch_size):
            rows = order[start:start + batch_size]
            real = len(rows)
            if pad:
                rows = rows + [rows[-1]] * (padded_size(real, batch_size) - real)
            encoded = tf.stack([tf.reshape(tf.io.encode_jpeg(
                np.ascontiguousarray(images[i][:, :, :3])), (1,)) for i in rows])   # (B, 1)
            result = self._signature(**{
                "input_text": tf.constant([prompts[i] for i in rows], dtype=tf.string),
                "image/encoded": encoded,
            })
            decoded = result["output_0"].numpy()
            for k, i in enumerate(rows[:real]):
                outputs[i] = decoded[k][0].decode("utf-8")
        return outputs


def padded_size(count: int, batch_size: int) -> int:
    """The next power of two >= ``count``, capped at ``batch_size``."""
    return min(batch_size, 1 << max(0, count - 1).bit_length())


def ink_estimate(image: np.ndarray) -> int:
    """Rough size of a word's answer: the number of ink pixels in its model input.

    Model inputs are the word on light paper, centred on pure black padding;
    the paper's bounding box is found first so the padding is not counted.
    """
    gray = np.asarray(image)[:, :, :3].max(axis=2)
    rows, cols = np.flatnonzero(gray.any(axis=1)), np.flatnonzero(gray.any(axis=0))
    if rows.size == 0:
        return 0
    paper = gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return int((paper < 128).sum())


def resolve_model_dir(model_dir: Optional[str] = None) -> Path:
    """``model_dir``, else ``$INKSIGHT_MODEL_DIR``, else the bundled model (unpacked once)."""
    chosen = model_dir or os.environ.get("INKSIGHT_MODEL_DIR")
    if chosen:
        path = Path(chosen)
        if not (path / "saved_model.pb").is_file():
            raise BackendNotAvailableError(f"{path} is not a SavedModel folder (no saved_model.pb)")
        return path
    from .bundle import extract_model
    return extract_model()


def has_strokes(raw: str) -> bool:
    try:
        return bool(detokenize(extract_tokens(raw)))
    except ValueError:
        return False


# --------------------------------------------------------------------------- program
def run_job(job_path: Path, *, model_dir: Optional[str] = None, tokenizer: str = "auto") -> Path:
    """Execute one ``job.json``; returns the path of the ``output.json`` written next to it."""
    workdir = job_path.parent
    job = json.loads(job_path.read_text(encoding="utf-8"))
    names, prompts = job["images"], job["prompts"]
    fallback = job.get("fallback_prompts")
    batch_size = int(job.get("batch_size") or 32)
    if len(names) != len(prompts) or (fallback is not None and len(fallback) != len(names)):
        raise ValueError("job.json: images, prompts and fallback_prompts must have equal length")

    session = ModelSession.get(resolve_model_dir(model_dir), tokenizer)
    images = [np.asarray(Image.open(workdir / name).convert("RGB")) for name in names]

    started = time.perf_counter()
    raw = session.generate(images, prompts, batch_size)
    outputs = [{"raw": text, "prompt": prompt, "fallback": False} for text, prompt in zip(raw, prompts)]
    if fallback is not None:
        retry = [i for i, out in enumerate(outputs) if not has_strokes(out["raw"]) and fallback[i]]
        if retry:
            logger.info("%d word(s) without strokes; retrying with the fallback prompt", len(retry))
            again = session.generate([images[i] for i in retry], [fallback[i] for i in retry], batch_size)
            for i, text in zip(retry, again):
                outputs[i] = {"raw": text, "prompt": fallback[i], "fallback": True}
    run_seconds = time.perf_counter() - started
    logger.info("derendered %d word(s) in %.1f s on %s", len(names), run_seconds,
                ", ".join(getattr(session, "devices", None) or ["?"]))

    result = {"outputs": outputs, "model": str(session.model_dir), "tokenizer": session.tokenizer,
              "devices": getattr(session, "devices", None),
              "load_seconds": session.load_seconds, "run_seconds": run_seconds}
    output = workdir / "output.json"
    output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return output


#: Marker files of the serve protocol.
READY, DONE, STOP = ".ready", ".done", ".stop"


def serve(spool: Path, *, model_dir: Optional[str] = None, tokenizer: str = "auto",
          poll: float = 0.05, idle_timeout: Optional[float] = None) -> int:
    """Execute every job that appears in ``spool`` until told to stop (see module docs)."""
    spool = Path(spool)
    spool.mkdir(parents=True, exist_ok=True)
    session = ModelSession.get(resolve_model_dir(model_dir), tokenizer)   # load once, up front
    logger.info("serving jobs from %s (model loaded in %.1f s)", spool, session.load_seconds)
    (spool / ".serving").write_text(str(os.getpid()), encoding="utf-8")
    last_work = time.monotonic()
    while not (spool / STOP).exists():
        pending = _pending_jobs(spool)
        if not pending:
            if idle_timeout is not None and time.monotonic() - last_work > idle_timeout:
                logger.info("no work for %.0f s; stopping", idle_timeout)
                break
            time.sleep(poll)
            continue
        for job_dir in pending:
            _serve_one(job_dir, model_dir=model_dir, tokenizer=tokenizer)
        last_work = time.monotonic()
    return 0


def _pending_jobs(spool: Path) -> List[Path]:
    """Job directories that are ready and not done, oldest first.

    Clients delete finished job directories at any time, so anything may
    vanish between listing and looking; such entries are skipped.
    """
    found: List[Tuple[float, Path]] = []
    for job_dir in spool.iterdir():
        try:
            if not (job_dir / DONE).exists():
                found.append(((job_dir / READY).stat().st_mtime, job_dir))
        except OSError:                       # not ready yet, not a directory, or just removed
            continue
    return [job_dir for _, job_dir in sorted(found)]


def _serve_one(job_dir: Path, **options: Any) -> None:
    buffer: List[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: buffer.append(record.getMessage())   # type: ignore[assignment]
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        run_job(job_dir / "job.json", **options)
        code = 0
    except Exception as exc:                  # a bad job must not take the server down
        logger.error("%s: %s", TOOL, exc)
        code = 1
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    (job_dir / "log.txt").write_text("\n".join(buffer) + "\n", encoding="utf-8")
    (job_dir / "exitcode").write_text(str(code), encoding="utf-8")
    (job_dir / DONE).write_text("", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=TOOL, description="Run the InkSight model on a job file.")
    parser.add_argument("job", nargs="?", default="job.json")
    parser.add_argument("--serve", metavar="SPOOL", help="keep the model loaded and run every "
                                                         "job that appears in this directory")
    parser.add_argument("--idle-timeout", type=float, help="--serve: stop after this many idle seconds")
    parser.add_argument("--model-dir", help="SavedModel folder (default: $INKSIGHT_MODEL_DIR "
                                            "or the bundled model)")
    parser.add_argument("--tokenizer", choices=TOKENIZERS, default="auto")
    return parser


def main(argv: Optional[Sequence[str]] = None, *, workdir: Optional[Path] = None) -> int:
    """Run the program; ``workdir`` replaces the current directory (in-process use)."""
    args = _parser().parse_args(argv)
    if args.serve:
        try:
            return serve(Path(args.serve), model_dir=args.model_dir, tokenizer=args.tokenizer,
                         idle_timeout=args.idle_timeout)
        except (InksightError, OSError) as exc:
            logger.error("%s: %s", TOOL, exc)
            return 1
    job = Path(args.job)
    if workdir is not None and not job.is_absolute():
        job = Path(workdir) / job
    try:
        run_job(job, model_dir=args.model_dir, tokenizer=args.tokenizer)
    except (InksightError, OSError, ValueError, KeyError) as exc:
        logger.error("%s: %s", TOOL, exc)
        return 1
    return 0


def cli() -> None:
    """Console-script entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    sys.exit(main())


if __name__ == "__main__":
    cli()
