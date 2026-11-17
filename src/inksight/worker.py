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

TOKENIZERS = ("auto", "tensorflow-text", "sentencepiece")


# --------------------------------------------------------------------------- TensorFlow
def configure_tensorflow() -> None:
    """Set the notebook's ``XLA_FLAGS`` (and quieter TF logs) before TF is imported."""
    flags = " ".join(XLA_FLAGS)
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
        self._signature = model.signatures["serving_default"]
        self.model_dir = model_dir
        self.load_seconds = time.perf_counter() - started
        logger.info("model loaded from %s in %.1f s (tokenizer: %s)",
                    self.loaded_from, self.load_seconds, self.tokenizer)

    @classmethod
    def get(cls, model_dir: Path, tokenizer: str = "auto") -> "ModelSession":
        """A cached session, so repeated in-process calls load the model once."""
        key = (str(Path(model_dir).resolve()), tokenizer)
        if key not in cls._cache:
            cls._cache[key] = cls(Path(model_dir), tokenizer)
        return cls._cache[key]

    def generate(self, images: Sequence[np.ndarray], prompts: Sequence[str],
                 batch_size: int = 32) -> List[str]:
        """Raw decoder text for every ``(image, prompt)`` pair, as in the notebook."""
        tf = self._tf
        outputs: List[str] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            encoded = tf.stack([tf.reshape(tf.io.encode_jpeg(np.ascontiguousarray(img[:, :, :3])), (1,))
                                for img in chunk])                         # (B, 1)
            result = self._signature(**{
                "input_text": tf.constant(list(prompts[start:start + batch_size]), dtype=tf.string),
                "image/encoded": encoded,
            })
            decoded = result["output_0"].numpy()
            outputs.extend(decoded[i][0].decode("utf-8") for i in range(decoded.shape[0]))
        return outputs


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
    logger.info("derendered %d word(s) in %.1f s", len(names), run_seconds)

    result = {"outputs": outputs, "model": str(session.model_dir), "tokenizer": session.tokenizer,
              "load_seconds": session.load_seconds, "run_seconds": run_seconds}
    output = workdir / "output.json"
    output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=TOOL, description="Run the InkSight model on a job file.")
    parser.add_argument("job", nargs="?", default="job.json")
    parser.add_argument("--model-dir", help="SavedModel folder (default: $INKSIGHT_MODEL_DIR "
                                            "or the bundled model)")
    parser.add_argument("--tokenizer", choices=TOKENIZERS, default="auto")
    return parser


def main(argv: Optional[Sequence[str]] = None, *, workdir: Optional[Path] = None) -> int:
    """Run the program; ``workdir`` replaces the current directory (in-process use)."""
    args = _parser().parse_args(argv)
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
