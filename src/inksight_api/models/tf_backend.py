"""TensorFlow SavedModel backend for the released ``InkSight-Small-p`` checkpoint.

The checkpoint can come from three places, all ending in the same
``tf.saved_model.load`` call:

* a zip archive, local or by URL (the authors' ``small-p-cpu.zip``),
* an already unpacked SavedModel directory,
* the Hugging Face Hub repository.
"""

from __future__ import annotations

import logging
import os
import shutil
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

from ..core.exceptions import ModelLoadError
from ..utils.download import download_file, extract_zip
from ..utils.lazy import require
from .base import ModelBackend, backends
from .inspect import find_saved_model_dir, unregistered_ops
from .tokenizer_patch import SENTENCEPIECE_OPS, load_patched
from .tf_runtime import import_tensorflow, import_tensorflow_text

logger = logging.getLogger(__name__)

#: The authors' CPU export, referenced in the official notebook.
DEFAULT_ZIP_URL = "https://storage.googleapis.com/derendering_model/small-p-cpu.zip"
DEFAULT_REPO_ID = "Derendering/InkSight-Small-p"


def default_cache_dir() -> Path:
    """Where downloaded / unpacked checkpoints live (``$INKSIGHT_CACHE`` overrides)."""
    return Path(os.environ.get("INKSIGHT_CACHE") or Path.home() / ".cache" / "inksight_api")


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


@backends.register("tf-saved-model", "tf")
class TFSavedModelBackend(ModelBackend):
    """Calls the ``serving_default`` signature of an exported TF model.

    Use :meth:`from_pretrained` (or one of the specific constructors). The
    plain constructor takes an already loaded signature, which also makes the
    class easy to test with a stand-in callable.

    Args:
        signature: The ``serving_default`` concrete function.
        tf: The imported ``tensorflow`` module (looked up if omitted).
        batch_size: Images per forward pass (the notebook uses 32).
    """

    def __init__(self, signature: Callable[..., dict[str, Any]], tf: Any | None = None,
                 batch_size: int = 32) -> None:
        self._signature = signature
        self._tf = tf if tf is not None else import_tensorflow()
        self.batch_size = batch_size

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_pretrained(cls, source: str | os.PathLike | None = None, *, batch_size: int = 32,
                        cache_dir: str | os.PathLike | None = None, tokenizer: str = "auto",
                        **hub_kwargs) -> "TFSavedModelBackend":
        """Load a checkpoint from wherever ``source`` points.

        Args:
            source: One of

                * ``None``: the authors' zip (:data:`DEFAULT_ZIP_URL`); if it
                  cannot be downloaded, the Hugging Face repository is tried.
                * a URL or local path ending in ``.zip``,
                * a directory containing a SavedModel,
                * a Hugging Face repository id such as ``Derendering/InkSight-Small-p``.
            batch_size: Images per forward pass.
            cache_dir: Where zips are downloaded and unpacked.
            tokenizer: ``"auto"``, ``"tensorflow-text"`` or ``"sentencepiece"``,
                see :meth:`from_local`.
            **hub_kwargs: ``loader`` / ``revision`` for :meth:`from_hub`.
        """
        common = {"batch_size": batch_size, "tokenizer": tokenizer}
        if source is None:
            try:
                return cls.from_zip(DEFAULT_ZIP_URL, cache_dir=cache_dir, **common)
            except OSError as exc:  # network trouble, URL gone, ...
                logger.warning("Could not fetch %s (%s); trying the Hugging Face Hub",
                               DEFAULT_ZIP_URL, exc)
                return cls.from_hub(**common, **hub_kwargs)

        text = os.fspath(source)
        if _is_url(text) or text.lower().endswith(".zip"):
            return cls.from_zip(text, cache_dir=cache_dir, **common)
        if Path(text).is_dir():
            return cls.from_local(text, **common)
        return cls.from_hub(text, **common, **hub_kwargs)

    @classmethod
    def from_zip(cls, source: str | os.PathLike = DEFAULT_ZIP_URL, *, batch_size: int = 32,
                 cache_dir: str | os.PathLike | None = None, tf: Any | None = None,
                 tokenizer: str = "auto") -> "TFSavedModelBackend":
        """Load a zipped SavedModel from a local path or a URL.

        The archive is unpacked once into the cache directory and reused on
        later calls. Needs neither ``huggingface_hub`` nor ``tf-keras``.

        Raises:
            FileNotFoundError: If a local zip does not exist, or the archive
                holds no ``saved_model.pb``.
            OSError: If a download fails.
        """
        cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        text = os.fspath(source)
        if _is_url(text):
            name = Path(urllib.parse.urlparse(text).path).stem or "model"
            target = cache / name
            archive = cache / f"{name}.zip"
            if not target.is_dir():
                logger.info("downloading %s", text)
                partial = archive.with_suffix(".zip.part")
                download_file(text, partial)
                partial.replace(archive)
        else:
            archive = Path(text)
            if not archive.is_file():
                raise FileNotFoundError(f"model archive not found: {archive}")
            target = cache / f"{archive.stem}-{archive.stat().st_size}"

        if not target.is_dir():
            staging = target.with_name(target.name + ".unpacking")
            shutil.rmtree(staging, ignore_errors=True)
            extract_zip(archive, staging)
            staging.replace(target)  # only a complete extraction gets the final name
        return cls.from_local(target, batch_size=batch_size, tf=tf, tokenizer=tokenizer)

    @classmethod
    def from_local(cls, path: str | os.PathLike, *, batch_size: int = 32, tf: Any | None = None,
                   tokenizer: str = "auto") -> "TFSavedModelBackend":
        """Load a SavedModel directory (``saved_model.pb`` may sit in a subfolder).

        Args:
            path: The directory.
            batch_size: Images per forward pass.
            tf: The imported ``tensorflow`` module (looked up if omitted).
            tokenizer: Who runs the SentencePiece tokenizer the graph needs.
                ``"tensorflow-text"`` uses the graph as exported.
                ``"sentencepiece"`` uses a rewritten copy of the graph that calls
                the ``sentencepiece`` package instead (works on Windows, see
                :mod:`.tokenizer_patch`). ``"auto"`` picks tensorflow-text when it
                is installed and sentencepiece otherwise.
        """
        if tokenizer not in {"auto", "tensorflow-text", "sentencepiece"}:
            raise ValueError("tokenizer must be 'auto', 'tensorflow-text' or 'sentencepiece'")
        tf = tf if tf is not None else import_tensorflow()
        model_dir = find_saved_model_dir(path)
        has_text_ops = tokenizer != "sentencepiece" and import_tensorflow_text()
        if tokenizer == "tensorflow-text" and not has_text_ops:
            raise ModelLoadError("tokenizer='tensorflow-text' was requested but it is not installed")

        try:
            if has_text_ops:
                model = tf.saved_model.load(str(model_dir))
            elif tokenizer == "sentencepiece" or cls._only_sentencepiece_missing(model_dir):
                model = load_patched(tf, model_dir)
            else:
                model = tf.saved_model.load(str(model_dir))
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(cls._explain_load_failure(model_dir, exc)) from exc
        return cls(model.signatures["serving_default"], tf, batch_size)

    @staticmethod
    def _only_sentencepiece_missing(model_dir: Path) -> bool:
        missing = set(unregistered_ops(model_dir))
        return bool(missing) and missing <= SENTENCEPIECE_OPS

    @classmethod
    def from_hub(cls, repo_id: str = DEFAULT_REPO_ID, *, loader: str = "auto",
                 revision: str | None = None, batch_size: int = 32,
                 tokenizer: str = "auto") -> "TFSavedModelBackend":
        """Download (and cache) a checkpoint from the Hugging Face Hub.

        Args:
            repo_id: Hub repository.
            loader: ``"keras"`` uses ``huggingface_hub.from_pretrained_keras`` as the
                notebook does (needs ``tf-keras``); ``"saved_model"`` downloads a
                snapshot and calls ``tf.saved_model.load``; ``"auto"`` tries Keras
                first and falls back.
            revision: Optional Hub revision (branch, tag or commit).
            batch_size: Images per forward pass.

        Raises:
            ModelLoadError: If no loader succeeds.
        """
        if loader not in {"auto", "keras", "saved_model"}:
            raise ValueError("loader must be 'auto', 'keras' or 'saved_model'")
        tf = import_tensorflow()
        has_text_ops = tokenizer != "sentencepiece" and import_tensorflow_text()
        hub = require("huggingface_hub", extra="hub", purpose=" to download from the Hugging Face Hub")

        errors: list[str] = []
        if loader == "keras" or (loader == "auto" and has_text_ops):
            try:
                kwargs = {"revision": revision} if revision else {}
                model = hub.from_pretrained_keras(repo_id, **kwargs)
                return cls(model.signatures["serving_default"], tf, batch_size)
            except Exception as exc:  # AttributeError on hub>=1.0, missing tf-keras, ...
                errors.append(f"keras loader: {exc!r}")
                if loader == "keras":
                    raise ModelLoadError(f"Could not load {repo_id}: {errors[-1]}") from exc
                logger.warning("Keras loader failed (%r); falling back to tf.saved_model.load", exc)

        try:
            path = hub.snapshot_download(repo_id, revision=revision)
            return cls.from_local(path, batch_size=batch_size, tf=tf, tokenizer=tokenizer)
        except Exception as exc:
            errors.append(f"saved_model loader: {exc!r}")
            raise ModelLoadError(f"Could not load {repo_id}: " + "; ".join(errors)) from exc

    @staticmethod
    def _explain_load_failure(model_dir: Path, exc: Exception) -> str:
        """Turn a load error into something actionable (which ops are missing)."""
        message = f"Could not load the SavedModel at {model_dir}: {exc}"
        try:
            missing = unregistered_ops(model_dir)
        except Exception:  # inspection is best effort
            return message
        if missing:
            message += ("\nThe model uses ops this TensorFlow does not provide: "
                        + ", ".join(sorted(missing)) + ".")
        return message

    # -- inference --------------------------------------------------------
    def generate(self, images: Sequence[Image.Image], prompts: Sequence[str]) -> list[str]:
        if len(images) != len(prompts):
            raise ValueError("images and prompts must have the same length")
        tf = self._tf
        outputs: list[str] = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start:start + self.batch_size]
            chunk_prompts = list(prompts[start:start + self.batch_size])

            encoded = tf.stack([
                tf.reshape(tf.io.encode_jpeg(np.asarray(img.convert("RGB"))[:, :, :3]), (1,))
                for img in chunk
            ])  # shape (B, 1), like the notebook's batch encoding
            result = self._signature(**{
                "input_text": tf.constant(chunk_prompts, dtype=tf.string),
                "image/encoded": encoded,
            })
            decoded = result["output_0"].numpy()
            outputs.extend(decoded[i][0].decode() for i in range(decoded.shape[0]))
        return outputs
