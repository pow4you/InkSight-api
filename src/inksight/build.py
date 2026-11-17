"""Preparing the backends: the Docker image, the local model, and the model bundle.

* :func:`build_docker_image` builds the image :class:`~inksight.runners.DockerRunner`
  runs (it does so by itself on first use).
* :func:`prepare_local` unpacks the bundled model - and, without
  tensorflow-text, rewrites its tokenizer ops - so that the first local call
  does not pay for it.
* :func:`bundle_model` (re-exported from :mod:`inksight.bundle`) compresses a
  SavedModel folder into the package.

The command line exposes all three as ``python -m inksight build``.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .bundle import bundle_model
from .errors import BackendBuildError
from .paths import DOCKER_DIR, MODEL_MANIFEST, PACKAGE_DIR

logger = logging.getLogger("inksight")

__all__ = ["DOCKER_REPOSITORY", "bundle_model", "build_docker_image", "default_image_name",
           "prepare_local", "source_fingerprint"]

#: Repository part of the Docker image name; the tag is a hash of the sources.
DOCKER_REPOSITORY = "inksight-backend"


# --------------------------------------------------------------------------- sources
def image_source_files() -> List[Path]:
    """Every file that influences the Docker image, in a stable order.

    The model enters through its manifest, which holds the archive's hash.
    """
    files = [p for p in PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts]
    files += [p for p in DOCKER_DIR.iterdir() if p.is_file()]
    if MODEL_MANIFEST.is_file():
        files.append(MODEL_MANIFEST)
    return sorted(files)


def source_fingerprint(extra: Iterable[str] = ()) -> str:
    """Short hash of the image sources (plus ``extra`` strings, e.g. build options).

    Used as the Docker image tag, so editing the package or replacing the
    model transparently triggers a rebuild.  Line endings are ignored, so a
    Windows checkout (CRLF) and a Linux one (LF) agree on the tag.
    """
    digest = hashlib.sha256()
    for path in image_source_files():
        digest.update(path.relative_to(PACKAGE_DIR).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    for item in extra:
        digest.update(b"\0" + item.encode())
    return digest.hexdigest()[:12]


# --------------------------------------------------------------------------- docker
def default_image_name(build_args: Optional[Dict[str, str]] = None) -> str:
    """Image name for the current sources and build arguments."""
    options = sorted(f"{key}={value}" for key, value in (build_args or {}).items())
    return f"{DOCKER_REPOSITORY}:{source_fingerprint(options)}"


def build_docker_image(image: Optional[str] = None, *, docker: str = "docker",
                       build_args: Optional[Dict[str, str]] = None,
                       quiet: bool = True) -> str:
    """Build the image containing TensorFlow, the model and ``inksight-model``.

    Args:
        image: Name to tag the image with; defaults to :func:`default_image_name`.
        docker: Docker-compatible CLI to invoke (``"docker"``, ``"podman"``, ...).
        build_args: ``--build-arg`` values, see ``docker/Dockerfile``.
        quiet: Capture the build log (shown only on failure) instead of streaming it.

    Raises:
        BackendBuildError: Docker is missing or the build failed.
    """
    if not MODEL_MANIFEST.is_file():
        raise BackendBuildError("the package carries no model bundle; create it first with "
                                "`python -m inksight build --bundle <SavedModel folder>`")
    image = image or default_image_name(build_args)
    command = [docker, "build", "--file", str(DOCKER_DIR / "Dockerfile"), "--tag", image]
    for key, value in (build_args or {}).items():
        command += ["--build-arg", f"{key}={value}"]
    command.append(str(PACKAGE_DIR))

    logger.warning("inksight: building Docker image %s - a one-time step that takes "
                   "a few minutes", image)
    _run_build_step(command, quiet=quiet, what="Docker image build")
    return image


# --------------------------------------------------------------------------- local
def prepare_local(model_dir: Optional[str] = None, tokenizer: str = "auto") -> Path:
    """Get the local runner ready: unpack the bundle, rewrite the tokenizer if needed.

    Returns the SavedModel folder the local runner will load.
    """
    from . import worker
    from .runners.local import LocalRunner

    path = LocalRunner(model_dir=model_dir, tokenizer=tokenizer).prepare()
    worker.import_tensorflow()
    if tokenizer == "sentencepiece" or (tokenizer == "auto" and not worker.has_tensorflow_text()):
        from .tokenizer_patch import patched_model_dir
        patched_model_dir(path)
    return path


# --------------------------------------------------------------------------- helpers
def _run_build_step(command: Sequence[str], *, quiet: bool, what: str) -> None:
    """Run one build command; on failure raise with the tail of its output."""
    try:
        if quiet:
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            output = completed.stdout or ""
        else:
            completed = subprocess.run(command)
            output = ""
    except FileNotFoundError as exc:
        raise BackendBuildError(f"{what} failed: '{command[0]}' is not installed") from exc
    if completed.returncode != 0:
        tail = "\n".join(output.splitlines()[-40:])
        raise BackendBuildError(
            f"{what} failed (exit status {completed.returncode}).\n"
            f"Command: {' '.join(map(str, command))}\n{tail}".rstrip()
        )
