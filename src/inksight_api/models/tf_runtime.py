"""TensorFlow environment setup required by the released checkpoint.

The official notebook sets two XLA flags *before* TensorFlow is imported to
keep the pre-2.18 GEMM lowering and deterministic autotuning that the
checkpoint was exported with. XLA reads ``XLA_FLAGS`` once, at TensorFlow's
first import, so the order matters: always go through :func:`import_tensorflow`
(or call :func:`configure_tensorflow` first) instead of ``import tensorflow``.
"""

from __future__ import annotations

import os
import sys
import warnings
from types import ModuleType
from typing import MutableMapping

from ..utils.lazy import require

#: Flags copied verbatim from the official notebook (including its spelling).
XLA_FLAGS: tuple[str, ...] = (
    "--xla_gpu_autotune_level=0",
    "--xla_disable_hlo_passes=custom-kernel-fusion-rewriter,custom_kernel-fusion-autotuner",
)


def configure_tensorflow(environ: MutableMapping[str, str] | None = None) -> bool:
    """Add the InkSight XLA flags to ``XLA_FLAGS`` (idempotent).

    Args:
        environ: Mapping to modify; defaults to ``os.environ``.

    Returns:
        ``True`` if the variable was changed, ``False`` if the flags were
        already present.
    """
    env = os.environ if environ is None else environ
    flags = " ".join(XLA_FLAGS)
    current = env.get("XLA_FLAGS", "")
    if flags in current:
        return False
    if environ is None and "tensorflow" in sys.modules:
        warnings.warn(
            "TensorFlow was imported before inksight_api set XLA_FLAGS, so the flags "
            "will not take effect. Call inksight_api.models.configure_tensorflow() "
            "before importing TensorFlow to match the released checkpoint's numerics.",
            RuntimeWarning,
            stacklevel=2,
        )
    env["XLA_FLAGS"] = f"{current} {flags}".strip()
    return True


def import_tensorflow() -> ModuleType:
    """Configure XLA, then import and return the ``tensorflow`` module."""
    configure_tensorflow()
    return require("tensorflow", extra="tf", purpose=" to run InkSight models")


def import_tensorflow_text() -> bool:
    """Import ``tensorflow_text`` if it is installed, to register its ops.

    The checkpoint's graph may use ops that live in tensorflow-text. The
    package is optional here because it has no Windows wheels after 2.10;
    whether a model really needs it only shows when the model is loaded.

    Returns:
        ``True`` if the ops are now registered, ``False`` if it is not installed.
    """
    import_tensorflow()
    try:
        import tensorflow_text  # noqa: F401
    except ImportError:
        return False
    return True
