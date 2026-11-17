"""Where things live: the bundled model, the Docker build glue, and the per-user data."""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: The installed ``inksight`` package directory.
PACKAGE_DIR = Path(__file__).resolve().parent

#: Dockerfile of the backend image.
DOCKER_DIR = PACKAGE_DIR / "docker"

#: Upstream material: the compressed model and its provenance (see ``vendor/UPSTREAM.md``).
VENDOR_DIR = PACKAGE_DIR / "vendor"

#: The released InkSight-Small-p checkpoint, as a split ``.tar.xz`` plus manifest.
MODEL_BUNDLE_DIR = VENDOR_DIR / "model"
MODEL_MANIFEST = MODEL_BUNDLE_DIR / "MANIFEST.json"

#: Small drawings shipped with the package; handy for smoke tests.
EXAMPLE_WORD = VENDOR_DIR / "examples" / "word.png"
EXAMPLE_IMAGE = VENDOR_DIR / "examples" / "page.png"


def user_data_dir() -> Path:
    """Per-user directory for the unpacked model.

    ``$INKSIGHT_HOME`` if set, else ``%LOCALAPPDATA%\\inksight`` on Windows,
    ``~/Library/Application Support/inksight`` on macOS and
    ``$XDG_DATA_HOME/inksight`` (default ``~/.local/share/inksight``) elsewhere.
    """
    override = os.environ.get("INKSIGHT_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "inksight"


def default_model_root() -> Path:
    """Where the bundled model is unpacked on first use."""
    return user_data_dir() / "model"
