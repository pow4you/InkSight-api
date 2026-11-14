"""Download / unzip helpers (ports of the notebook's ``download_file`` and ``unzip_file``)."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path


def download_file(url: str, destination: str | Path, *, timeout: float = 60.0) -> Path:
    """Stream ``url`` to ``destination`` and return the destination path."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as response, open(destination, "wb") as out:
        shutil.copyfileobj(response, out)
    return destination


def extract_zip(archive: str | Path, target_dir: str | Path = ".") -> Path:
    """Extract ``archive`` into ``target_dir`` (created if needed)."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(target_dir)
    return target_dir
