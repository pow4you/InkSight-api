"""The model that ships inside the package: compressing it, and unpacking it on load.

The released InkSight-Small-p checkpoint is a TensorFlow SavedModel of about
690 MB.  It travels with the package as an xz-compressed tar archive (about
450 MB; the weights are bfloat16 and do not compress much further), split into
parts below 100 MB so that the repository stays within the file-size limits of
Git hosts.  ``vendor/model/MANIFEST.json`` records the parts and their hashes.

On first use :func:`extract_model` streams the parts through the decompressor
into the per-user data directory (about 20 seconds, once) and verifies the
hash on the way.  Later loads reuse the unpacked copy; a new bundle (different
hash) unpacks next to it.

Re-creating the bundle from a SavedModel folder::

    python -m inksight build --bundle model/small-p-cpu
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import lzma
import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .errors import BackendBuildError, BackendNotAvailableError
from .paths import MODEL_BUNDLE_DIR, MODEL_MANIFEST, default_model_root

logger = logging.getLogger("inksight")

#: Name of the checkpoint (the top-level folder inside the archive).
MODEL_NAME = "small-p-cpu"

#: Where the authors publish the same checkpoint.
MODEL_SOURCE_URL = "https://storage.googleapis.com/derendering_model/small-p-cpu.zip"

#: Size of each archive part.  Stays below the 100 MB per-file limit of GitHub.
PART_SIZE = 95 * 2 ** 20

PathLike = Union[str, "os.PathLike[str]"]


# --------------------------------------------------------------------------- manifest
def bundle_info(manifest: PathLike = MODEL_MANIFEST) -> Optional[Dict[str, Any]]:
    """The bundle's manifest, or None if the package carries no model."""
    path = Path(manifest)
    if not path.is_file():
        return None
    info = json.loads(path.read_text(encoding="utf-8"))
    parts = [path.parent / part["file"] for part in info["parts"]]
    return info if all(part.is_file() for part in parts) else None


def bundle_parts(manifest: PathLike = MODEL_MANIFEST) -> List[Path]:
    info = bundle_info(manifest)
    if info is None:
        raise BackendNotAvailableError(
            f"no bundled model found ({manifest} or its parts are missing); "
            "recreate it with `python -m inksight build --bundle <SavedModel folder>`")
    return [Path(manifest).parent / part["file"] for part in info["parts"]]


# --------------------------------------------------------------------------- compress
def bundle_model(source: PathLike, out_dir: PathLike = MODEL_BUNDLE_DIR, *,
                 preset: int = 6, part_size: int = PART_SIZE) -> Dict[str, Any]:
    """Compress a SavedModel folder into ``out_dir`` as split ``.tar.xz`` + manifest.

    Old parts in ``out_dir`` are replaced.  The archive is reproducible: entries
    are sorted and carry no timestamps or owners.

    Args:
        source: Folder containing ``saved_model.pb`` and ``variables/``.
        out_dir: Destination (default: the package's ``vendor/model``).
        preset: xz compression level (0-9).
        part_size: Maximum size of one part in bytes.

    Returns:
        The manifest that was written.
    """
    source = Path(source)
    if not (source / "saved_model.pb").is_file():
        raise BackendBuildError(f"{source} is not a SavedModel folder (no saved_model.pb)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{MODEL_NAME}.tar.xz.*"):
        old.unlink()

    files = sorted(p for p in source.rglob("*") if p.is_file())
    unpacked = sum(p.stat().st_size for p in files)
    started = time.perf_counter()
    logger.warning("inksight: compressing %s (%.0f MB) - this takes a few minutes",
                   source, unpacked / 1e6)

    writer = _PartWriter(out_dir, f"{MODEL_NAME}.tar.xz", part_size)
    with lzma.open(writer, "wb", preset=preset) as xz, \
            tarfile.open(fileobj=xz, mode="w|", format=tarfile.PAX_FORMAT) as tar:
        for path in files:
            info = tar.gettarinfo(str(path), arcname=f"{MODEL_NAME}/{path.relative_to(source).as_posix()}")
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            info.mode = 0o644
            with open(path, "rb") as handle:
                tar.addfile(info, handle)
    writer.close()

    manifest = {
        "name": MODEL_NAME,
        "format": "tar.xz",
        "source": MODEL_SOURCE_URL,
        "sha256": writer.digest.hexdigest(),
        "size": writer.total,
        "unpacked_size": unpacked,
        "files": [p.relative_to(source).as_posix() for p in files],
        "parts": writer.parts,
    }
    (out_dir / MODEL_MANIFEST.name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.warning("inksight: wrote %d parts, %.0f MB (%.0f%%) in %.0f s", len(writer.parts),
                   writer.total / 1e6, 100 * writer.total / unpacked, time.perf_counter() - started)
    return manifest


class _PartWriter(io.RawIOBase):
    """A write-only file that spreads its bytes over numbered part files."""

    def __init__(self, directory: Path, stem: str, part_size: int) -> None:
        self.directory, self.stem, self.part_size = directory, stem, part_size
        self.digest = hashlib.sha256()
        self.parts: List[Dict[str, Any]] = []
        self.total = 0
        self._handle: Optional[io.BufferedWriter] = None
        self._part_digest = hashlib.sha256()
        self._written = 0

    def writable(self) -> bool:
        return True

    def write(self, data: Any) -> int:
        view = memoryview(data).cast("B")
        done = 0
        while done < len(view):
            if self._handle is None or self._written == self.part_size:
                self._next_part()
            chunk = view[done:done + self.part_size - self._written]
            self._handle.write(chunk)
            self._part_digest.update(chunk)
            self.digest.update(chunk)
            self._written += len(chunk)
            done += len(chunk)
        self.total += done
        return done

    def _next_part(self) -> None:
        self._finish_part()
        name = f"{self.stem}.{len(self.parts):03d}"
        self.parts.append({"file": name})
        self._handle = open(self.directory / name, "wb")
        self._part_digest = hashlib.sha256()
        self._written = 0

    def _finish_part(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self.parts[-1].update(size=self._written, sha256=self._part_digest.hexdigest())
            self._handle = None

    def close(self) -> None:
        self._finish_part()
        super().close()


# --------------------------------------------------------------------------- unpack
def extract_model(root: Optional[PathLike] = None, *, manifest: PathLike = MODEL_MANIFEST,
                  force: bool = False) -> Path:
    """Unpack the bundled model (once) and return the SavedModel folder.

    Args:
        root: Directory to unpack into; default :func:`inksight.paths.default_model_root`.
        manifest: Manifest of the bundle to unpack.
        force: Unpack again even if a complete copy exists.

    Raises:
        BackendNotAvailableError: The package carries no model.
        BackendBuildError: The archive is damaged (hash mismatch).
    """
    info = bundle_info(manifest)
    parts = bundle_parts(manifest)
    root = Path(root) if root is not None else default_model_root()
    target = root / f"{info['name']}-{info['sha256'][:12]}"
    model_dir = target / info["name"]
    if (model_dir / "saved_model.pb").is_file() and not force:
        return model_dir

    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".{target.name}.{os.getpid()}.unpacking"
    shutil.rmtree(staging, ignore_errors=True)
    started = time.perf_counter()
    logger.warning("inksight: unpacking the bundled model into %s (once, ~%.0f MB)",
                   target, info["unpacked_size"] / 1e6)
    reader = _PartReader(parts)
    try:
        with tarfile.open(fileobj=reader, mode="r|xz") as tar:
            for member in tar:
                if not (member.isfile() or member.isdir()) or _escapes(member.name):
                    raise BackendBuildError(f"unexpected entry {member.name!r} in the model archive")
                tar.extract(member, staging, **_EXTRACT_OPTIONS)
        if reader.digest.hexdigest() != info["sha256"]:
            raise BackendBuildError("the bundled model archive is damaged (SHA-256 mismatch); "
                                    "reinstall the package or rebuild the bundle")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        reader.close()

    if force:
        shutil.rmtree(target, ignore_errors=True)
    try:
        staging.replace(target)          # only a complete, verified copy gets the final name
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        if not (model_dir / "saved_model.pb").is_file():   # not just a concurrent unpack
            raise
    logger.warning("inksight: model unpacked in %.0f s", time.perf_counter() - started)
    return model_dir


#: Python >= 3.12 (and recent 3.10/3.11 patch releases) can refuse unsafe members itself.
_EXTRACT_OPTIONS = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}


def _escapes(name: str) -> bool:
    parts = Path(name).parts
    return Path(name).is_absolute() or ".." in parts or not parts or parts[0] != MODEL_NAME


class _PartReader(io.RawIOBase):
    """Reads the part files back to back as one stream, hashing what passes through."""

    def __init__(self, parts: List[Path]) -> None:
        self._parts = list(parts)
        self._handle: Optional[io.BufferedReader] = None
        self.digest = hashlib.sha256()

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        while True:
            if self._handle is None:
                if not self._parts:
                    return 0
                self._handle = open(self._parts.pop(0), "rb")
            count = self._handle.readinto(buffer)
            if count:
                self.digest.update(memoryview(buffer)[:count])
                return count
            self._handle.close()
            self._handle = None

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        super().close()
