"""Small, dependency-free helpers."""

from .download import download_file, extract_zip
from .lazy import require

__all__ = ["download_file", "extract_zip", "require"]
