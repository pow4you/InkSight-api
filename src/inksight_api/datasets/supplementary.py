"""Access to the supplementary dataset release (``derendering_supp.zip``).

It holds sample images from IMGUR5K, IAM and HierText, the pre-computed
model outputs as InkML for three model variants and three inference modes,
and human-traced reference inks. Layout (see the notebook)::

    derendering_supp/
      <Dataset>/images_sample/<id>.png
      <model>_<Dataset>_inkml/{d+t,r+d,vanilla}/<id>.inkml
      human_tracing_hash_to_ink.npy

Licence note from the release: these outputs and the model are Apache-2.0 but for
**research, non-commercial use** only (they derive from non-commercial datasets).
"""

from __future__ import annotations

import random
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image

from ..core.ink import Ink
from ..core.prompts import InferenceMode
from ..io.inkml import InkMLReader
from ..utils.download import download_file, extract_zip
from .legacy_pickle import load_legacy_ink_dict


class DatasetName(str, Enum):
    IMGUR5K = "IMGUR5K"
    IAM = "IAM"
    HIERTEXT = "HierText"


class ModelVariant(str, Enum):
    SMALL_I = "Small-i"
    LARGE_I = "Large-i"
    SMALL_P = "Small-p"


class SupplementaryDataset:
    """Typed access to the downloaded supplementary data.

    Args:
        root: Directory that holds (or will hold) ``derendering_supp``.
    """

    URL = "https://storage.googleapis.com/derendering_model/derendering_supp.zip"

    def __init__(self, root: str | Path = "derendering_supp") -> None:
        self.root = Path(root)
        self._reader = InkMLReader()

    # -- acquisition ------------------------------------------------------
    @property
    def is_available(self) -> bool:
        return self.root.is_dir()

    def download(self, *, force: bool = False) -> Path:
        """Fetch and unpack the release next to :attr:`root` (skipped if already there)."""
        if self.is_available and not force:
            return self.root
        archive = self.root.parent / f"{self.root.name}.zip"
        download_file(self.URL, archive)
        extract_zip(archive, self.root.parent)
        return self.root

    # -- paths ------------------------------------------------------------
    def image_dir(self, dataset: DatasetName | str) -> Path:
        return self.root / DatasetName(dataset).value / "images_sample"

    def inkml_dir(self, model: ModelVariant | str, dataset: DatasetName | str,
                  mode: InferenceMode | str) -> Path:
        name = f"{ModelVariant(model).value.lower()}_{DatasetName(dataset).value}_inkml"
        return self.root / name / InferenceMode.from_alias(mode).alias

    # -- samples ----------------------------------------------------------
    def sample_ids(self, dataset: DatasetName | str) -> list[str]:
        """Ids (file stems) of the sample images of ``dataset``, sorted."""
        directory = self.image_dir(dataset)
        return sorted(p.stem for p in directory.glob("*.png")) if directory.is_dir() else []

    def random_sample_ids(self, dataset: DatasetName | str, n: int, *, seed: int | None = None) -> list[str]:
        """``n`` distinct random sample ids."""
        return random.Random(seed).sample(self.sample_ids(dataset), n)

    def load_image(self, dataset: DatasetName | str, sample_id: str) -> Image.Image:
        return Image.open(self.image_dir(dataset) / f"{sample_id}.png").convert("RGB")

    def load_ink(self, model: ModelVariant | str, dataset: DatasetName | str,
                 mode: InferenceMode | str, sample_id: str) -> Ink:
        """Pre-computed model ink for a sample (``ink.text`` is the OCR / recognised text)."""
        return self._reader.read(self.inkml_dir(model, dataset, mode) / f"{sample_id}.inkml")

    def human_traced_inks(self) -> dict[str, Ink]:
        """Human reference traces keyed by sample id (HierText samples).

        Loaded with a restricted unpickler, see :mod:`.legacy_pickle`.
        """
        return load_legacy_ink_dict(self.root / "human_tracing_hash_to_ink.npy")

    def iter_samples(self, dataset: DatasetName | str, model: ModelVariant | str,
                     mode: InferenceMode | str) -> Iterator[tuple[str, Path, Path]]:
        """Yield ``(sample_id, image_path, inkml_path)`` for samples that have both files."""
        for sample_id in self.sample_ids(dataset):
            image = self.image_dir(dataset) / f"{sample_id}.png"
            inkml = self.inkml_dir(model, dataset, mode) / f"{sample_id}.inkml"
            if inkml.exists():
                yield sample_id, image, inkml

    # -- bulk rendering (notebook: pregenerate_videos) ----------------------
    def render_videos(self, cache_dir: str | Path, *,
                      datasets: Iterable[DatasetName | str] = tuple(DatasetName),
                      models: Iterable[ModelVariant | str] = tuple(ModelVariant),
                      modes: Iterable[InferenceMode | str] = tuple(InferenceMode),
                      animator=None) -> list[Path]:
        """Render an mp4 for every available sample; existing files are skipped.

        Needs the ``viz`` extra and ``ffmpeg``. Files are named
        ``<Model>_<Dataset>_<mode>_<id>.mp4``.
        """
        from ..imaging.preprocessing import PadColor, SquarePadPreprocessor
        from ..visualization.animation import InkAnimator

        animator = animator or InkAnimator()
        backdrop = SquarePadPreprocessor(pad_color=PadColor.WHITE)
        cache_dir = Path(cache_dir)
        written: list[Path] = []
        for dataset in datasets:
            for model in models:
                for mode in modes:
                    for sample_id, image_path, inkml_path in self.iter_samples(dataset, model, mode):
                        target = cache_dir / (f"{ModelVariant(model).value}_{DatasetName(dataset).value}"
                                              f"_{InferenceMode.from_alias(mode).alias}_{sample_id}.mp4")
                        if target.exists():
                            continue
                        image = backdrop.prepare(Image.open(image_path).convert("RGB")).image
                        animator.save(self._reader.read(inkml_path), target, image=image)
                        written.append(target)
        return written
