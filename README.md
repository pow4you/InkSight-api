# inksight-api

An object-oriented, extensible Python API around Google Research's
[InkSight](https://github.com/google-research/inksight) (offline-to-online handwriting
conversion). It is a restructured port of the official `colab.ipynb`: same model, same
prompts, same pre- and post-processing, organised into layers with documented extension points.

## Install

```bash
pip install -e ".[all]"          # everything
pip install -e ".[tf,viz]"       # model + plotting, no OCR (word-level only)
pip install -e .                 # data model, InkML/SVG only (NumPy + Pillow)
```

| Extra | Installs | Needed for |
|---|---|---|
| `tf` | TensorFlow 2.20, sentencepiece, tensorflow-text (not on Windows) | running the model |
| `hub` | huggingface_hub, tf-keras | loading from the Hugging Face Hub instead of a zip |
| `viz` | matplotlib | plots, animations |
| `tesseract` | pytesseract (plus the `tesseract-ocr` system package) | full pages |
| `doctr` | python-doctr | full pages |

Videos (`.mp4`) need `ffmpeg`; GIFs do not. Python 3.10 to 3.13.

## Where the model comes from

```python
InkSight.from_pretrained()                                # downloads the authors' small-p-cpu.zip
InkSight.from_pretrained("C:/models/small-p-cpu.zip")     # a zip you already have
InkSight.from_pretrained("https://.../small-p-cpu.zip")   # any zip URL
InkSight.from_pretrained("models/small-p-cpu")            # an unpacked SavedModel folder
InkSight.from_pretrained("Derendering/InkSight-Small-p")  # Hugging Face (needs the hub extra)
```

Zips are unpacked once into `~/.cache/inksight_api` (override with `INKSIGHT_CACHE` or
`cache_dir=`) and reused. The zip and folder routes need only TensorFlow. If the default
zip cannot be downloaded, the Hugging Face repository is tried.

## Windows

The released graph tokenizes its text input and output with three SentencePiece ops
that only `tensorflow-text` provides, and that package has no Windows wheels after
version 2.10 (2022). Nothing else in the graph needs it.

So when `tensorflow-text` is not installed, the package rewrites the graph once: those
three ops are replaced by calls into the `sentencepiece` Python package (the same
tokenizer library, available on every platform), using the tokenizer model that is
embedded in the graph. The rewritten copy is stored next to the original as
`<model folder>.sentencepiece` and reused; the weights are hard-linked, not copied.

```powershell
pip install -e ".[tf,viz]"
```

```python
from inksight_api import InkSight

kit = InkSight.from_pretrained()        # or a path to small-p-cpu.zip / the unpacked folder
paths = kit.convert_word("word.jpg", mode="vanilla").ink.to_polylines()
```

You can force either route on any platform with
`InkSight.from_pretrained(..., tokenizer="sentencepiece")` or `tokenizer="tensorflow-text"`.

One thing this does not change: the model core was exported from JAX and must be
compiled by TensorFlow's XLA compiler at run time. If the Windows build of TensorFlow
cannot do that, the first call fails with an XLA error, and the remaining options are
WSL2 (Linux inside Windows, where `pip install -e ".[all]"` works as is) or Docker.

## Docker

```powershell
docker build -t inksight .

# one word -> stroke paths as JSON (run from the folder with your images)
docker run --rm -v inksight-cache:/cache -v "${PWD}:/work" inksight word.jpg -o word.json

# a full page -> SVG, plus the 3-panel figure
docker run --rm -v inksight-cache:/cache -v "${PWD}:/work" inksight page.jpg --page -o page.svg --figure result.jpg
```

The `inksight-cache` volume keeps the downloaded model and docTR weights between runs.
The image is CPU only. In `cmd.exe` use `%cd%` instead of `${PWD}`. For a Python shell in
the container: `docker run --rm -it -v inksight-cache:/cache -v "${PWD}:/work" --entrypoint python inksight`.

## Command line

```
inksight IMAGE [--page] [--mode vanilla|r+d|d+t] [--text WORD] [--model SOURCE]
               [--segmenter doctr|tesseract] [-o FILE] [--figure FILE]
```

Without `-o` it prints JSON to stdout: `{"text": ..., "strokes": [[[x, y], ...], ...]}`.
With `-o` the suffix picks the format (`.json`, `.svg`, `.inkml`). The default mode is
`vanilla`, which involves no text. `--model` takes the same sources as `from_pretrained`.

## Quick start

```python
from inksight_api import InkSight, write_ink

kit = InkSight.from_pretrained(segmenter="doctr")   # downloads small-p-cpu.zip on first use

word = kit.convert_word("word.jpg")                 # one word per image
page = kit.convert_page("page.jpg")                 # segment -> derender -> stitch

page.ink            # Ink in source-image pixels
page.text           # recognised text
page.words[0]       # per word: box, model input, raw output, ink

write_ink(page.ink, "page.svg")                     # or .inkml, .json
page.plot()                                         # needs the viz extra
```

`ink.to_polylines()` gives plain `(N, 2)` NumPy arrays per stroke, for anything
downstream that wants paths (plotters, G-code, your own SVG).

Do not `import tensorflow` yourself before loading the model. The released checkpoint
needs two XLA flags that only apply if set before TensorFlow's first import; the package
sets them when it imports TensorFlow. If you must import TensorFlow first, call
`inksight_api.models.configure_tensorflow()` before that.

## Layers

| Package | Contents | Extension point |
|---|---|---|
| `core` | `Stroke`, `Ink` (immutable, NumPy-backed), `BoundingBox`, `CropTransform`, `InferenceMode`, `InkTokenizer` | |
| `imaging` | `load_image`, `SquarePadPreprocessor`, `PadColor` | subclass `ImagePreprocessor` |
| `segmentation` | `TesseractSegmenter`, `DoctrSegmenter`, `WholeImageSegmenter` | subclass `WordSegmenter`, `@segmenters.register` |
| `models` | `InkSightModel`, `TFSavedModelBackend`, `ModelOutput` | subclass `ModelBackend`, `@backends.register` |
| `pipeline` | `InkPipeline`, `WordInkPipeline`, `PageInkPipeline`, results | override `_segment`, `_prepare`, `_infer`, `_to_source` |
| `visualization` | `plot_ink`, `plot_page_result`, `InkAnimator`, `RainbowInkRenderer` | subclass `InkRenderer`, `@renderers.register` |
| `io` | `read_ink`, `write_ink` (InkML and JSON in/out, SVG out) | subclass `InkReader` / `InkWriter`, register by suffix |
| `datasets` | `SupplementaryDataset` (the `derendering_supp.zip` release) | |

Each layer imports only from the ones above it. Heavy dependencies load lazily, so
`import inksight_api` does not pull in TensorFlow or matplotlib.

## Extending

A segmentor, registered by name:

```python
from inksight_api import BoundingBox, InkSight
from inksight_api.segmentation import WordBox, WordSegmenter, segmenters

@segmenters.register("halves")
class HalvesSegmenter(WordSegmenter):
    def segment(self, image):
        half = image.width // 2
        return [WordBox(BoundingBox(0, 0, half, image.height)),
                WordBox(BoundingBox(half, 0, half, image.height))]

kit = InkSight.from_pretrained(segmenter="halves")
```

A model runtime is one method:

```python
from inksight_api import InkSightModel, ModelBackend

class MyBackend(ModelBackend):
    def generate(self, images, prompts):   # -> one decoder string per image
        ...

model = InkSightModel(MyBackend())
```

One pipeline stage, leaving the others alone:

```python
from inksight_api import PageInkPipeline

class ConfidentWordsOnly(PageInkPipeline):
    def _segment(self, image):
        return [w for w in super()._segment(image) if (w.confidence or 0) > 0.5]
```

## Inference modes

| `InferenceMode` | alias | prompt |
|---|---|---|
| `VANILLA` | `vanilla` | `Derender the ink.` |
| `RECOGNIZE_AND_DERENDER` (default) | `r+d` | `Recognize and derender.` |
| `DERENDER_WITH_TEXT` | `d+t` | `Derender the ink: <text>` |

Words that come back empty are retried once, as a single batch, with `VANILLA`
(`InkSightModel(..., fallback_mode=None)` turns that off).

## Differences from the notebook

- The four duplicated pad/scale helpers are one `SquarePadPreprocessor`; the
  `(ratio, dx, dy, min_x, min_y, angle)` tuple is a `CropTransform`. The rotation angle,
  hard-coded to 0 in the notebook, is not modelled.
- Fallback retries are batched instead of one call per empty word.
- docTR's predictor is built once and reused (the notebook rebuilt it per page).
- GIFs are written with Pillow instead of ImageMagick.
- Sample ids use the file stem. The notebook's `name.strip('.png')` also strips leading
  and trailing `p`, `n`, `g` characters from ids.
- The human-tracing `.npy` is read with a restricted unpickler that only accepts NumPy
  arrays and classes named `Ink` / `Stroke`, instead of `np.load(allow_pickle=True)`.
- Without `tensorflow-text`, the graph's SentencePiece ops are swapped for the
  `sentencepiece` package (see "Windows"). With it installed, the graph runs as exported.
- `ink.text` after *recognize and derender* is the decoder output with ink tokens removed.
  The notebook never reads the text, so treat this as best effort.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

The suite uses a fake backend and fake segmentors, so it runs without TensorFlow or OCR.

## Licence

Apache-2.0; original notebook code Copyright 2024 Google LLC (see `NOTICE`). The model
weights and dataset are released by the authors for research, non-commercial use only.
