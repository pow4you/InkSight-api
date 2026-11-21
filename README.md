# inksight
---

Handwriting to pen strokes with InkSight: **an image goes in, strokes come out.**

`inksight` is a small, class-based Python API around the model Google Research
released with *InkSight: Offline-to-Online Handwriting Conversion by Teaching
Vision-Language Models to Read and Write* (Mitrevski et al., TMLR 2025). It
wraps the authors' model - it does not re-implement it - and the model ships
with the package. It hides everything the official notebook expects you to do
by hand (downloading and unzipping the checkpoint, TensorFlow setup, padding
word crops, prompts, token decoding, mapping strokes back, finding the words
on a page).

| Method | Class | What it does |
|--------|-------|--------------|
| `"word"` | `InkSightWord` | the image shows one word; it goes to the model as it is |
| `"page"` | `InkSightPage` | a word detector (docTR, PaddleOCR or Tesseract) finds the words, the model derenders each |

```python
import inksight

drawing = inksight.vectorize("page.png", method="page")   # a path, a NumPy array or a PIL image

for stroke in drawing:        # each stroke: (N, 2) float array of x, y pixel coordinates
    print(stroke.shape)

drawing.text                  # what the model read
drawing.save("page.svg")
```

The API is the same as [`polyvec`](../PolyVector-api)'s: `vectorize()`,
`VectorDrawing`, a `Vectorizer` hierarchy with a registry, runners for "here"
and "in Docker", a two-stage method whose first stage (there keypoints, here
word boxes) can be run, saved and passed back on its own.

## Install

The project is set up for [uv](https://docs.astral.sh/uv/) (Python 3.10-3.13
for TensorFlow):

```bash
uv sync --extra tf --extra paddle     # model + PaddleOCR word detector, plus pytest
```

or, with plain pip: `pip install -e ".[tf,paddle]"`.

| Extra | Installs | Needed for |
|-------|----------|------------|
| (none) | NumPy, Pillow | the API itself, and running the model through Docker |
| `tf` | TensorFlow 2.20, sentencepiece, tensorflow-text (not on Windows) | running the model in this Python |
| `doctr` | python-doctr 1.x (PyTorch) | the notebook's word detector (default for pages) |
| `paddle` | paddlepaddle 3, paddleocr 3 | the newer PP-OCR word detector |
| `tesseract` | pytesseract (plus the `tesseract` program) | a lightweight word detector |

### The model

The released InkSight-Small-p checkpoint (a 692 MB TensorFlow SavedModel) is
part of the package, xz-compressed to 461 MB and split into five parts below
100 MB (`src/inksight/vendor/model`). It is unpacked on first use - about
20 seconds, once - into the per-user data directory (`%LOCALAPPDATA%\inksight`,
`~/.local/share/inksight`, or `$INKSIGHT_HOME`) and reused afterwards; the
archive's SHA-256 is verified while unpacking. Nothing is downloaded.

```bash
python -m inksight build --local    # unpack now instead of on the first call
python -m inksight info             # shows the methods, the model and which runners are ready
```

### Where the model runs

* **Locally (default when TensorFlow is installed).** `LocalRunner` loads the
  model into the calling Python process once (about 6 s) and keeps it.
* **Docker (default when TensorFlow is not installed).** With Docker running,
  the first call builds an `inksight-backend` image (Linux, TensorFlow with
  tensorflow-text, the model unpacked; a few minutes, once) and every call
  after that runs in a throw-away, network-less container. Nothing to install
  but Docker; each call also pays the model loading.

**Speed.** The model writes its answer one ink token at a time, and a word
takes a few hundred tokens. On a desktop CPU (measured: Ryzen 7 5800X) that is
roughly 0.6-0.9 s per token - **about 4 to 5 minutes per model call**. The
words of a call are decoded together, `batch_size` (32) at a time, so a page
costs little more than a single word. Expect about 7 GB of RAM in use. The
weights are bfloat16, which CPUs without native bfloat16 support compute
slowly. A GPU (Linux: `pip install "tensorflow[and-cuda]==2.20.0"`) is the way
to make it fast; the authors' notebook targets one too.

### On a GPU, through Docker

The checkpoint's graph is plain TensorFlow/XLA (no CPU-only compiled modules),
so the same SavedModel runs on an NVIDIA GPU; the XLA flags the notebook sets
are GPU flags to begin with.

```bash
python -m inksight build --docker --gpu                 # CUDA image, separate tag; once
python -m inksight vectorize page.png --runner docker-gpu
```

```python
InkSightPage(runner="docker-gpu")                        # = DockerRunner(gpus="all")
InkSightPage(runner=DockerRunner(gpus="device=0"))       # pick a card
```

or set `INKSIGHT_DOCKER_GPUS=all` to make every Docker runner use the GPU. The
image installs `tensorflow[and-cuda]==2.20.0` (CUDA and cuDNN as pip wheels), so
the host only needs the NVIDIA driver and the NVIDIA Container Toolkit. On
Windows, Docker Desktop's WSL 2 backend passes the GPU through out of the box;
check with `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi`.
`output.json` (and the log) name the devices TensorFlow saw, so a run that
silently fell back to the CPU is visible.

Notes: TensorFlow's Windows wheels are CPU-only, so on Windows Docker (or WSL)
is the only GPU route. The weights are bfloat16: GPUs before Ampere (e.g. a GTX
1080 Ti, compute capability 6.1) have no native bfloat16 and XLA computes it
in float32 - still far faster than a CPU, just not as fast as an RTX 30xx+.
AMD GPUs would need `tensorflow-rocm` on Linux, which is not set up here.

### Windows

TensorFlow 2.20 has native Windows CPU wheels and the model's XLA-compiled core
runs on them. The one missing piece is `tensorflow-text`, which has no Windows
wheels after 2.10: the graph tokenizes its prompt and output with three of its
SentencePiece ops (`SentencepieceOp`, `SentencepieceTokenizeOp`,
`SentencepieceDetokenizeOp`). Nothing else in the graph needs it.

So when `tensorflow-text` is missing, the graph is rewritten once
(`src/inksight/tokenizer_patch.py`): the three ops become calls into the
`sentencepiece` Python package - the same tokenizer library - with the
tokenizer model embedded in the graph and the same arguments the ops received.
The rewritten copy sits next to the unpacked model as `small-p-cpu.sentencepiece`
(weights hard-linked, not copied). On Linux both routes are available, and the
test suite checks that they give identical answers. Force either route with
`LocalRunner(tokenizer="sentencepiece")` or `tokenizer="tensorflow-text"`.

## Usage

### One call

```python
import numpy as np, inksight

image = np.asarray(...)                                     # (H, W), (H, W, 3) or (H, W, 4)
drawing = inksight.vectorize(image, method="word")
drawing = inksight.vectorize("page.jpg", method="page", word_detector="paddleocr")
```

### Reusable objects

Create a vectorizer once when processing several images; the model (local
runner) and the word detector then stay loaded.

```python
from inksight import InkSightPage, InkSightWord

word = InkSightWord(mode="vanilla")
page = InkSightPage(word_detector="paddleocr")

for path in paths:
    page.vectorize(path).save(path.with_suffix(".svg"))
```

### The result: `VectorDrawing`

```python
drawing.strokes        # list of (N, 2) float arrays: x right, y down, in input pixels
drawing.width, drawing.height
len(drawing), drawing.num_points, drawing.lengths()
drawing.text           # recognised text (recognize-and-derender mode)
drawing.to_svg(stroke="black", stroke_width=1.5)        # SVG text
drawing.save("out.svg")
drawing.metadata       # {"elapsed", "runner", "tokenizer", "mode", "text", "words": [...]}
drawing.metadata["words"][0]   # {"box", "text", "ocr_text", "raw", "prompt", "fallback", "strokes": (start, stop)}
```

The centre of the pixel in row `i`, column `j` is the point `(j + 0.5, i + 0.5)`.
Strokes are in writing order; a stroke can be a single point (the dot of an i).

### Word boxes (full pages)

InkSight derenders one word at a time. `InkSightPage.vectorize()` finds the
words and derenders them, but each stage is reachable on its own:

```python
from inksight import InkSightPage, Words

page = InkSightPage(word_detector="paddleocr")
words = page.detect_words("page.png")                    # stage 1 only
words.boxes, words.texts                                 # (N, 4) x0, y0, x1, y1   and   N strings
words.save("page.words.json")

drawing = page.vectorize("page.png", words=words)               # stage 2 with given boxes
drawing = page.vectorize("page.png", words="page.words.json")   # ... or a saved file
drawing = page.vectorize("page.png", words=np.array([[x0, y0, x1, y1], ...]))
```

| Detector | Name | Notes |
|----------|------|-------|
| `DoctrWordDetector` | `"doctr"` (default) | what the official notebook uses (docTR 1.x, PyTorch); boxes grown by 10 %. Runs in its own worker process (on Linux, PyTorch and TensorFlow abort in one process). |
| `PaddleOCRWordDetector` | `"paddleocr"` | PaddleOCR 3 / PP-OCR: text lines plus the recogniser's word positions; `level="line"` derenders whole lines. Downloads its models on first use. Runs in its own worker process too (on Linux, Paddle segfaults if TensorFlow was loaded first); oneDNN is off (Paddle 3 bug). |
| `TesseractWordDetector` | `"tesseract"` | needs the `tesseract` program |
| `WholeImageDetector` | `"whole"` | the image is one word |

### Options

| Option | Applies to | Meaning |
|--------|-----------|---------|
| `runner` | both | `"auto"` (TensorFlow here if installed, else Docker), `"local"`, `"docker"`, `"docker-gpu"`, or a `Runner` object |
| `mode` | both | `"r+d"` recognize and derender (default, as in the notebook), `"vanilla"` no text involved, `"d+t"` derender the given text |
| `fallback_mode` | both | retried for words that come back without strokes (default `"vanilla"`, as in the notebook; `None` disables) |
| `batch_size` | both | words per forward pass (32) |
| `invert` | both | set for light-on-dark writing |
| `timeout` | both | abort the model after this many seconds (Docker runner; the local runner cannot abort in-process work) |
| `workdir` | both | keep the staged word images, `job.json`, the raw `output.json` and `log.txt` in this folder (debugging) |
| `word_detector` | page | a `WordDetector` or its name |

`vectorize()` on `InkSightWord` takes `text=` (for `"d+t"`); on `InkSightPage`
it takes `words=`. After a call, `vectorizer.last_run.output` holds the model
program's log. If the model fails, a `VectorizationError` is raised with the
log in `.log`.

### Command line

```bash
python -m inksight vectorize page.png -o page.svg                  # page, docTR
python -m inksight vectorize page.png -d paddleocr -o page.json    # strokes + text as JSON
python -m inksight vectorize word.png -m word --mode vanilla
python -m inksight words page.png -d paddleocr -o page.words.json
python -m inksight vectorize page.png -w page.words.json
python -m inksight build --docker        # or --local, or --bundle model/small-p-cpu
```

## Design

```
Vectorizer                    abstract: image in -> VectorDrawing out
└── ModelVectorizer           abstract: ... by running the InkSight model
    ├── InkSightWord          the image is one word
    └── InkSightPage          ── has a ──┐
                                         ▼
WordDetector                  abstract: image in -> Words out
├── DoctrWordDetector         the notebook's engine
├── PaddleOCRWordDetector     PP-OCR (newer)
├── TesseractWordDetector
└── WholeImageDetector

Runner                        abstract: execute the model program in a directory
├── LocalRunner               TensorFlow in this process (model kept loaded)
└── DockerRunner              a container; builds its image on demand
```

Each level adds one responsibility and leaves one method to the level below:

* **`Vectorizer`** (`src/inksight/base.py`) fixes the public contract.
  `vectorize()` is a template method - load any image form as 8-bit RGB,
  `preprocess()`, `_vectorize()`, `postprocess()` - and subclasses that set a
  `name` register themselves for `inksight.vectorize(..., method=name)` and the
  command line.
* **`ModelVectorizer`** (`src/inksight/model.py`) owns the plumbing around the
  model: cropping and padding each region to the 224 x 224 input, prompts and
  the fallback policy, a scratch directory with the inputs as files, invoking
  the model program through a runner, decoding the ink tokens and mapping the
  strokes back. A concrete method only implements `_stage()`, which says which
  regions to derender.
* **`InkSightWord` / `InkSightPage`** (`src/inksight/methods/`) are therefore
  tiny: the word method is one line of substance; the page method adds the
  word-detection stage.
* **`Runner`** (`src/inksight/runners/`) decouples *what* runs from *where*. The
  model program (`inksight-model`, `src/inksight/worker.py`) always runs inside
  a directory with bare file names as arguments (`job.json` + word PNGs in,
  `output.json` out), so the same call works in-process, in Docker, or
  wherever a new runner sends it.

### Repository layout

```
src/inksight/
├── base.py, model.py           the two abstract levels
├── methods/                    InkSightWord, InkSightPage
├── words/                      Words, WordDetector, docTR / PaddleOCR / Tesseract
├── runners/                    LocalRunner, DockerRunner
├── worker.py                   the model program (inksight-model)
├── drawing.py, images.py       result type, image loading, word-crop padding
├── prompts.py, tokens.py       inference modes, ink-token decoding
├── bundle.py, tokenizer_patch.py   the compressed model; running it without tensorflow-text
├── build.py, __main__.py       backend preparation, command line
├── docker/                     Dockerfile of the backend image          (ours)
└── vendor/                     the released model, compressed            (theirs, unmodified)
tests/      examples/
```

### Extending

**A new way of choosing regions** - subclass `ModelVectorizer`:

```python
from inksight import ModelVectorizer, ModelJob, Words

class Lines(ModelVectorizer):
    name = "lines"               # registers it: inksight.vectorize(img, method="lines")

    def _stage(self, rgb, **inputs):
        rows = ...               # find text lines somehow
        return ModelJob(Words(boxes=rows))
```

**A different word detector** - subclass `WordDetector`, implement
`detect(image) -> Words`, give it a `name` to select it by name, or pass an
instance as `InkSightPage(word_detector=...)`.

**A different place to run** (a GPU box over SSH, a job queue) - subclass
`Runner`, implement `is_available()` and `run()`, pass it as `runner=...`.

**Different pre/post-processing** - override `preprocess(rgb)` or
`postprocess(drawing)` on any vectorizer.

## What is wrapped, and how faithfully

* **Model**: the authors' `small-p-cpu` SavedModel, byte for byte (see
  [`src/inksight/vendor/UPSTREAM.md`](src/inksight/vendor/UPSTREAM.md)).
* **Pre-processing**: the notebook's: crop, scale to fit 224 x 224 keeping the
  aspect ratio, centre on black, JPEG-encode; the notebook's XLA flags are set
  before TensorFlow is imported.
* **Prompts, decoding, fallback**: the notebook's three prompts, its
  `detokenize`, and its retry of empty words with the vanilla prompt (batched
  here instead of one call per word).
* **Page flow**: docTR with boxes grown by 10 %, as in the notebook - but
  docTR's PyTorch build (1.x): its TensorFlow build, which the notebook used,
  ended with 0.10, and that cannot load its weights under TensorFlow 2.20.
  PaddleOCR and Tesseract are additions.
* **Tokenizer**: the graph as exported where tensorflow-text exists (Linux,
  the Docker image); on Windows the rewritten graph described above.
* **Coordinates**: the model's coordinates are pixel indices of its input;
  they are shifted by half a pixel to the pixel-centre convention above.
* Not kept from the previous version of this package: the InkML/JSON readers
  and writers, plotting and animation, and the dataset loaders.

### Platform support

| Platform | Docker runner | Local runner |
|----------|---------------|--------------|
| Windows 11 | **tested** (Docker Desktop, Python 3.13) | **tested** (Python 3.13, TensorFlow 2.20 CPU, sentencepiece route) |
| Linux | **tested** inside a Linux container (`tests/docker/Dockerfile`); not run from a Linux host | **tested** in that container (Python 3.12, tensorflow-text route) |
| macOS | not tested | not tested |

## Tests

```bash
uv run pytest                              # unit tests always run; model tests need a runner
INKSIGHT_RUNNER=docker uv run pytest       # model tests through the Docker image
docker build -f tests/docker/Dockerfile -t inksight-tests . && docker run --rm inksight-tests   # Linux
```

Tests that need the model or an OCR engine skip themselves when those are missing.

## Licence and citation

The wrapper code is Apache-2.0 (see `LICENSE` and `NOTICE`; the original
notebook code is Copyright 2024 Google LLC). The bundled model weights are
released by the authors for **research, non-commercial use only**, and the
model card asks that they not be used for privacy-intruding purposes such as
forging handwriting. If you use the model, cite the paper:

```bibtex
@article{mitrevski2025inksight,
  title   = {InkSight: Offline-to-Online Handwriting Conversion by Teaching
             Vision-Language Models to Read and Write},
  author  = {Mitrevski, Blagoj and Rak, Arina and Schnitzler, Julian and Li, Chengkun
             and Maksai, Andrii and Berent, Jesse and Musat, Claudiu},
  journal = {Transactions on Machine Learning Research},
  year    = {2025}
}
```
