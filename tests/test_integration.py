"""End-to-end tests against the real model (and the real OCR engines).

They run when the model can run - TensorFlow in this environment, or Docker
(in which case the first run builds the image) - and skip otherwise.  Set
``INKSIGHT_RUNNER=docker`` to test the container on a machine that also has
TensorFlow.

The model is slow on a CPU (minutes per batch of words: it decodes hundreds of
ink tokens one by one), so the model calls are shared between tests through
module-scoped fixtures: three calls in all, four where tensorflow-text exists.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from PIL import Image

import inksight
from inksight import InkSightPage, InkSightWord, Words

#: The example page's five words, measured from the image (ink extent + 6 px).
PAGE_WORD_BOXES = np.array([[27, 34, 132, 79], [131, 34, 251, 79],
                            [27, 124, 96, 169], [95, 124, 144, 169], [144, 124, 294, 169]])


def require_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        pytest.skip(f"optional dependency '{name}' is not installed")


def _ink_coverage(drawing, image, radius=3.0):
    """Fraction of stroke vertices lying within ``radius`` px of an ink pixel."""
    gray = inksight.load_grayscale(image)
    ink_rows, ink_cols = np.nonzero(gray < 128)
    ink = np.c_[ink_cols + 0.5, ink_rows + 0.5]
    points = np.concatenate(drawing.strokes)
    nearest = np.sqrt(((points[:, None, :] - ink[None, :, :]) ** 2).sum(-1)).min(1)
    return float((nearest <= radius).mean())


# --------------------------------------------------------------- shared model calls
@pytest.fixture(scope="module")
def word_drawing(example_word, model_runner):
    """InkSightWord, recognize-and-derender, on the example word."""
    return InkSightWord(runner=model_runner).vectorize(example_word)


@pytest.fixture(scope="module")
def page_drawing(example_page, model_runner):
    """InkSightPage with given boxes (no OCR involved), vanilla mode, one batch of five."""
    return InkSightPage(runner=model_runner, mode="vanilla").vectorize(
        example_page, words=Words(boxes=PAGE_WORD_BOXES))


# ------------------------------------------------------------------------ tests
@pytest.mark.model
def test_word_on_example(example_word, model_runner, word_drawing):
    drawing = word_drawing
    assert drawing.method == "word"
    assert (drawing.width, drawing.height) == Image.open(example_word).size
    assert len(drawing) >= 4
    assert _ink_coverage(drawing, example_word) > 0.9
    assert "hello" in drawing.text.lower()                 # recognize-and-derender read it
    assert drawing.metadata["runner"] == model_runner.name
    word = drawing.metadata["words"][0]
    assert word["prompt"] == "Recognize and derender." and not word["fallback"]


@pytest.mark.model
def test_page_with_given_words(example_page, page_drawing):
    """Stage 2 alone, five words in one batch."""
    drawing = page_drawing
    assert len(drawing.metadata["words"]) == len(PAGE_WORD_BOXES)
    assert drawing.metadata["detector"] == "given"
    assert _ink_coverage(drawing, example_page) > 0.9
    for word in drawing.metadata["words"]:                  # each word's strokes stay in its box
        start, stop = word["strokes"]
        assert stop > start
        points = np.concatenate(drawing.strokes[start:stop])
        x0, y0, x1, y1 = word["box"]
        assert np.all((points >= [x0 - 4, y0 - 4]) & (points <= [x1 + 4, y1 + 4]))


@pytest.mark.model
def test_svg_output(page_drawing, tmp_path):
    again = inksight.VectorDrawing.from_svg(page_drawing.save(tmp_path / "page.svg"))
    assert len(again) == len(page_drawing)


@pytest.mark.ocr
@pytest.mark.parametrize("detector,module", [("paddleocr", "paddleocr"), ("doctr", "doctr")])
def test_word_detectors_find_the_five_words(example_page, detector, module):
    require_module(module)
    engine = inksight.get_detector(detector)
    try:
        words = engine.detect(example_page)
    finally:
        engine.close()                      # free the worker process before the next test

    assert len(words) == 5
    expected_texts = ["hello", "world", "ink", "to", "strokes"]
    assert sum(t.lower() == e for t, e in zip(words.texts, expected_texts)) >= 4   # OCR may slip
    for found, expected in zip(words.boxes, PAGE_WORD_BOXES):   # boxes overlap the true words
        overlap = np.minimum(found[2:], expected[2:]) - np.maximum(found[:2], expected[:2])
        assert np.all(overlap > 0)


@pytest.mark.model
@pytest.mark.ocr
def test_page_end_to_end_with_paddleocr(example_page, model_runner):
    """The one-call path: the newer word detector and the model chained in vectorize()."""
    require_module("paddleocr")
    drawing = inksight.vectorize(example_page, method="page", runner=model_runner,
                                 word_detector="paddleocr")

    assert len(drawing.metadata["words"]) == 5
    assert drawing.metadata["detector"] == "paddleocr"
    assert _ink_coverage(drawing, example_page) > 0.9
    assert "world" in drawing.text.lower()


@pytest.mark.model
def test_sentencepiece_route_matches_tensorflow_text(example_word, model_runner, word_drawing):
    """The tokenizer rewrite used on Windows gives the same answer as tensorflow-text."""
    if model_runner.name != "local" or word_drawing.metadata["tokenizer"] != "tensorflow-text":
        pytest.skip("needs the local runner with tensorflow-text (Linux)")
    rewritten = InkSightWord(runner=inksight.LocalRunner(tokenizer="sentencepiece")).vectorize(example_word)

    assert rewritten.metadata["tokenizer"] == "sentencepiece"
    assert rewritten.metadata["words"][0]["raw"] == word_drawing.metadata["words"][0]["raw"]
