"""Shared fixtures: a fake model backend and fake segmentors (no TensorFlow needed)."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest
from PIL import Image, ImageDraw

from inksight_api import BoundingBox, InkTokenizer, ModelBackend
from inksight_api.segmentation import WordBox, WordSegmenter

TOK = InkTokenizer()


def encode(strokes, text: str = "") -> str:
    """Inverse of ``InkTokenizer.decode`` (test helper): strokes of (x, y) -> decoder text."""
    parts = [text] if text else []
    for stroke in strokes:
        parts.append(f"<ink_token_{TOK.start_token}>")
        for x, y in stroke:
            parts.append(f"<ink_token_{x}>")
            parts.append(f"<ink_token_{y + TOK.tokens_per_dimension}>")
    return " ".join(parts)


class FakeBackend(ModelBackend):
    """Stands in for the network. ``responder(image, prompt) -> decoder text``."""

    def __init__(self, responder=None):
        self.calls: list[list[str]] = []  # prompts per generate() call
        self.responder = responder or (lambda image, prompt: encode(
            [[(100, 100), (112, 112), (124, 124)]], text="word"))

    def generate(self, images, prompts):
        assert len(images) == len(prompts)
        self.calls.append(list(prompts))
        return [self.responder(img, p) for img, p in zip(images, prompts)]


class FixedSegmenter(WordSegmenter):
    def __init__(self, word_boxes):
        self.word_boxes = word_boxes

    def segment(self, image):
        return list(self.word_boxes)


@pytest.fixture
def page_image():
    img = Image.new("RGB", (400, 300), "white")
    ImageDraw.Draw(img).rectangle([100, 50, 200, 100], outline="black")
    return img


@pytest.fixture
def two_words():
    return [WordBox(BoundingBox(100, 50, 100, 50), text="hello"),
            WordBox(BoundingBox(250, 150, 80, 80), text="world")]
