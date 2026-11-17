"""Derender the bundled example images with both methods.

    python examples/quickstart.py [page-image] [output-folder]

Writes <name>_word.svg (the example word), <name>_page.svg and
<name>.words.json into the output folder (default: the current directory).
Needs the 'tf' extra (or Docker) and, for the page, the 'paddle' extra.
On a CPU each model call takes a few minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

import inksight
from inksight import InkSightPage, InkSightWord
from inksight.paths import EXAMPLE_IMAGE, EXAMPLE_WORD


def main() -> None:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else EXAMPLE_IMAGE
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    # --- One word: the image goes to the model as it is.  Here from a file path.
    word = InkSightWord()                            # recognize-and-derender, as in the notebook
    drawing = word.vectorize(EXAMPLE_WORD)
    print("word:", drawing, f"read {drawing.text!r} in {drawing.metadata['elapsed']:.0f} s")
    drawing.save(out_dir / f"{EXAMPLE_WORD.stem}_word.svg")

    # --- A page: find the words, derender each.  Here from a NumPy array.
    pixels = np.asarray(Image.open(image_path).convert("RGB"))
    page = InkSightPage(word_detector="paddleocr")

    words = page.detect_words(pixels)                # stage 1 on its own
    print("words:", words, words.texts)
    words.save(out_dir / f"{stem}.words.json")

    drawing = page.vectorize(pixels, words=words)    # stage 2 with those boxes
    print("page: ", drawing, f"in {drawing.metadata['elapsed']:.0f} s")
    print("text: ", drawing.text)
    drawing.save(out_dir / f"{stem}_page.svg", stroke="crimson", stroke_width=1.5)

    # The strokes are plain arrays of x, y pixel coordinates, in writing order.
    longest = max(drawing, key=len)
    print(f"longest stroke: {len(longest)} points, from {longest[0].round(1)} to {longest[-1].round(1)}")

    # The same thing in one call:
    #   inksight.vectorize(image_path, method="page", word_detector="paddleocr").save("out.svg")
    print("available methods:", inksight.available_methods())


if __name__ == "__main__":
    main()
