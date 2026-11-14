"""Word-level and full-page conversion with the released Small-p model.

    pip install -e ".[all]"
    python examples/quickstart.py word.jpg page.jpg
"""

import sys

from inksight_api import InkSight, write_ink
from inksight_api.visualization import InkAnimator, plot_page_result

word_path, page_path = sys.argv[1], sys.argv[2]

# First argument: a .zip path/URL, a SavedModel folder or a Hugging Face repo id (None = default zip).
kit = InkSight.from_pretrained(None, segmenter="doctr")  # or segmenter="tesseract"

# --- one word ---------------------------------------------------------------
word = kit.convert_word(word_path)
print("word:", word.text, "|", word.ink)
write_ink(word.ink, "word.svg")
# Animate in the model's 224 x 224 frame, over the image the model saw.
first = word.words[0]
InkAnimator().save(first.output.ink, "word.gif", image=first.model_input)

# --- a full page ------------------------------------------------------------
page = kit.convert_page(page_path)
print(f"page: {len(page.words)} words in {page.elapsed_seconds:.1f}s")
write_ink(page.ink, "page.inkml")
write_ink(page.ink, "page.svg")
plot_page_result(page, save_to="result.jpg")
