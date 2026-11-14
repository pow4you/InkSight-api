"""Extension points without any model download: a stub backend, a custom segmentor
and a custom renderer. Runs with only the ``viz`` extra installed.

    python examples/custom_components.py
"""

from PIL import Image

from inksight_api import BoundingBox, InkSight, InkSightModel, ModelBackend, write_ink
from inksight_api.segmentation import WordBox, WordSegmenter, segmenters
from inksight_api.visualization import InkRenderer, plot_ink, renderers


class DiagonalBackend(ModelBackend):
    """Pretends to be the network: always 'writes' one diagonal stroke."""

    def generate(self, images, prompts):
        start, y0 = 450, 225  # stroke-start token and y offset of the 224-px vocabulary
        points = [(40, 40), (112, 112), (184, 184)]
        tokens = [start] + [t for x, y in points for t in (x, y + y0)]
        return ["".join(f"<ink_token_{t}>" for t in tokens)] * len(images)


@segmenters.register("halves")
class HalvesSegmenter(WordSegmenter):
    """Split the page into a left and a right 'word'."""

    def segment(self, image):
        half = image.width // 2
        return [WordBox(BoundingBox(0, 0, half, image.height)),
                WordBox(BoundingBox(half, 0, half, image.height))]


@renderers.register("mono")
class MonoRenderer(InkRenderer):
    """Plain single-colour lines."""

    def draw(self, ax, ink, *, segment_budget=None):
        return [ax.plot(s.x, s.y, color="tab:orange", linewidth=3)[0] for s in ink if len(s) > 1]


kit = InkSight(InkSightModel(DiagonalBackend()), segmenter="halves")
page = kit.convert_page(Image.new("RGB", (600, 300), "white"))
print(page.ink, "| boxes:", len(page.boxes))

write_ink(page.ink, "custom.svg")
ax = plot_ink(page.ink, image=page.image, canvas_size=None, renderer=renderers.create("mono"))
ax.figure.savefig("custom.png", dpi=80)
print("wrote custom.svg and custom.png")
