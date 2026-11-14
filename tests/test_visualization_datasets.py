import pickle
import shutil
import subprocess
import sys

import pytest
from PIL import Image

from conftest import FakeBackend, FixedSegmenter
from inksight_api import Ink, InkSightModel, PageInkPipeline, Stroke
from inksight_api.datasets import DatasetName, ModelVariant, SupplementaryDataset
from inksight_api.io import write_ink
from inksight_api.visualization import (InkAnimator, InkRenderer, RainbowInkRenderer, StrokeStyle,
                                        adaptive_line_width, plot_ink, plot_page_result, renderers)

pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt  # noqa: E402

INK = Ink([Stroke([(10, 10), (50, 60), (90, 20), (120, 100)]), Stroke([(0, 0)]),
           Stroke([(20, 150), (200, 160)])])


def test_renderer_skips_degenerate_strokes_and_honours_budget():
    fig, ax = plt.subplots()
    r = RainbowInkRenderer()
    assert len(r.draw(ax, INK)) == 2  # the single-point stroke is skipped
    assert [len(c.get_segments()) for c in r.draw(ax, INK, segment_budget=4)] == [3, 1]
    assert [len(c.get_segments()) for c in r.draw(ax, INK, segment_budget=2)] == [2]
    assert r.draw(ax, INK, segment_budget=0) == []
    plt.close(fig)


def test_stroke_colors_follow_notebook_scheme():
    r = RainbowInkRenderer()
    first, last = r.stroke_color(0, 5), r.stroke_color(4, 5)
    # reversed rainbow: the first stroke gets the red end, the last the violet end
    assert first[0] > first[2] and last[2] > last[0]
    assert max(first) <= 0.65 + 1e-9  # HSV value scaled by 0.65


def test_plot_ink_frame_and_backdrop():
    img = Image.new("RGB", (224, 224), "white")
    ax = plot_ink(INK, image=img)
    assert ax.get_xlim() == (0, 224) and ax.get_ylim() == (224, 0)  # y points down
    ax2 = plot_ink(INK, canvas_size=None)  # fit to ink
    assert ax2.get_xlim()[0] == 0 and ax2.get_ylim()[0] > ax2.get_ylim()[1]
    plt.close("all")


def test_overlay_does_not_change_image_framing():
    img = Image.new("RGB", (100, 50), "white")
    ink = Ink([Stroke([(10, 10), (500, 400)])])  # runs far outside the image
    ax = plot_ink(ink, image=img, canvas_size=None)
    assert ax.get_xlim() == (-0.5, 99.5) and ax.get_ylim() == (49.5, -0.5)
    plt.close("all")


def test_style_options():
    style = StrokeStyle(outline_color=None, linewidth=3, colormap="viridis")
    fig, ax = plt.subplots()
    (lc, _) = RainbowInkRenderer(style).draw(ax, INK)
    assert lc.get_linewidth()[0] == 3 and not lc.get_path_effects()
    plt.close(fig)


def test_custom_renderer_plugs_into_plot_ink():
    @renderers.register("test-mono")
    class Mono(InkRenderer):
        def draw(self, ax, ink, *, segment_budget=None):
            return [ax.plot(*zip(*s))[0] for s in ink if len(s) > 1]

    try:
        ax = plot_ink(INK, renderer=renderers.create("test-mono"))
        assert len(ax.lines) == 2
    finally:
        renderers.unregister("test-mono")
        plt.close("all")


def test_animator_gif_has_one_frame_per_segment_plus_one(tmp_path):
    ink = Ink([Stroke([(10, 10), (50, 60), (90, 20)]), Stroke([(20, 150), (200, 160)])])
    out = InkAnimator(fps=5, dpi=40, figsize=(2, 2)).save(ink, tmp_path / "w.gif")
    with Image.open(out) as gif:
        assert gif.n_frames == 3 + 1  # 2 + 1 segments, plus the empty first frame
    with pytest.raises(ValueError):
        InkAnimator().save(ink, tmp_path / "w.xyz")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_animator_mp4(tmp_path):
    out = InkAnimator(fps=5, dpi=40, figsize=(2, 2)).save(INK, tmp_path / "w.mp4")
    assert out.stat().st_size > 0


def test_page_figure(page_image, two_words, tmp_path):
    result = PageInkPipeline(InkSightModel(FakeBackend()), FixedSegmenter(two_words)).run(page_image)
    assert adaptive_line_width(result) == pytest.approx((50 + 80) / 2 / 150)
    fig = plot_page_result(result, save_to=str(tmp_path / "page.png"), dpi=60)
    assert len(fig.axes) == 3 and (tmp_path / "page.png").stat().st_size > 0
    assert len(fig.axes[2].collections) == 2 and fig.axes[2].get_title().startswith("InkSight Result")
    ax = result.plot()  # PipelineResult.plot convenience
    assert len(ax.collections) == 2
    plt.close("all")


# ------------------------------------------------------------------------------ dataset
@pytest.fixture
def supp(tmp_path):
    root = tmp_path / "derendering_supp"
    img_dir = root / "HierText" / "images_sample"
    img_dir.mkdir(parents=True)
    for sample_id in ["png_1", "abc"]:  # 'png_1' would be mangled by the notebook's name.strip('.png')
        Image.new("RGB", (60, 30), "white").save(img_dir / f"{sample_id}.png")
        for mode in ("d+t", "r+d", "vanilla"):
            ink = Ink([Stroke([(1, 1), (100, 100)])], text=f"txt-{sample_id}")
            write_ink(ink, root / "small-p_HierText_inkml" / mode / f"{sample_id}.inkml")
    return SupplementaryDataset(root)


def test_dataset_paths_and_loading(supp):
    assert supp.is_available and supp.sample_ids(DatasetName.HIERTEXT) == ["abc", "png_1"]
    assert supp.sample_ids("IAM") == []
    folder = supp.inkml_dir(ModelVariant.SMALL_P, DatasetName.HIERTEXT, "d+t")
    assert folder.parts[-2:] == ("small-p_HierText_inkml", "d+t")
    ink = supp.load_ink("Small-p", "HierText", "r+d", "png_1")
    assert ink.text == "txt-png_1" and supp.load_image(DatasetName.HIERTEXT, "abc").size == (60, 30)
    assert len(supp.random_sample_ids(DatasetName.HIERTEXT, 2, seed=1)) == 2
    triples = list(supp.iter_samples(DatasetName.HIERTEXT, ModelVariant.SMALL_P, "vanilla"))
    assert [t[0] for t in triples] == ["abc", "png_1"]
    assert list(supp.iter_samples(DatasetName.HIERTEXT, ModelVariant.LARGE_I, "vanilla")) == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_render_videos_skips_existing(supp, tmp_path):
    animator = InkAnimator(fps=5, dpi=30, figsize=(1, 1))
    kwargs = dict(datasets=[DatasetName.HIERTEXT], models=[ModelVariant.SMALL_P],
                  modes=["vanilla"], animator=animator)
    first = supp.render_videos(tmp_path / "cache", **kwargs)
    assert sorted(p.name for p in first) == ["Small-p_HierText_vanilla_abc.mp4",
                                              "Small-p_HierText_vanilla_png_1.mp4"]
    assert supp.render_videos(tmp_path / "cache", **kwargs) == []


def test_download_is_skipped_when_present(supp, monkeypatch):
    monkeypatch.setattr("inksight_api.datasets.supplementary.download_file",
                        lambda *a, **k: pytest.fail("should not download"))
    assert supp.download() == supp.root


# The human-tracing file pickles the notebook's own Ink/Stroke classes (defined in __main__).
_WRITE_LEGACY = """
import sys, numpy as np
class Stroke:
    def __init__(self, pts):
        self.x = [p[0] for p in pts]; self.y = [p[1] for p in pts]
class Ink:
    def __init__(self, strokes): self.strokes = strokes
np.save(sys.argv[1], {"abc": Ink([Stroke([(1, 2), (3, 4)]), Stroke([(5, 6)])])}, allow_pickle=True)
"""


def test_human_traces_load_without_the_notebook_classes(supp):
    target = supp.root / "human_tracing_hash_to_ink.npy"
    subprocess.run([sys.executable, "-c", _WRITE_LEGACY, str(target)], check=True)
    inks = supp.human_traced_inks()
    assert list(inks) == ["abc"] and isinstance(inks["abc"], Ink)
    assert [list(s) for s in inks["abc"]] == [[(1.0, 2.0), (3.0, 4.0)], [(5.0, 6.0)]]


def test_human_traces_loader_refuses_other_classes(supp):
    import numpy as np

    target = supp.root / "human_tracing_hash_to_ink.npy"
    np.save(target, {"x": subprocess.CompletedProcess([], 0)}, allow_pickle=True)
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        supp.human_traced_inks()
